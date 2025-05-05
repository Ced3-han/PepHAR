from tkinter import N
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from datasets.padding_collate import PaddingCollate
from evaluate.geometry import rotation_matrix_to_quaternion
from modules.common.geometry import *
from ._base import register_transform
from utils.protein.constants import BBHeavyAtom, BBDihed


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)

def dihedral_angle(v1, v2, v3, v4):
    """
    Args:
        v1: (..., 3)
        v2: (..., 3)
        v3: (..., 3)
        v4: (..., 3)
    Returns:
        (...)
    """
    # TODO: require tests
    v12 = normalize_vector(v1 - v2, dim=-1)
    v23 = normalize_vector(v2 - v3, dim=-1)
    v34 = normalize_vector(v3 - v4, dim=-1)
    v12_x_v23 = normalize_vector(torch.cross(v12, v23, dim=-1), dim=-1)
    v23_x_v34 = normalize_vector(torch.cross(v23, v34, dim=-1), dim=-1)
    cos_theta = torch.sum(v12_x_v23 * v23_x_v34, dim=-1)
    theta = torch.acos(cos_theta)
    sign = torch.sign(torch.sum(v12_x_v23 * v34, dim=-1))
    return theta * sign


@register_transform('to_aa_prediction')
class ToAAPredicitonTask():
    def __init__(self):
        pass
        #self.rng = torch.Generator().manual_seed(42)

    def _get_segment_of_peptide(self, data):
        res_nb = data['res_nb']
        is_peptide = data['is_peptide']
        peptide_nb = res_nb[is_peptide].sort()[0]
        available_pair = torch.isin(peptide_nb + 1, peptide_nb)
        available_nb = peptide_nb[available_pair]
        if len(available_nb) == 0:
            return None
        # peptide_idx = available_nb[torch.randint(len(available_nb), (1,), generator=self.rng)]
        # left_idx = peptide_nb[torch.randint((peptide_nb <= peptide_idx).sum(), (1,), generator=self.rng)]
        # should not include predefined random
        peptide_idx = available_nb[torch.randint(len(available_nb), (1,))]
        left_idx = peptide_nb[torch.randint((peptide_nb <= peptide_idx).sum(), (1,))]
        keep_peptide_nb = peptide_nb[(left_idx <= peptide_nb) & (peptide_nb <= peptide_idx)]
        keep_peptide = (torch.isin(res_nb, keep_peptide_nb) & is_peptide)
        assert keep_peptide.sum() == len(keep_peptide_nb)
        idx1 = ((res_nb == peptide_idx) & (is_peptide)).nonzero()
        idx2 = ((res_nb == peptide_idx + 1) & (is_peptide)).nonzero()
        assert len(keep_peptide) > 0
        assert len(idx1) == 1 and len(idx2) == 1, f'idx1: {idx1}, idx2: {idx2}'
        return keep_peptide, idx1.item(), idx2.item()

    # def _get_dihedral_angle(self, pos_heavyatom1, pos_heavyatom2):
    #     N1, CA1, C1 = pos_heavyatom1[0, :], pos_heavyatom1[1, :], pos_heavyatom1[2, :]
    #     N2, CA2, C2 = pos_heavyatom2[0, :], pos_heavyatom2[1, :], pos_heavyatom2[2, :]
    #     # psi = dihedral_angle(N1, C1, CA1, N2)
    #     # phi = dihedral_angle(C2, CA2, N2, C1)， 注意到顺序有影响
    #     psi_1 = dihedral_from_four_points(N1, CA1, C1, N2) # from geometry package
    #     phi_2 = dihedral_from_four_points(C1, N2, CA2, C2)
    #     return psi_1, phi_2

    def _get_dihedral_angle(self, pos_heavyatom1, pos_heavyatom2):
        N1, CA1, C1 = pos_heavyatom1[0, :], pos_heavyatom1[1, :], pos_heavyatom1[2, :]
        N2, CA2, C2 = pos_heavyatom2[0, :], pos_heavyatom2[1, :], pos_heavyatom2[2, :]
        psi = dihedral_angle(N1, C1, CA1, N2)
        phi = dihedral_angle(C2, CA2, N2, C1)
        return psi, phi

    def __call__(self, data):
        assert isinstance(data, dict)
        pos_heavyatom = data['pos_heavyatom']
        N = pos_heavyatom[:, 0, :]
        CA = pos_heavyatom[:, 1, :]
        C = pos_heavyatom[:, 2, :]
        aa = data['aa']
        pos_coord = torch.stack([CA, C, N], dim=1)  # (L, 3, 3)
        is_peptide = data['is_peptide']

        res = self._get_segment_of_peptide(data)
        if res is None:
            anchor = torch.tensor(-1, dtype=torch.int64)
            label_angle = torch.zeros(4, dtype=torch.float32)
            label_type = torch.tensor(0, dtype=torch.int64)
        else:
            keep_peptide, idx1, idx2 = res
            keep_idx = keep_peptide | (~is_peptide)
            aa, pos_coord, is_peptide = aa[keep_idx], pos_coord[keep_idx], is_peptide[keep_idx]
            assert len(aa) == (~data['is_peptide']).sum() + keep_peptide.sum()
            anchor = torch.cumsum(keep_idx, dim=0)[idx1] - 1
            psi, phi = self._get_dihedral_angle(data['pos_heavyatom'][idx1], data['pos_heavyatom'][idx2])
            label_type = data['aa'][idx2]
            label_angle = torch.stack([torch.cos(psi), torch.sin(psi), torch.cos(phi), torch.sin(phi)], dim=0)
        return {
            'aa': aa,
            'coord': pos_coord,
            'is_peptide': is_peptide,
            'anchor': anchor,
            'label_angle': label_angle,
            'label_type': label_type,
        }



@register_transform('to_aa_prediction_new')
class ToAAPredicitonTaskV2():
    def __init__(self):
        pass

    def __call__(self, data):
        # based on i, predict psi_i and phi_i+1 and aa_i+1
        # psi i
        psi = data['pep_dihed'][:,BBDihed.PSI]
        psi_mask = data['pep_dihed_mask'][:,BBDihed.PSI]
        # phi i+1
        phi = F.pad(data['pep_dihed'][1:,BBDihed.PHI],(0,1),value=0)
        phi_mask = F.pad(data['pep_dihed_mask'][1:,BBDihed.PHI],(0,1),value=False)
        # label angles
        label_angles = torch.stack([torch.cos(psi),torch.sin(psi),torch.cos(phi),torch.sin(phi)],dim=-1)
        label_angles_mask = torch.stack([psi_mask,psi_mask,phi_mask,phi_mask],dim=-1).bool()
        label_types = data['pep_aa']
        label_types = F.pad(label_types[1:],(0,1),value=20)
        label_types_mask = (data['pep_aa']!=20).bool()
        label_types_mask = F.pad(label_types_mask[1:],(0,1),value=False)
        # others
        is_peptide = torch.cat([torch.zeros_like(data['rec_tag']),torch.ones_like(data['pep_tag'])],dim=0).bool()
        aa = torch.cat([data['rec_aa'],data['pep_aa']],dim=0)
        mask = (aa!=20).bool()
        # coord
        return {
            'pep_coords':data['pep_pos_heavyatom'][:,[[1,2,0]]].squeeze(1), # CA, C, N
            'rec_coords':data['rec_pos_heavyatom'][:,[[1,2,0]]].squeeze(1),
            'pep_aa':data['pep_aa'],
            'rec_aa':data['rec_aa'], # include for sample
            'label_angles':label_angles,
            'label_angles_mask':label_angles_mask,
            'label_types':label_types,
            'label_types_mask':label_types_mask,
            'mask':mask,
            'is_peptide':is_peptide,
            'aa':aa
        }

class ShufflePeptide():
    def __init__(self):
        pass
    
    @staticmethod
    def _assert_order(data):
        try:
            LV = (data['pep_aa'] != 20).sum()
            for key in ['label_angles_mask', 'label_types_mask']:
                assert (~data[key][LV:]).all(), 'the valid amino acid should be in the front'
            for key in ['pep_aa']:
                assert (data[key][:LV] < 20).all() and (data[key][LV:] == 20).all(), 'the valid amino acid should be in the front'
        except AssertionError as e:
            print('[DEBUG]', LV, data['pep_aa'])
            raise e

    def __call__(self, data):
        self._assert_order(data)
        pep_mask = data['pep_aa'] != 20
        LP = pep_mask.shape[0]
        LV = pep_mask.sum()
        pep_order = torch.cat([torch.randperm(LV), torch.arange(LV, LP)], dim=0)
        # print('order', pep_order)
        # print('aa', data['pep_aa'])
        # print('label_angles_mask', data['label_angles_mask'])

        aa = torch.cat([data['rec_aa'], data['pep_aa'][pep_order]],dim=0)
        mask = (aa != 20).bool()
        ret = {
            'pep_coords': data['pep_coords'][pep_order],
            'rec_coords': data['rec_coords'],
            'pep_aa': data['pep_aa'][pep_order],
            'rec_aa': data['rec_aa'], # include for sample

            'label_angles': data['label_angles'][pep_order], 
            'label_angles_mask': data['label_angles_mask'][pep_order],
            'label_types': data['label_types'][pep_order],
            'label_types_mask': data['label_types_mask'][pep_order],

            'pep_dihed': data['pep_dihed'][pep_order],
            'pep_dihed_mask': data['pep_dihed_mask'][pep_order],

            'mask': mask,
            'is_peptide': data['is_peptide'],
            'aa': aa,
        }
        
        self._assert_order(ret)
        return ret


@register_transform('to_aa_prediction_v3')
class ToAAPredicitonTaskV3():
    def __init__(self, train_bidirection=True):
        self.train_bidirection = train_bidirection


    def __call__(self, data):
        # based on i, predict psi_i and phi_i+1 and aa_i+1
        # psi i
        
        LP = data['pep_aa'].shape[0]
        LV = (data['pep_aa'] != 20).sum()

        psi = data['pep_dihed'][:, BBDihed.PSI]
        phi = data['pep_dihed'][:,BBDihed.PHI]
        psi_mask = data['pep_dihed_mask'][:, BBDihed.PSI]
        phi_mask = data['pep_dihed_mask'][:,BBDihed.PHI]

        next_angles = psi, F.pad(phi[1:], (0, 1), value=0)
        next_angles_mask = psi_mask, F.pad(phi_mask[1:], (0, 1), value=False)
        prev_angles = F.pad(psi[:LV - 1], (1, LP - LV), value=0), phi
        prev_angles_mask = F.pad(psi_mask[:LV - 1], (1, LP - LV), value=False), phi_mask
        assert (prev_angles_mask[0] == prev_angles_mask[1]).all()
        assert (next_angles_mask[0] == next_angles_mask[1]).all()
        assert prev_angles[0].shape == (LP, )
        assert prev_angles_mask[0].shape == (LP, )


        label_next_angles = torch.stack([torch.cos(next_angles[0]), torch.sin(next_angles[0]), torch.cos(next_angles[1]), torch.sin(next_angles[1])], dim=-1)
        label_next_angles_mask = torch.stack([next_angles_mask[0], next_angles_mask[0], next_angles_mask[1], next_angles_mask[1]], dim=-1)
        label_prev_angles = torch.stack([torch.cos(prev_angles[0]), torch.sin(prev_angles[0]), torch.cos(prev_angles[1]), torch.sin(prev_angles[1])], dim=-1)
        label_prev_angles_mask = torch.stack([prev_angles_mask[0], prev_angles_mask[0], prev_angles_mask[1], prev_angles_mask[1]], dim=-1)

        # label angles
        types = data['pep_aa']
        types_mask = (data['pep_aa']!=20).bool()

        label_next_types = F.pad(types[1:], (0, 1), value=20)
        label_next_types_mask = F.pad(types_mask[1:], (0, 1), value=False)
        label_prev_types = F.pad(types[:LV - 1], (1, LP - LV), value=20)
        label_prev_types_mask = F.pad(types_mask[:LV - 1], (1, LP - LV), value=False)
        assert label_prev_types.shape == (LP,)
        assert label_prev_types_mask.shape == (LP,)

        # others
        is_peptide = torch.cat([torch.zeros_like(data['rec_tag']), torch.ones_like(data['pep_tag'])],dim=0).bool()
        aa = torch.cat([data['rec_aa'], data['pep_aa']],dim=0)
        mask = (aa!=20).bool()

        ret = {
            # coord
            'pep_coords': data['pep_pos_heavyatom'][:,[[1,2,0]]].squeeze(1), # CA, C, N
            'rec_coords': data['rec_pos_heavyatom'][:,[[1,2,0]]].squeeze(1),
            'pep_aa': data['pep_aa'],
            'rec_aa': data['rec_aa'], # include for sample

            'label_angles': torch.stack([label_prev_angles, label_next_angles], dim=1),  # (L, 2, 4)
            'label_angles_mask': torch.stack([label_prev_angles_mask, label_next_angles_mask], dim=1),  # (L, 2, 4)
            'label_types': torch.stack([label_prev_types, label_next_types], dim=1),  # (L, 2)
            'label_types_mask': torch.stack([label_prev_types_mask, label_next_types_mask], dim=1),  # (L, 2)

            'pep_dihed': data['pep_dihed'],
            'pep_dihed_mask': data['pep_dihed_mask'],
            'mask': mask,
            'is_peptide': is_peptide,
            'aa': aa,
        }
        if self.train_bidirection:
            ret = ShufflePeptide()(ret)
        # else:
        #     ret['label_angles_mask'] = torch.stack([torch.zeros_like(label_prev_angles_mask).bool(), label_next_angles_mask], dim=1)
        #     ret['label_types_mask'] = torch.stack([torch.zeros_like(label_prev_types_mask).bool(), label_next_types_mask], dim=1)
        
        return ret


@register_transform('to_aa_prediction_v4')
class ToAAPredicitonTaskV4():
    def __init__(self, enable_mask=True):
        self.enable_mask = enable_mask

    def _get_label(self, data):
        # pep_coord = data['pep_coord']
        LP = data['pep_aa'].shape[0]
        LV = (data['pep_aa'] != 20).sum()
        
        # # dihedral_from_four_points(N1, CA1, C1, N2)
        # psi = dihedral_from_four_points(pep_coord[:-1, 2], pep_coord[:-1, 0], pep_coord[:-1, 1], pep_coord[1:, 2])
        # # dihedral_from_four_points(C1, N2, CA2, C2)
        # phi = dihedral_from_four_points(pep_coord[:-1, 1], pep_coord[1:, 2], pep_coord[1:, 0], pep_coord[1:, 1])
        # psi = F.pad(psi, (0, 1), value=0.0)
        # phi = F.pad(phi, (1, 0), value=0.0)
        # psi_mask = torch.cat([torch.ones(LP - 1), torch.zeros(1)]).bool()
        # phi_mask = torch.cat([torch.zeros(1), torch.ones(LP - 1)]).bool()

        psi = data['pep_dihed'][:, BBDihed.PSI]
        phi = data['pep_dihed'][:,BBDihed.PHI]
        psi_mask = data['pep_dihed_mask'][:, BBDihed.PSI]
        phi_mask = data['pep_dihed_mask'][:,BBDihed.PHI]

        # print((psi - psi_gt).cos(), (phi - phi_gt).cos())
        # # assert (psi - psi_gt).cos().abs().max() < 0.1
        # # assert (phi - phi_gt).cos().abs().max() < 0.1
        # assert (psi_mask == psi_mask_gt).all()
        # assert (phi_mask == phi_mask_gt).all()

        prev_angles = torch.stack([F.pad(psi[:LV - 1], (1, LP - LV), value=0), phi], dim=-1)
        next_angles = torch.stack([psi, F.pad(phi[1:], (0, 1), value=0)], dim=-1)
        prev_angles_mask = torch.stack([F.pad(psi_mask[:LV - 1], (1, LP - LV), value=False), phi_mask], dim=-1)
        next_angles_mask = torch.stack([psi_mask, F.pad(phi_mask[1:], (0, 1), value=False)], dim=-1)
        assert next_angles.shape == (LP, 2)
        assert prev_angles.shape == (LP, 2)
        assert next_angles_mask.shape == (LP, 2)
        assert prev_angles_mask.shape == (LP, 2)
        

        label_angles = torch.stack([prev_angles, next_angles], dim=1)  # (L, 2, 2, 2)
        label_angles_mask = torch.stack([prev_angles_mask, next_angles_mask], dim=1)  # (L, 2, 2, 2)
        assert label_angles.shape == (LP, 2, 2)
        assert label_angles_mask.shape == (LP, 2, 2)

        return label_angles, label_angles_mask

    def _get_mask(self, data):
        LP = data['pep_aa'].shape[0]
        U = torch.rand(1).item() if self.enable_mask else 0.0
        K = int(LP * U)
        indices = torch.randperm(LP)
        hidden_mask = torch.cat([torch.ones(K), torch.zeros(LP - K)], dim=0).bool()[indices]
        return hidden_mask

    def _get_order(self, hidden_mask):
        if self.enable_mask:
            pep_order = torch.randperm((~hidden_mask).sum().item())
        else:
            pep_order = torch.arange(hidden_mask.shape[0])
        return pep_order

    def __call__(self, data):
        pep_coord = data['pep_coord']
        pep_aa = data['pep_aa']
        qry_coord = pep_coord
        qry_aa = pep_aa

        hidden_mask = self._get_mask(data)
        pep_order = self._get_order(hidden_mask)
        pep_coord = pep_coord[~hidden_mask]
        pep_aa = pep_aa[~hidden_mask]
        
        LP = data['pep_aa'].shape[0]
        label_angle, label_angle_mask = self._get_label(data)
        label_known = torch.zeros(LP, 2, 2).bool()
        for i in range(LP):
            if i + 1 < LP and not hidden_mask[i + 1]:
                label_known[i, 1, 0] = True
                label_known[i, 1, 1] = True
            if i - 1 >= 0 and not hidden_mask[i - 1]:
                label_known[i, 0, 0] = True
                label_known[i, 0, 1] = True

        return {
            # coord
            
            'rec_coord': data['rec_coord'],
            'rec_aa': data['rec_aa'], 
            'pep_coord': pep_coord[pep_order], 
            'pep_aa': pep_aa[pep_order],
            'qry_coord': qry_coord,
            'qry_aa': qry_aa,

            'label_angle': label_angle,
            'label_angle_mask': label_angle_mask,
            'label_known': label_known,
        }



@register_transform('to_aa_prediction_abl_pos')
class ToAAPredicitonTaskAblPos():
    def __init__(self, enable_mask=True):
        self.enable_mask = enable_mask

    def _get_label(self, data):
        # pep_coord = data['pep_coord']
        LP = data['pep_aa'].shape[0]
        LV = (data['pep_aa'] != 20).sum()
        assert LP == LV
        
        x = data['pep_coord'][:, 0]
        rot = construct_3d_basis(data['pep_coord'][:, 0], data['pep_coord'][:, 1], data['pep_coord'][:, 2])
        o = rotation_matrix_to_quaternion(rot)
        xo = torch.concat([x, o], dim=-1)
        # print(x.shape, o.shape, xo.shape)
        # print(x, o)
        xo_mask = torch.ones(xo.shape).bool()
        assert xo.shape == (LP, 7)
        assert xo_mask.shape == (LP, 7)
        

        prev_angles = F.pad(xo[:LV - 1], (0, 0, 1, 0), value=0)
        next_angles = F.pad(xo[1:], (0, 0, 0, 1), value=0)
        prev_angles_mask = F.pad(xo_mask[:LV - 1], (0, 0, 1, 0), value=False)
        next_angles_mask = F.pad(xo_mask[1:], (0, 0, 0, 1), value=False)
        assert next_angles.shape == (LP, 7), f'{next_angles.shape}'
        assert prev_angles.shape == (LP, 7), f'{prev_angles.shape}'
        assert next_angles_mask.shape == (LP, 7), f'{next_angles_mask.shape}'
        assert prev_angles_mask.shape == (LP, 7), f'{prev_angles_mask.shape}'
        
        label_angles = torch.stack([prev_angles, next_angles], dim=1)  # (L, 2, 2, 2)
        label_angles_mask = torch.stack([prev_angles_mask, next_angles_mask], dim=1)  # (L, 2, 2, 2)
        assert label_angles.shape == (LP, 2, 7)
        assert label_angles_mask.shape == (LP, 2, 7)

        return label_angles, label_angles_mask

    def _get_mask(self, data):
        LP = data['pep_aa'].shape[0]
        U = torch.rand(1).item() if self.enable_mask else 0.0
        K = int(LP * U)
        indices = torch.randperm(LP)
        hidden_mask = torch.cat([torch.ones(K), torch.zeros(LP - K)], dim=0).bool()[indices]
        return hidden_mask

    def _get_order(self, hidden_mask):
        if self.enable_mask:
            pep_order = torch.randperm((~hidden_mask).sum().item())
        else:
            pep_order = torch.arange(hidden_mask.shape[0])
        return pep_order

    def __call__(self, data):
        pep_coord = data['pep_coord']
        pep_aa = data['pep_aa']
        qry_coord = pep_coord
        qry_aa = pep_aa

        hidden_mask = self._get_mask(data)
        pep_order = self._get_order(hidden_mask)
        pep_coord = pep_coord[~hidden_mask]
        pep_aa = pep_aa[~hidden_mask]
        
        LP = data['pep_aa'].shape[0]
        label_angle, label_angle_mask = self._get_label(data)
        label_known = torch.zeros(LP, 2).bool()
        for i in range(LP):
            if i + 1 < LP and not hidden_mask[i + 1]:
                label_known[i, 1] = True
            if i - 1 >= 0 and not hidden_mask[i - 1]:
                label_known[i, 0] = True

        return {
            # coord
            
            'rec_coord': data['rec_coord'],
            'rec_aa': data['rec_aa'], 
            'pep_coord': pep_coord[pep_order], 
            'pep_aa': pep_aa[pep_order],
            'qry_coord': qry_coord,
            'qry_aa': qry_aa,

            'label_angle': label_angle,
            'label_angle_mask': label_angle_mask,
            'label_known': label_known,
        }

