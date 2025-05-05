import math
import torch
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from .geometry import *
from .tools import coord_from, data_to_batch, to_device
from utils.protein.constants import BBDihed
from torch.utils.data._utils.collate import default_collate
from evaluate.metrics import *
from Bio.SVDSuperimposer import SVDSuperimposer


def get_next_structure(psi_cos, psi_sin, phi_cos, phi_sin):
    """
    based on reference peptide structure with 0 dihedrals, construct new reference with defined dihedrals
    :param psi_cos:
    :param psi_sin:
    :param phi_cos:
    :param phi_sin:
    :return:
        reference peptide dihedrals in the i th local frame coordinates
    """
    assert psi_cos.shape == psi_sin.shape and psi_cos.shape == phi_cos.shape and psi_cos.shape == phi_sin.shape

    point_dict = get_peptide_position()

    # 1. rotate C2 along CA2->N2 axis with psi_2
    direction_n_ca = F.normalize(point_dict['N2'] - point_dict['CA2'], dim=-1)
    mat4 = axis_angle_to_rotation_matrix(direction_n_ca, torch.stack([phi_cos, phi_sin], dim=-1),
                                         angle_type='tri')  # (K, 3, 3)
    point_dict['C2'] = point_dict['CA2'] + torch.matmul(mat4,
                                                        (point_dict['C2'] - point_dict['CA2']).unsqueeze(-1)).squeeze(
        -1)  # (K, 3)

    # 2. rotate along CA1, standard along C1->CA1
    direction_c_ca = F.normalize(point_dict['CA1'] - point_dict['C1'], dim=-1)
    mat5 = axis_angle_to_rotation_matrix(direction_c_ca, torch.stack([psi_cos, psi_sin], dim=-1),
                                         angle_type='tri')  # (K, 3, 3)
    point_dict['C2'] = point_dict['C1'] + torch.matmul(mat5,
                                                        (point_dict['C2'] - point_dict['C1']).unsqueeze(-1)).squeeze(
        -1)  # (K, 3)
    point_dict['CA2'] = point_dict['C1'] + torch.matmul(mat5,
                                                         (point_dict['CA2'] - point_dict['C1']).unsqueeze(-1)).squeeze(
        -1)  # (K, 3)
    point_dict['N2'] = point_dict['C1'] + torch.matmul(mat5,
                                                        (point_dict['N2'] - point_dict['C1']).unsqueeze(-1)).squeeze(
        -1)  # (K, 3)

    shape = point_dict['CA2'].shape
    point_dict['N1'] = point_dict['N1'].expand(shape)
    point_dict['CA1'] = point_dict['CA1'].expand(shape)
    point_dict['C1'] = point_dict['C1'].expand(shape)

    return point_dict

def generate_next_coord(last_coord, angle):
    """
		Args:
			last_coord: (..., 3, 3)
			angle: (..., 4)
	"""
    assert last_coord.shape[:-2:] == angle.shape[:-1], f'{last_coord.shape} vs {angle.shape}'
    x_last = last_coord[..., 0, :]  # (.., 3)
    mat = construct_3d_basis(last_coord[..., 0, :], last_coord[..., 1, :], last_coord[..., 2, :])  # (..., 3, 3)
    structure = get_next_structure(angle[..., 0], angle[..., 1], angle[..., 2], angle[..., 3])  # (K, 3)
    # back to global
    new_x0 = x_last + torch.matmul(mat, structure['CA2'].unsqueeze(-1)).squeeze(-1)
    new_x1 = x_last + torch.matmul(mat, structure['C2'].unsqueeze(-1)).squeeze(-1)
    new_x2 = x_last + torch.matmul(mat, structure['N2'].unsqueeze(-1)).squeeze(-1)
    new_pos_coord = torch.stack([new_x0, new_x1, new_x2], dim=-2)  # (..., 3, 3)
    return new_pos_coord



class AutoregressiveSampler_new():
    def __init__(self, data, prediction_model, device='cpu') -> None:
        super().__init__()
        self.data = data
        self.prediction_model = prediction_model # model.eval()
        self.device = device

    def _get_metric(self, pep, clip=False):
        pred = {'aa':pep['aa'],'coord':pep['coord'], 'length':len(pep['aa'])}
        gt = {'aa':self.data['pep_aa'],'coord':self.data['pep_coords'], 'length':len(self.data['pep_aa'])}
        # init


        if clip == False:
            if gt['length'] >= pred['length']:
                delta = gt['length'] - pred['length']
                pred_aa = torch.cat([pred['aa'], torch.ones(delta, ) * -1], dim=0)
                pred_pos = torch.concat([pred['coord'], torch.zeros(delta, 3, 3)], dim=0)
                gt_aa = gt['aa']
                gt_pos = gt['coord']
            else:
                delta = pred['length'] - gt['length']
                gt_aa = torch.cat([gt['aa'], torch.ones(delta, ) * -1], dim=0)
                gt_pos = torch.concat([gt['coord'], torch.zeros(delta, 3, 3)], dim=0)
                pred_aa = pred['aa']
                pred_pos = pred['coord']
        else:
            # clip
            if gt['length'] >= pred['length']:
                delta = gt['length'] - pred['length']
                gt_aa = gt['aa'][:pred['length']]
                gt_pos = gt['coord'][:pred['length']]
                pred_aa = pred['aa']
                pred_pos = pred['coord']
            else:
                delta = pred['length'] - gt['length']
                pred_aa = pred['aa'][:gt['length']]
                pred_pos = pred['coord'][:gt['length']]
                gt_aa = gt['aa']
                gt_pos = gt['coord']

        # seq recovery
        recovery = (pred_aa == gt_aa).float().mean().item()

        # rmsd
        sup = SVDSuperimposer()
        sup.set(gt_pos.view(-1, 3).clone().detach().numpy(), pred_pos.view(-1, 3).clone().detach().numpy())
        sup.run()
        rmsd = float(sup.get_rms())

        return recovery, rmsd

    def _sample_next(self, coord_all, type_all):
        """
         using receptor information and generated peptide information for the next residue
        """
        # prepare data
        aa = torch.cat([self.data['rec_aa'],type_all],dim=0)

        data_query = {
            'rec_coords' : self.data['rec_coords'].unsqueeze(0), # (B,L,3,3)
            'pep_coords': coord_all.unsqueeze(0), # (B,L,3,3)
            'is_peptide':torch.tensor([False]*len(self.data['rec_aa']) + [True]*coord_all.shape[0]).unsqueeze(0),
            'aa':aa.unsqueeze(0), # (B,L)
            'mask':(aa!=20).bool().unsqueeze(0) # (B,L)
        }
        batch_data = to_device(data_query, self.device)


        # forward
        with torch.no_grad():
            features = self.prediction_model._batch_to_feature(batch_data)
            type_logits, type_angles = self.prediction_model.sample(*features)  # (1,L,21), (1,L,4)

        # get label
        type_logits,type_angles = type_logits.squeeze(0).cpu(),type_angles.squeeze(0).cpu()


        # TODO: fix this?
        prob = F.softmax(type_logits, dim=-1)  # (21, )
        type_new = prob[:20].multinomial(1).squeeze(-1) # (1,)
        #type_new = torch.argmax(prob[:20])  # (1,)

        coord_new = generate_next_coord(coord_all[-1], type_angles)  # (3, 3), (4,)

        return coord_new, type_new, type_logits.unsqueeze(0)

    def _sample_peptide(self, L, coord_init, aa_init):
        coord_all = coord_init.clone().detach().unsqueeze(0) # (L,3,3)
        type_all = aa_init.clone().detach() # (L)
        logits_all = F.one_hot(aa_init, num_classes=21).view(-1).unsqueeze(0) # (L,21)

        for l in range(0, L):
            if l == 0:
                continue
            else:
                new_coord, new_aa, new_logits = self._sample_next(coord_all, type_all)  # use prediction model
                # if new_logits[:,20].item() >= 0.5:
                #     break
                coord_all = torch.cat([coord_all, new_coord.view(-1, 3, 3)], dim=0)  # (L + 1, 3, 3)
                type_all = torch.cat([type_all, new_aa.reshape(-1)], dim=0)  # (L + 1, )
                logits_all = torch.cat([logits_all, new_logits], dim=0) # (L+1, 21)

        return coord_all, type_all, logits_all

    def sample(self, N, L):
        peptide_list = []
        logits_list = []
        metrics = {'recovery': [1.], 'rmsd': [0.], 'recovery_clip': [1.],
                   'rmsd_clip': [0.],'length':[len(self.data['pep_aa'])]}


        # generate
        for i in range(N):
            rand_idx = 0
            coord_seed = self.data['pep_coords'][rand_idx] # (3,3)
            aa_seed = self.data['pep_aa'][rand_idx].view(-1) # (1,)

            # Start generation
            coord_all, type_all, logtis_all = self._sample_peptide(L, coord_seed, aa_seed)
            peptide_list.append({
                'aa': type_all,
                'coord': coord_all,
                'is_peptide': torch.tensor([True] * len(type_all)),
            })
            logits_list.append(logtis_all)

        # metric
        for i,pep in enumerate(peptide_list):
            recovery, rmsd = self._get_metric(pep,False)
            metrics['recovery'].append(recovery)
            metrics['rmsd'].append(rmsd)
            metrics['length'].append(len(pep['aa']))
            recovery_clip, rmsd_clip = self._get_metric(pep, True)
            metrics['recovery_clip'].append(recovery_clip)
            metrics['rmsd_clip'].append(rmsd_clip)
            #metrics['ppl'] = torch.exp(F.cross_entropy(logits_list[i][1:], self.data['rec_aa'][1:len(pep['aa'])])).item()

        return peptide_list, torch.stack(logits_list,dim=0), metrics
