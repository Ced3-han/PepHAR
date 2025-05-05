from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
import torch.nn.functional as F
from ._base import register_transform


@register_transform('to_aa_density')
class ToAADensityTransform():
    def __init__(self):
        pass

    def __call__(self, data):
        assert isinstance(data, dict)
        N = data['pos_heavyatom'][:, 0, :]
        CA = data['pos_heavyatom'][:, 1, :]
        C = data['pos_heavyatom'][:, 2, :]
        return {
            'aa': data['aa'],
            'coord': torch.stack([CA, C, N], dim=1),  # (L, 3, 3)
            'is_query': data['is_peptide'],
        }


@register_transform('to_aa_density_new')
class ToAADensityTransform_new():
    def __init__(self):
        pass

    def __call__(self, data):
        pep_length = len(data['pep_aa'])
        pep_coord = data['pep_pos_heavyatom'][:,[1,2,0]].squeeze(1)
        rec_coord = data['rec_pos_heavyatom'][:,[1,2,0]].squeeze(1)
        coord = torch.cat([rec_coord,pep_coord],dim=0)
        aa = torch.cat([data['rec_aa'],data['pep_aa']],dim=0).long()
        is_query = torch.cat([torch.zeros(len(data['rec_aa']),),torch.ones(pep_length,)],dim=0).bool()
        return {
            'aa':aa,
            'coord':coord,
            'is_query':is_query,
        }


@register_transform('to_aa_density_new_first')
class ToAADensityTransform_new_first():
    def __init__(self):
        pass

    def __call__(self, data):
        pep_length = len(data['pep_aa'])
        pep_coord = data['pep_pos_heavyatom'][:,[1,2,0]].squeeze(1)
        rec_coord = data['rec_pos_heavyatom'][:,[1,2,0]].squeeze(1)

        coord = torch.cat([rec_coord,pep_coord[0].unsqueeze(0)],dim=0)
        aa = torch.cat([data['rec_aa'],data['pep_aa'][0].unsqueeze(0)],dim=0).long()
        is_query = torch.cat([torch.zeros(len(data['rec_aa']),),torch.ones(1,)],dim=0).bool()
        return {
            'aa':aa,
            'coord':coord,
            'is_query':is_query,
        }


@register_transform('to_aa_density_v3')
class ToAADensityTransformV3():
    def __init__(self):
        pass

    def __call__(self, data):
        return {
            'rec_coords': data['rec_pos_heavyatom'][:,[[1,2,0]]].squeeze(1),
            'pep_coords': data['pep_pos_heavyatom'][:,[[1,2,0]]].squeeze(1), # CA, C, N
            'rec_aa': data['rec_aa'], # include for sample
            'pep_aa': torch.full_like(data['pep_aa'], 20),
            'rec_mask': data['rec_aa'] != 20,
            'pep_mask': data['pep_aa'] != 20,
            
            'label': data['pep_aa'],
            'label_mask': data['pep_aa'] != 20,
        }


@register_transform('to_aa_density_v4')
class ToAADensityTransformV4():
    def __init__(self):
        pass

    def _get_mask(self, data):
        LP = data['pep_aa'].shape[0]
        U = torch.rand(1).item()
        K = int(LP * U)
        indices = torch.randperm(LP)
        hidden_mask = torch.cat([torch.ones(K), torch.zeros(LP - K)], dim=0).bool()[indices]
        return hidden_mask

    def _get_order(self, hidden_mask):
        pep_order = torch.randperm((~hidden_mask).sum().item())
        return pep_order
    
    def __call__(self, data):
        pep_coord = data['pep_coord']
        pep_aa = data['pep_aa']

        hidden_mask = self._get_mask(data)
        pep_order = self._get_order(hidden_mask)
        qry_coord = pep_coord[hidden_mask]
        label_class = pep_aa[hidden_mask]
        qry_aa = torch.zeros(qry_coord.shape[0]).long()
        label_class_mask = torch.ones(qry_coord.shape[0]).bool()
        
        pep_coord = pep_coord[~hidden_mask]
        pep_aa = pep_aa[~hidden_mask]


        return {
            # coord
            'rec_coord': data['rec_coord'],
            'rec_aa': data['rec_aa'], 
            'pep_coord': pep_coord[pep_order], 
            'pep_aa': pep_aa[pep_order],
            'qry_coord': qry_coord,
            'qry_aa': qry_aa,

            'label_class': label_class,
            'label_class_mask': label_class_mask,
            
            'pep_mask_heavyatom': data['pep_mask_heavyatom'],
            'pep_pos_heavyatom': data['pep_pos_heavyatom'],
        }
