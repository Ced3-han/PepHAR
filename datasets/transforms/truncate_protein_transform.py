import torch
from ._base import register_transform
from utils.protein.constants import BBHeavyAtom


def select(v, index):
    if isinstance(v, list):
        return [v[i] for i in index]
    elif isinstance(v, torch.Tensor):
        return v[index]
    else:
        raise NotImplementedError


@register_transform('truncate_protein')
class TruncateProtein():
    """
    Truncate the receptor and change the order of peptide saved in order.
    """
    def __init__(self, max_length=256, reserve_ratio=0.0):
        self.max_length = max_length
        self.reserve_ratio = reserve_ratio

    def __call__(self, data):
        assert isinstance(data, dict)
        aa, pos_heavyatom, is_peptide = data['aa'], data['pos_heavyatom'], data['is_peptide']
        size = aa.shape[0]
        num_peptide = is_peptide.sum().item()
        num_reserve = int(self.reserve_ratio * num_peptide)
        if size + num_reserve > self.max_length:
            num_remain = self.max_length - num_reserve
            ca_position = pos_heavyatom[:, 1, :]  # (L, 3)
            distance_map = ca_position.unsqueeze(1) - ca_position[is_peptide].unsqueeze(0)  # (L, 1, 3) - (1, Lq, 3) = (L, Lq, 3)
            distance_map = (distance_map ** 2).sum(dim=-1)  # (L, Lq)
            distance = distance_map.min(dim=-1)[0]  # (L, )
            nearest_neighbor_index = distance.argsort()  # (L, )
            nearest_neighbor_index = nearest_neighbor_index[:num_remain]  # (num_remain, )
            data = {k: select(v, nearest_neighbor_index) for k, v in data.items() if k!= 'name'}

        order = torch.argsort(data['chain_nb'] * (2 ** 20) + data['res_nb'])
        data = {k: select(v, order) for k, v in data.items() if k!= 'name'}
        return data

@register_transform('truncate_protein_new')
class TruncateProtein_v2():
    """
    Truncate the receptor and change the order of peptide saved in order.
    the total pair length is 224 + 32 pad all!
    so we don't need to padding again in the collate function in dataloader
    """
    def __init__(self, rec_length=224, pep_length=32):
        self.rec_length = rec_length
        self.pep_length = pep_length
        self.pad_dict = {'pep_aa':20, 'pep_pos_heavyatom':0., 'pep_mask_heavyatom':False,
                         'rec_aa':20, 'rec_pos_heavyatom':0., 'rec_mask_heavyatom':False,
                         'pep_tag':False, 'rec_tag':False, 'pep_dihed':0., 'pep_dihed_mask':False}

    def _pad_last(self, x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    def __call__(self, data):
        data_tmp = data.copy() # don;t modify dict in inner function!
        if len(data['rec_aa']) < 224:
            for k,v in data.items():
                if k in self.pad_dict and k.startswith('rec'):
                    data_tmp[k] = self._pad_last(data[k],224,value=self.pad_dict[k])
        else:
            # length = self.max_length - len(data['pep_aa'])
            dist_cb = torch.cdist(data['pep_pos_heavyatom'][:,BBHeavyAtom.CB],
                                    data['rec_pos_heavyatom'][:,BBHeavyAtom.CB]) # choose cb here
            dist_cb = dist_cb.min(dim=0)[0] # along pep dim
            rec_idx = dist_cb.argsort()[:self.rec_length]
            for k,v in data.items():
                if k.startswith('rec'):
                    data_tmp[k] = select(v,rec_idx)

        if len(data['pep_aa']) < 32:
            for k,v in data.items():
                if k in self.pad_dict and k.startswith('pep'):
                    data_tmp[k] = self._pad_last(data[k],32,value=self.pad_dict[k])

        return data_tmp



@register_transform('truncate_receptor')
class TruncateReceptor():
    """
    Truncate receptor to 224 if receptor is longer than 224
    """
    def __init__(self, rec_length=224):
        self.rec_length = rec_length
        self.pad_dict = {'rec_aa':20, 'rec_pos_heavyatom':0., 'rec_mask_heavyatom':False,
                         'rec_tag':False}

    def _pad_last(self, x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    def __call__(self, data):
        data_tmp = data.copy() # don't modify dict in inner function!

        if len(data['rec_aa']) > self.rec_length:
            dist_cb = torch.cdist(data['pep_pos_heavyatom'][:,BBHeavyAtom.CB],
                                    data['rec_pos_heavyatom'][:,BBHeavyAtom.CB]) # choose cb here
            dist_cb = dist_cb.min(dim=0)[0] # along pep dim
            rec_idx = dist_cb.argsort()[:self.rec_length]
            for k,v in data.items():
                if k.startswith('rec'):
                    data_tmp[k] = select(v,rec_idx)

        return data_tmp



@register_transform('truncate_protein_v3')
class TruncateProteinV3():
    """
    Truncate the receptor and change the order of peptide saved in order.
    the total pair length is 224 + 32 pad all!
    so we don't need to padding again in the collate function in dataloader
    """
    def __init__(self, rec_length=224, pep_length=32):
        self.rec_length = rec_length
        self.pep_length = pep_length
        self.pad_dict = {'pep_aa':20, 'pep_pos_heavyatom':0., 'pep_mask_heavyatom':False,
                         'rec_aa':20, 'rec_pos_heavyatom':0., 'rec_mask_heavyatom':False,
                         'pep_tag':False, 'rec_tag':False, 'pep_dihed':0., 'pep_dihed_mask':False}


    def __call__(self, data):
        data = data.copy() # don;t modify dict in inner function!
        LR = data['rec_aa'].shape[0]
        LP = data['pep_aa'].shape[0]
        if LR > self.rec_length:
            dist_cb = torch.cdist(data['pep_pos_heavyatom'][:,BBHeavyAtom.CB], data['rec_pos_heavyatom'][:,BBHeavyAtom.CB]) # choose cb here
            assert dist_cb.shape == (LP, LR), f'{dist_cb.shape} != ({LP}, {LR})'
            dist_cb = dist_cb.min(dim=0)[0] # along pep dim
            assert dist_cb.shape == (LR, ), f'{dist_cb.shape} != ({LR}, )'
            rec_idx = dist_cb.argsort()[:self.rec_length]
            for k,v in data.items():
                if k.startswith('rec'):
                    data[k] = select(v,rec_idx)
        
        assert data['rec_aa'].shape[0] <= self.rec_length
        assert data['pep_aa'].shape[0] <= self.pep_length

        
        data['rec_coord'] = data['rec_pos_heavyatom'][:,[1,2,0]]
        data['pep_coord'] = data['pep_pos_heavyatom'][:,[1,2,0]]

        return {
            'rec_coord': data['rec_coord'],
            'pep_coord': data['pep_coord'],
            
            'rec_aa': data['rec_aa'],
            'pep_aa': data['pep_aa'],

            'pep_dihed': data['pep_dihed'],
            'pep_dihed_mask': data['pep_dihed_mask'],
            'pep_mask_heavyatom': data['pep_mask_heavyatom'],
            'pep_pos_heavyatom': data['pep_pos_heavyatom'],
        }
