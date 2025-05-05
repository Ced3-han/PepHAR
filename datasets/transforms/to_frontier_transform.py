from ._base import register_transform
import torch
from utils.protein.constants import BBHeavyAtom
from modules.common.geometry import local_to_global,global_to_local,construct_3d_basis


def select(v, index):
    if isinstance(v, list):
        return [v[i] for i in index]
    elif isinstance(v, torch.Tensor):
        return v[index]
    else:
        raise NotImplementedError

@register_transform('to_frontier_transform')
class ToFrontierTransform():

    def __init__(self, rec_length=256):
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

        # all pad/clip to 256, 255 + pocket gly 1
        if len(data['rec_aa']) > self.rec_length-1:
            dist_cb = torch.cdist(data['pep_pos_heavyatom'][:,BBHeavyAtom.CB],
                                    data['rec_pos_heavyatom'][:,BBHeavyAtom.CB]) # choose cb here
            dist_cb = dist_cb.min(dim=0)[0] # along pep dim
            rec_idx = dist_cb.argsort()[:self.rec_length-1]
            for k,v in data.items():
                if k.startswith('rec'):
                    data_tmp[k] = select(v,rec_idx)
        else:
            for k, v in data.items():
                if k in self.pad_dict and k.startswith('rec'):
                    data_tmp[k] = self._pad_last(data[k], self.rec_length-1, value=self.pad_dict[k])

        label_type = data_tmp['pep_aa'][0].view(-1) # (1,)
        #label_coord = data_tmp['pep_pos_heavyatom'][0,[[1,2,0]]].view(-1) # (3,3)->(9,)
        aa = data_tmp['rec_aa']
        mask = (aa!=20).bool()
        coord = data_tmp['rec_pos_heavyatom'][:,[[1,2,0]]].squeeze(1) # (L,3,3), CA,C,N

        # relative position to the first aa compared to the center of pocket
        # 注意到这里pad可能会有影响？取没pad之前的
        pocket_CA = torch.mean(data['rec_pos_heavyatom'][:,1],dim=0)
        pocket_C = torch.mean(data['rec_pos_heavyatom'][:,2],dim=0)
        pocket_N = torch.mean(data['rec_pos_heavyatom'][:,0],dim=0)
        R = construct_3d_basis(pocket_CA,pocket_C,pocket_N) # (3,3_index)
        t = pocket_CA # (3,)
        # global 2 local, p <- R^{T}(q - t)
        label_coord = data_tmp['pep_pos_heavyatom'][0, [[1, 2, 0]]].squeeze(0).transpose(-1,-2) # (3,3_index)
        label_coord = R.transpose(-1,-2)@(label_coord-t.unsqueeze(-1)) # (3,3_index)
        label_coord = label_coord.transpose(-1,-2).reshape(-1) # (3_index,3), can't view here

        # pad for the first pocket gly
        pocket_coord = torch.stack([pocket_CA,pocket_C,pocket_N],dim=0).unsqueeze(0)
        coord = torch.cat([pocket_coord,coord],dim=0)
        aa = torch.cat([torch.tensor(5).long().view(-1),aa],dim=0) # GLY = 5
        mask = torch.cat([torch.tensor([True]).bool().view(-1),mask],dim=0)


        return {
            'label_type':label_type,
            'label_coord':label_coord,
            'aa':aa,
            'mask':mask,
            'coord':coord,
            'R':R,
            't':t,
        }

