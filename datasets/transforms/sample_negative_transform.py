import torch
import torch.nn.functional as F
import math

from evaluate.geometry import quaternion_to_rotation_matrix
from ._base import register_transform


def random_rotation_matrix(n):
    """
    Generate N random matrices
    :param n:
    :return: R [N, 3, 3]
    """
    # u = F.normalize(torch.randn((n, 3)), p=2, dim=1)  # (n, 3)
    # t = 2 * math.pi * torch.rand((n, ))  # (n)
    # ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    # st, ct = torch.sin(t), torch.cos(t)
    # R = torch.stack([
    #     torch.stack([ct + ux ** 2 * (1 - ct), ux * uy * (1 - ct) - uz * st, ux * uz * (1 - ct) + uy * st], dim=1),
    #     torch.stack([uy * ux * (1 - ct) + uz * st, ct + uy ** 2 * (1 - ct), uy * uz * (1 - ct) - ux * st], dim=1),
    #     torch.stack([uz * ux * (1 - ct) - uy * st, uz * uy * (1 - ct) + ux * st, ct + uz ** 2 * (1 - ct)], dim=1),
    # ], dim=1)
    # return R
    o = F.normalize(torch.randn((n, 4)), p=2, dim=1)  # (n, 3)
    R = quaternion_to_rotation_matrix(o)
    return R

def _negative_sampleing(n, mu, sig=1.0):
    '''
    Perturb Gaussian mu
    :param    n: int, number of negative samples
    :param    mu: tensor (m, 3), mean of the distribution
    :return   new_coord (m, 3)
    '''
    assert isinstance(mu, torch.Tensor)
    assert mu.shape[1] == 3
    m = mu.shape[0]
    sample_index = torch.randint(0, m, (n, ))  # (n, )
    gaussian_center = mu[sample_index]
    center = gaussian_center + torch.randn((n, 3)) * sig
    R = random_rotation_matrix(n)
    new_coord = torch.stack([center, center + R[:, 0, :], center + R[:, 1, :]], dim=1)  # (n, 3, 3)
    assert new_coord.shape == (n, 3, 3)
    return new_coord


@register_transform('sample_negative')
class SampleNegativeType(object):

    def __init__(self, ratio=1.0, sig=1.0):
        self.ratio = ratio
        self.sig = sig

    def __call__(self, data):
        assert isinstance(data, dict)
        aa, pos_coord, is_query = data['aa'], data['coord'], data['is_query']

        n_samples = int(self.ratio * is_query.sum().item()) # 1:1 evaluate negative
        is_protein = ~data['is_query'] # receptor flag
        new_coord = _negative_sampleing(n_samples, pos_coord[is_protein][:, 0, :], sig=self.sig)
        aa = torch.cat([aa, torch.full((n_samples,), fill_value=20)], dim=0)
        pos_coord = torch.cat([pos_coord, new_coord], dim=0)
        is_query = torch.cat([is_query, torch.full((n_samples,), fill_value=True)], dim=0)

        return {
            'aa': aa,
            'coord': pos_coord,
            'is_query': is_query,
        }


@register_transform('sample_negative_new')
class SampleNegativeType_new(object):

    def __init__(self, ratio=1.0, sig=1.0):
        self.ratio = ratio
        self.sig = sig

    def __call__(self, data):
        assert isinstance(data, dict)
        aa, pos_coord, is_query = data['aa'], data['coord'], data['is_query']

        n_samples = int(self.ratio * is_query.sum().item()) # 1:1 evaluate negative
        is_protein = ~data['is_query'] # receptor flag
        new_coord = _negative_sampleing(n_samples, pos_coord[is_protein][:, 0, :], sig=self.sig)
        pos_coord = torch.cat([pos_coord, new_coord], dim=0)

        aa = torch.cat([aa, torch.full((n_samples,), fill_value=20)], dim=0)
        is_query = torch.cat([is_query, torch.full((n_samples,), fill_value=True)], dim=0)

        input_aa = torch.cat([data['aa'][is_protein],torch.full((is_query.sum().item(),),fill_value=21)],dim=0)
        mask = (input_aa!=20).bool()
        is_query[~mask] =  False # padding UNK is not query point

        return {
            'label_types': aa,
            'coord': pos_coord,
            'is_query': is_query,
            'aa' : input_aa,
            'mask':mask
        }


@register_transform('sample_negative_v3')
class SampleNegativeTypeV3(object):

    def __init__(self, ratio=1.0, sig=1.0):
        self.ratio = ratio
        self.sig = sig

    def __call__(self, data):
        assert isinstance(data, dict)
        max_pep_length = data['pep_aa'].shape[0]
        n_samples = int(self.ratio * data['pep_mask'].sum().item()) # 1:1 evaluate negative
        coords_new = _negative_sampleing(n_samples, data['rec_coords'][:, 0, :], sig=self.sig)
        coords_new = torch.cat([coords_new, torch.zeros(max_pep_length - n_samples, 3, 3)], dim=0)
        aa_new = torch.full((max_pep_length,), fill_value=20)
        mask_new = torch.cat([torch.ones(n_samples), torch.zeros(max_pep_length - n_samples)], dim=0).bool()

        ret = {
            'rec_coords': data['rec_coords'],
            'pep_coords': torch.cat([data['pep_coords'], coords_new], dim=0),
            'rec_aa': data['rec_aa'],
            'pep_aa': torch.cat([data['pep_aa'], aa_new], dim=0),
            'rec_mask': data['rec_mask'],
            'pep_mask': torch.cat([data['pep_mask'], mask_new], dim=0),
            
            'label': torch.cat([data['label'], aa_new], dim=0),
            'label_mask': torch.cat([data['label_mask'], mask_new], dim=0),
        }
        # print({k: v.shape for k, v in ret.items()})
        return ret


@register_transform('sample_negative_v4')
class SampleNegativeTypeV4(object):

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, data):
        assert isinstance(data, dict)
        assert data['rec_coord'].shape[0] == data['rec_aa'].shape[0], f'{data["rec_coord"].shape} {data["rec_aa"].shape}'
        assert data['pep_coord'].shape[0] == data['pep_aa'].shape[0], f'{data["pep_coord"].shape} {data["pep_aa"].shape}'
        assert data['qry_coord'].shape[0] == data['qry_aa'].shape[0], f'{data["qry_coord"].shape} {data["qry_aa"].shape}'
        assert data['label_class'].shape == data['label_class_mask'].shape, f'{data["label_class"].shape} {data["label_class_mask"].shape}'
        assert data['qry_aa'].shape == data['label_class'].shape, f'{data["qry_aa"].shape} {data["label_class"].shape}'

        # n_positive = data['qry_aa'].shape[0]
        # n_samples = int(self.ratio * n_positive) # 1:1 evaluate negative
        n_samples = 32
        # centers = torch.cat([data['rec_coord'][:, 0], data['pep_coord'][:, 0]], dim=0)
        mask = data['pep_mask_heavyatom'][:, 1]
        centers = data['pep_pos_heavyatom'][mask, 1]  # C-alphas
        coords_new = _negative_sampleing(n_samples, centers, sig=self.sigma)

        ret = {
            'rec_coord': data['rec_coord'],
            'pep_coord': data['pep_coord'],
            'rec_aa': data['rec_aa'],
            'pep_aa': data['pep_aa'],

            'qry_coord': torch.cat([data['qry_coord'], coords_new], dim=0),
            'qry_aa': torch.cat([data['qry_aa'], torch.zeros(n_samples).int()], dim=0),


            'label_class': torch.cat([data['label_class'], torch.full((n_samples,), 20).int()], dim=0),
            'label_class_mask': torch.cat([data['label_class_mask'], torch.ones(n_samples).bool()], dim=0),
        }
        # print({k: v.shape for k, v in ret.items()})
        return ret