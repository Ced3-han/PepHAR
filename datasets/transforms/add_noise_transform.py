import torch
import torch.nn.functional as F
from evaluate.geometry import construct_3d_basis, rotation_matrix_to_quaternion
from evaluate.tools import coord_from

from ._base import register_transform

@register_transform("add_noise")
class AddSpatialNoise():
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, data):
        assert isinstance(data, dict)
        pos_coord = data['coord']
        L, _, _ = pos_coord.shape
        random_displancement = torch.randn(L, 3, 3) * self.sigma
        pos_coord = pos_coord + random_displancement
        return {
            'aa': data['aa'],
            'coord': pos_coord,  # (L, 3, 3)
            'is_query': data['is_query'],
        }

@register_transform("add_noise_new")
class AddSpatialNoiseNew():
    def __init__(self, x_sigma=0.5, o_sigma=0.2):
        self.x_sigma = x_sigma
        self.o_sigma = o_sigma
    
    def _rand_coord(self, coord):
        x_init = coord[:, 0]
        o_init = rotation_matrix_to_quaternion(construct_3d_basis(coord[:, 0], coord[:, 1], coord[:, 2]))
        x_init = x_init + self.x_sigma * torch.randn_like(x_init)
        o_init = o_init + self.o_sigma * torch.randn_like(o_init)
        o_init = F.normalize(o_init, p=2, dim=-1)
        coord = coord_from(x_init, o_init)
        return coord
    
    def __call__(self, data):
        assert isinstance(data, dict)
        rec_coord = self._rand_coord(data['rec_coord'])
        pep_coord = self._rand_coord(data['pep_coord'])   

        return {
            **data,
            'rec_coord':rec_coord,
            'pep_coord':pep_coord,
        }
