import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import math

from utils.train import recursive_to
from .geometry import quaternion_to_rotation_matrix, get_peptide_position, construct_3d_basis, \
    axis_angle_to_rotation_matrix
from torch.utils.data import default_collate

def select(v, index):
    if isinstance(v, list):
        return [v[i] for i in index]
    elif isinstance(v, torch.Tensor):
        return v[index]
    else:
        raise NotImplementedError


def masked_select(v, mask):
    if isinstance(v, str):
        return ''.join([s for i, s in enumerate(v) if mask[i]])
    elif isinstance(v, list):
        return [s for i, s in enumerate(v) if mask[i]]
    elif isinstance(v, torch.Tensor):
        return v[mask]
    else:
        raise NotImplementedError


def to_protein(data_full):
    return {k: masked_select(v, ~data_full.get('is_peptide')) for k, v in data_full.items()}


def to_peptide(data_full):
    is_peptide = data_full.get('is_peptide')
    res_nb = data_full.get('res_nb')[is_peptide]
    order = torch.argsort(res_nb)
    # print(res_nb, order)
    aa = data_full.get('aa')[is_peptide][order]
    pos_heavyatom = data_full.get('pos_heavyatom')[is_peptide][order]
    N = pos_heavyatom[:, 0, :]
    CA = pos_heavyatom[:, 1, :]
    C = pos_heavyatom[:, 2, :]
    return {'aa': aa, 'coord': torch.stack([CA, C, N], dim=1), 'is_peptide': torch.ones(len(aa), dtype=torch.bool)}


def concatenate_data(*data_list):
    for data in data_list:
        assert set(data.keys()) == set(data_list[0].keys()), f'{set(data.keys())} != {set(data_list[0].keys())}'
        keys = list(data.keys())
        for key in keys:
            assert len(data[key]) == len(
                data[keys[0]]), f'{key} has different length {len(data[key])} vs {len(data[keys[0]])}'
    keys = data_list[0].keys()
    data = {}
    for k in keys:
        var_k = data_list[0][k]
        if isinstance(var_k, list):
            data[k] = sum([d[k] for d in data_list], start=[])
        elif isinstance(var_k, torch.Tensor):
            data[k] = torch.cat([d[k] for d in data_list], dim=0)
    return data


## Preparing Data

def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    raise TypeError('batch must be a dict')


def to_numpy(batch):
    if isinstance(batch, dict):
        return {k: v.detach().cpu().numpy() for k, v in batch.items()}
    raise TypeError('batch must be a dict')


def data_to_batch(*data_list, collate_type='default'):
    for data in data_list:
        assert set(data.keys()) == set(data_list[0].keys())
    collate_fn = default_collate
    return collate_fn(data_list)


def batch_to_data(batch):
    data_list = []
    n = batch['aa'].shape[0]
    for i in range(n):
        data = {k: v[i] for k, v in batch.items()}
        data_list.append(data)
    return data_list


def coord_from_xm(x, mat):
    assert x.shape[:-1] == mat.shape[:-2], f'x:{x.shape}, mat:{mat.shape}'
    device = x.device
    structure = recursive_to(get_peptide_position(), device)
    # back to global
    x0 = x + torch.matmul(mat, structure['CA1'].unsqueeze(-1)).squeeze(-1)
    x1 = x + torch.matmul(mat, structure['C1'].unsqueeze(-1)).squeeze(-1)
    x2 = x + torch.matmul(mat, structure['N1'].unsqueeze(-1)).squeeze(-1)
    return torch.stack([x0, x1, x2], dim=-2)


def coord_from(x, o):
    assert x.shape[:-1] == o.shape[:-1], f'x:{x.shape}, o:{o.shape}'
    mat = quaternion_to_rotation_matrix(o)
    # print(o.shape)
    # print(o[..., 0].shape)
    # print(mat.shape)
    return coord_from_xm(x, mat)


### Generation of structures

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
    assert psi_cos.device == psi_sin.device and psi_cos.device == phi_cos.device and psi_cos.device == phi_sin.device

    device = psi_cos.device
    point_dict = recursive_to(get_peptide_position(), device)

    # 1. rotate C2 along CA2->N2 axis with psi_2
    direction_n_ca = F.normalize(point_dict['N2'] - point_dict['CA2'], dim=-1)
    # direction_n_ca = F.normalize(point_dict['N2'] - point_dict['CA2'], dim=-1), standard defination
    mat4 = axis_angle_to_rotation_matrix(direction_n_ca, torch.stack([phi_cos, -phi_sin], dim=-1), angle_type='tri')  # (K, 3, 3)
    point_dict['C2'] = point_dict['CA2'] + torch.matmul(mat4, (point_dict['C2'] - point_dict['CA2']).unsqueeze(-1)).squeeze(-1)  # (K, 3)

    # 2. rotate along CA1
    direction_c_ca = F.normalize(point_dict['C1'] - point_dict['CA1'], dim=-1)
    # direction_c_ca = F.normalize(point_dict['CA1'] - point_dict['C1'], dim=-1), standard defination
    mat5 = axis_angle_to_rotation_matrix(direction_c_ca, torch.stack([psi_cos, psi_sin], dim=-1), angle_type='tri')  # (K, 3, 3)
    point_dict['C2'] = point_dict['CA1'] + torch.matmul(mat5, (point_dict['C2'] - point_dict['CA1']).unsqueeze(-1)).squeeze(-1)  # (K, 3)
    point_dict['CA2'] = point_dict['CA1'] + torch.matmul(mat5, (point_dict['CA2'] - point_dict['CA1']).unsqueeze(-1)).squeeze(-1)  # (K, 3)
    point_dict['N2'] = point_dict['CA1'] + torch.matmul(mat5, (point_dict['N2'] - point_dict['CA1']).unsqueeze(-1)).squeeze(-1)  # (K, 3)

    shape = point_dict['CA2'].shape
    point_dict['N1'] = point_dict['N1'].expand(shape)
    point_dict['CA1'] = point_dict['CA1'].expand(shape)
    point_dict['C1'] = point_dict['C1'].expand(shape)

    return point_dict

def get_prev_structure(psi_cos, psi_sin, phi_cos, phi_sin):
    structure = get_next_structure(psi_cos, psi_sin, phi_cos, phi_sin)  # (K, 3)
    center = structure['CA2']
    rot_inv = torch.inverse(construct_3d_basis(structure['CA2'], structure['C2'], structure['N2']))  # (..., 3, 3)
    for key in structure.keys():
        structure[key] = torch.matmul(rot_inv, (structure[key] - center).unsqueeze(-1)).squeeze(-1)
    # print(structure)

    # assert torch.allclose(structure['CA2'], torch.tensor((0.000, 0.000, 0.000))), f'{structure["CA2"]}'
    # assert torch.allclose(structure['C2'], torch.tensor((1.517, -0.000, -0.000))), f'{structure["C2"]}'
    # assert torch.allclose(structure['N2'], torch.tensor((-0.572, 1.337, 0.000))), f'{structure["N2"]}'
    return structure


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


def generate_prev_coord(last_coord, angle):
    """
        Args:
            last_coord: (..., 3, 3)
            angle: (..., 4)
    """
    assert last_coord.shape[:-2:] == angle.shape[:-1], f'{last_coord.shape} vs {angle.shape}'
    structure = get_prev_structure(angle[..., 0], angle[..., 1], angle[..., 2], angle[..., 3])  # (K, 3)
    x_last = last_coord[..., 0, :]  # (.., 3)
    mat = construct_3d_basis(last_coord[..., 0, :], last_coord[..., 1, :], last_coord[..., 2, :])  # (..., 3, 3)
    # back to global
    new_x0 = x_last + torch.matmul(mat, structure['CA1'].unsqueeze(-1)).squeeze(-1)
    new_x1 = x_last + torch.matmul(mat, structure['C1'].unsqueeze(-1)).squeeze(-1)
    new_x2 = x_last + torch.matmul(mat, structure['N1'].unsqueeze(-1)).squeeze(-1)
    new_pos_coord = torch.stack([new_x0, new_x1, new_x2], dim=-2)  # (..., 3, 3)
    return new_pos_coord



def generate_next_amino_acid(data, anchor: torch.Tensor, angle: torch.Tensor):
    """
	:param data:
	:param anchor: (1, )
	:param angle: (4, )
	"""
    # print(anchor, angle)
    pos_coord = data.get('coord')
    last_coord = pos_coord[anchor]
    new_pos_coord = generate_next_coord(last_coord, angle)
    data_new = {
        'aa': torch.tensor([0]),
        'coord': new_pos_coord,
        'is_peptide': torch.tensor([True]),
    }
    return data_new


## datat transform

def to_protein(data_full):
    return {k: masked_select(v, ~data_full.get('is_peptide')) for k, v in data_full.items()}


def to_peptide(data_full):
    is_peptide = data_full.get('is_peptide')
    res_nb = data_full.get('res_nb')[is_peptide]
    order = torch.argsort(res_nb)
    # print(res_nb, order)
    aa = data_full.get('aa')[is_peptide][order]
    pos_heavyatom = data_full.get('pos_heavyatom')[is_peptide][order]
    N = pos_heavyatom[:, 0, :]
    CA = pos_heavyatom[:, 1, :]
    C = pos_heavyatom[:, 2, :]
    return {'aa': aa, 'coord': torch.stack([CA, C, N], dim=1), 'is_peptide': torch.ones(len(aa), dtype=torch.bool)}


class ToStandard():
    def __call__(self, data) -> dict:
        assert isinstance(data, dict)
        N = data['pos_heavyatom'][:, 0, :]
        CA = data['pos_heavyatom'][:, 1, :]
        C = data['pos_heavyatom'][:, 2, :]
        return {
            'aa': data['aa'],
            'coord': torch.stack([CA, C, N], dim=1),  # (L, 3, 3)
            'is_peptide': data['is_peptide'],
        }
