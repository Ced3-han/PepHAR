import math
from pathlib import Path
import warnings

import torch
from Bio import BiopythonWarning
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder

from .geometry import axis_angle_to_rotation_matrix, construct_3d_basis
from .tools import concatenate_data
from modules.common.geometry import dihedral_from_four_points

from utils.protein.constants import (AA, restype_to_allatom_names,
                        restype_to_heavyatom_names,
                        rigid_group_heavy_atom_positions)
import torch.nn.functional as F


def _pad_data(data, pad_value=0):
    """
        :param data: (N, ...)
        :param pad_value:
        :return: (N + 1, ...)
    """
    assert data.dim() > 0
    N = data.shape[0]
    return torch.cat([data, torch.full((1,), pad_value, dtype=data.dtype)], dim=0)


def _coord_and_type_to_heavyatom(coord, aa):
    """
        :param coord: (N, 3, 3)
        :param aa: (N,)
        :return: (N, 15, 3)
    """
    assert coord.dim() == 3
    assert coord.shape[0] == aa.shape[0]

    pos_center = coord[:, 0]  # (N, 3)
    mat_rotation = construct_3d_basis(coord[:, 0], coord[:, 1], coord[:, 2])  # (N, 3, 3)

    # dihedral_from_four_points(N1, CA1, C1, N2)
    psi = F.pad(dihedral_from_four_points(coord[:-1, 2], coord[:-1, 0], coord[:-1, 1], coord[1:, 2]), (0, 1), 'constant', value=0.0)
    # dihedral_from_four_points(C1, N2, CA2, C2)
    # phi = F.pad(dihedral_from_four_points(coord[:-1, 1], coord[1:, 2], coord[1:, 0], coord[1:, 1]), (1, 0), 'constant', value=0.0)

    pos_heavyatom = []
    mask_heavyatom = []
    for i, aa_res in enumerate(aa):
        restype = AA(aa_res.item())

        # atom_list = []  # (name, position)
        # for atom_name, pa_nb, pos_delta in rigid_group_heavy_atom_positions[restype]:
        #     pos_delta = torch.tensor(pos_delta)
        #     atom_pos = pos_delta if pa_nb == 0 else pos_delta + atom_list[pa_nb - 1][1]
        #     if atom_name in ['C', 'CA', 'N', 'O']:
        #         atom_list.append((atom_name, atom_pos))
        # pos_dict = {name: pos for name, pos in atom_list}
        pos_dict = {name: torch.tensor(pos) for name, nb, pos in rigid_group_heavy_atom_positions[restype] if nb == 0 or nb == 3}
        mat_o = axis_angle_to_rotation_matrix(F.normalize(pos_dict['C'] - pos_dict['CA'], p=2, dim=0), psi[i] + math.pi, angle_type='rad')  # (K, 3, 3)
        pos_dict['O'] = pos_dict['C'] + mat_o @ pos_dict['O']

        # print(pos_dict)

        pos_heavyatom.append(torch.stack([
            pos_center[i] + mat_rotation[i] @ pos_dict[atom_name] if atom_name in pos_dict else torch.zeros(3)
            for atom_name in restype_to_heavyatom_names[restype]
        ])) # 15 atoms here
        mask_heavyatom.append(torch.stack([
            torch.tensor(True) if atom_name in pos_dict else torch.tensor(False)
            for atom_name in restype_to_heavyatom_names[restype]
        ]))

    return torch.stack(pos_heavyatom, dim=0), torch.stack(mask_heavyatom, dim=0)


def generate_full_info_from_pep(data_peptide):
    aa, coord = data_peptide['pep_aa'], data_peptide['pep_coord']
    N = len(aa)
    chain_nb = torch.full((N,), 255, dtype=torch.int64)
    chain_id = ['Z'] * N
    resseq = torch.arange(1, N + 1, dtype=torch.int64)
    icode = [' '] * N
    pos_heavyatom, mask_heavyatom = _coord_and_type_to_heavyatom(coord, aa)  # (N, 15, 3), (N, 15)
    # is_peptide = torch.ones((N,), dtype=torch.bool)
    return {
        'chain_nb': chain_nb,
        'chain_id': chain_id,
        'aa': aa,
        'res_nb': resseq,
        'resseq': resseq,
        'icode': icode,
        'pos_heavyatom': pos_heavyatom,
        'mask_heavyatom': mask_heavyatom,
        # 'is_peptide': is_peptide,
    }


# def save_protein_peptide_pdb(data_protein, data_peptide, path=None):
#     assert all([k in data_protein for k in
#                 ['chain_nb', 'chain_id', 'aa', 'res_nb', 'resseq', 'icode', 'pos_heavyatom', 'mask_heavyatom',
#                  'is_peptide']])
#     assert all([k in data_peptide for k in ['aa', 'coord', 'is_peptide']])
#     data_peptide_full = generate_full_info(data_peptide)
#     if 'name' in data_protein:
#         tmp = data_protein.pop('name')
#     data_all = concatenate_data(data_peptide_full, data_protein)
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     save_pdb(data_all, path)
#     # save_pdb(data_peptide_full, path)


def extract_full_info_from_pep(data):
    N = data['pep_aa'].shape[0]
    return {
        'chain_nb': torch.full((N,), 255, dtype=torch.int64),
        'chain_id': ['Z'] * N,
        'aa': data['pep_aa'],
        'res_nb': data['pep_res_nb'],
        'resseq': data['pep_resseq'],
        'icode': data['pep_icode'],
        'pos_heavyatom': data['pep_pos_heavyatom'],
        'mask_heavyatom': data['pep_mask_heavyatom'],
        # 'is_peptide': is_peptide,
    }

def extract_full_info_from_rec(data):
    return {
        'chain_nb': data['rec_chain_nb'],
        'chain_id': data['rec_chain_id'],
        'aa': data['rec_aa'],
        'res_nb': data['rec_res_nb'],
        'resseq': data['rec_resseq'],
        'icode': data['rec_icode'],
        'pos_heavyatom': data['rec_pos_heavyatom'],
        'mask_heavyatom': data['rec_mask_heavyatom'],
        # 'is_peptide': is_peptide,
    }


def save_pdb_rec_pep(data_full, peptide, path=None):

    if 'pep_coord' in peptide.keys() and 'pep_aa' in peptide.keys():
        data_pep = generate_full_info_from_pep(peptide)
    elif 'pep_pos_heavyatom' in peptide.keys():
        data_pep = extract_full_info_from_pep(peptide)
    else:
        raise NotImplementedError

    data_rec = extract_full_info_from_rec(data_full)

    data_all = concatenate_data(data_rec, data_pep)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_pdb(data_all, path)
    print(f'saved, rec: {data_rec["aa"].size()}, pep: {data_pep["aa"].size()}')
    


def save_pdb(data, path=None):
    """
    Args:
        data:   A dict that contains: `chain_nb`, `chain_id`, `aa`, `resseq`, `icode`,
                `pos_heavyatom`, `mask_heavyatom`.
    """

    def _mask_select(v, mask):
        if isinstance(v, str):
            return ''.join([s for i, s in enumerate(v) if mask[i]])
        elif isinstance(v, list):
            return [s for i, s in enumerate(v) if mask[i]]
        elif isinstance(v, torch.Tensor):
            return v[mask]
        else:
            return v

    def _build_chain(builder, aa_ch, pos_heavyatom_ch, mask_heavyatom_ch, chain_id_ch, resseq_ch, icode_ch):
        builder.init_chain(chain_id_ch[0])
        builder.init_seg('    ')

        for aa_res, pos_allatom_res, mask_allatom_res, resseq_res, icode_res in \
                zip(aa_ch, pos_heavyatom_ch, mask_heavyatom_ch, resseq_ch, icode_ch):
            if not AA.is_aa(aa_res.item()):
                print('[Warning] Unknown amino acid type at %d%s: %r' % (resseq_res.item(), icode_res, aa_res.item()))
                continue
            restype = AA(aa_res.item())
            builder.init_residue(
                resname=str(restype),
                field=' ',
                resseq=resseq_res.item(),
                icode=icode_res,
            )
            for i, atom_name in enumerate(restype_to_heavyatom_names[restype]):
                if atom_name == '': continue  # No expected atom
                if (~mask_allatom_res[i]).any(): continue  # Atom is missing
                if len(atom_name) == 1:
                    fullname = ' %s  ' % atom_name
                elif len(atom_name) == 2:
                    fullname = ' %s ' % atom_name
                elif len(atom_name) == 3:
                    fullname = ' %s' % atom_name
                else:
                    fullname = atom_name  # len == 4
                builder.init_atom(atom_name, pos_allatom_res[i].tolist(), 0.0, 1.0, ' ', fullname, )

    warnings.simplefilter('ignore', BiopythonWarning)
    builder = StructureBuilder()
    builder.init_structure(0)
    builder.init_model(0)

    unique_chain_nb = data['chain_nb'].unique().tolist()
    for ch_nb in unique_chain_nb:
        mask = (data['chain_nb'] == ch_nb)
        aa = _mask_select(data['aa'], mask)
        pos_heavyatom = _mask_select(data['pos_heavyatom'], mask)
        mask_heavyatom = _mask_select(data['mask_heavyatom'], mask)
        chain_id = _mask_select(data['chain_id'], mask)
        resseq = _mask_select(data['resseq'], mask)
        icode = _mask_select(data['icode'], mask)

        _build_chain(builder, aa, pos_heavyatom, mask_heavyatom, chain_id, resseq, icode)

    structure = builder.get_structure()
    if path is not None:
        io = PDBIO()
        io.set_structure(structure)
        io.save(path)
    return structure