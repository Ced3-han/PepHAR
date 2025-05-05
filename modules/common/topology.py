import torch
import torch.nn.functional as F

from utils.protein.constants import AA, restype_to_heavyatom_names


def detect_disulfide_bonds(aa, pos_heavyatom, threshold=3.0):
    """
    Args:
        aa: Amino acid types, (B, L).
        pos_heavyatom:  Coordinates of heavy-atoms, (B, L, maxn_ha, 3).
    Returns:
        is_CYD: Disulfide cystine mask, BoolTensor, (B, L).
        disulfide_matrix:  Disulfide bond matrix, BoolTensor, (B, L, L).
    """
    L = aa.size(1)
    SG_index = restype_to_heavyatom_names[AA.CYS].index('SG')
    is_CYS = (aa == AA.CYS)
    both_CYS = is_CYS[:, :, None] & is_CYS[:, None, :]   # (B, L, L)

    SG_pos = pos_heavyatom[:, :, SG_index]  # (B, L, 3)
    disulfide_matrix = (torch.cdist(SG_pos, SG_pos) <= threshold) & both_CYS   # (B, L, L)
    disulfide_matrix[:, range(L), range(L)] = False # not self-interation

    is_CYD = disulfide_matrix.any(dim=-1) # (B, L)
    return is_CYD, disulfide_matrix


def count_residue_hops(chain_nb, res_nb, inf_val=99999, signed=False):
    """
    Args:
        chain_nb:   The number of chain that the residue belongs to, (B, L).
        res_nb:     Residue sequential number in the chain, (B, L).
    Returns:
        hops:   Number of hops between any two residues, (B, L, L).
    """
    hops = res_nb[:, None, :] - res_nb[:, :, None]    # (B, L, L)
    same_chain = (chain_nb[:, None, :] == chain_nb[:, :, None]) # (B, L, L)
    hops = torch.where(same_chain, hops, torch.full_like(hops, fill_value=inf_val))
    if not signed:
        hops = hops.abs()
    return hops


def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    Args:
        chain_nb, res_nb
    Returns:
        consec: A flag tensor indicating whether residue-i is connected to residue-(i+1), 
                BoolTensor, (B, L-1)[b, i].
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()   # (B, L-1)
    same_chain = (chain_nb[:, 1:] == chain_nb[:, :-1])
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])
    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag
