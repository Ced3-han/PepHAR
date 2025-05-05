import torch
import torch.nn as nn


def node_mask_to_pair_mask(mask_1d):
    """
    Args:
        mask_1d:    (N, L).
    Returns:
        mask_2d:    (N, L, L).
    """
    return (mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(-2))


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def mask_frames_identity(R, mask):
    """
    Args:
        R:  (N, L, 3, 3).
        mask:   (N, L).
    Return:
        R_masked.
    """
    N, L = mask.size()
    I = torch.eye(3).to(R).view(1, 1, 3, 3).expand_as(R)    # (N, L, 3, 3)
    mask_R = mask.view(N, L, 1, 1).repeat(1, 1, 3, 3)
    R_masked = torch.where(mask_R, R, I)
    return R_masked


def mask_coordinates_zero(t, mask):
    N, L = mask.size()
    zero = torch.zeros_like(t)  # (N, L, 3)
    mask_t = mask.view(N, L, 1).repeat(1, 1, 3)
    t_masked = torch.where(mask_t, t, zero)
    return t_masked


def mask_positions(p, mask):
    N, L = mask.size()
    zero = torch.zeros_like(p)
    mask_p = mask.view(N, L, 1, 1).repeat(1, 1, p.size(2), p.size(3))
    p_masked = torch.where(mask_p, p, zero)
    return p_masked


def nan_to_zero(x):
    return torch.where(x.isfinite(), x, 0)
