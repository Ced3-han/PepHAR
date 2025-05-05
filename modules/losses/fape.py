import torch
from torch.nn import Module
from easydict import EasyDict

from modules.common.geometry import global_to_local


class FrameAlignmentLoss(Module):

    EPSILON = 1e-4

    def __init__(self, d_clamp=10, scale=10, clamp_prob=0.9, clamp_mode='pair'):
        super().__init__()
        self.d_clamp = d_clamp
        self.scale = scale
        self.clamp_prob = clamp_prob
        assert clamp_mode in ('pair', 'data', )
        self.clamp_mode = clamp_mode

    def forward(self, R_pred, t_pred, R_true, t_true, mask, L1 = None):
        """
        Args:
            R_pred: Predicted frame basis, (N, L, 3, 3).
            t_pred: Predicted frame translation, also the external coordinate of C_alpha, (N, L, 3).
        
            R_true: Ground truth frame basis, (N, L, 3, 3).
            t_true: Ground truth frame translation, also the external coordinate of C_alpha, (N, L, 3).
            
            mask:   Masks, (N, L).

            L1: length of first protein
        """
        N, L, _ = t_pred.size()
        p_pred_repeat = t_pred.unsqueeze(1).repeat(1, L, 1, 1)  # (N, L_rep, L_all, 3)
        p_pred_local_ij = global_to_local(R_pred, t_pred, p_pred_repeat)    # (N, L, L, 3)

        p_true_repeat = t_true.unsqueeze(1).repeat(1, L, 1, 1)
        p_true_local_ij = global_to_local(R_true, t_true, p_true_repeat)    # (N, L, L, 3)

        d_ij = torch.sqrt(((p_pred_local_ij - p_true_local_ij) ** 2).sum(dim=-1) + self.EPSILON)    # (N, L, L)

        if self.clamp_mode == 'pair':
            clamp_flag = torch.bernoulli(torch.full_like(d_ij, fill_value=self.clamp_prob)).bool()  # (N, L, L)
        elif self.clamp_mode == 'data':
            clamp_flag = torch.bernoulli(torch.full_like(d_ij, fill_value=self.clamp_prob)).bool()
            clamp_flag = clamp_flag[:, :1, :1].expand([N, L, L])    # (N, 1, 1) -> (N, L, L)
        clamp_flag = clamp_flag * (d_ij > self.d_clamp)
        d_ij_clamped = torch.where(clamp_flag, torch.full_like(d_ij, fill_value=self.d_clamp), d_ij)

        mask_pair = mask.view(N, L, 1) * mask.view(N, 1, L) # (N, L, L)

        if L1 is None:
            loss = torch.where(mask_pair, d_ij_clamped, torch.zeros_like(d_ij_clamped)).sum((-1, -2)) / \
                   mask_pair.sum((-1, -2))
            loss = loss.mean() / self.scale
        else:
            loss_11 = torch.where(mask_pair[:, :L1, :L1], d_ij_clamped[:, :L1, :L1],
                                  torch.zeros_like(d_ij_clamped[:, :L1, :L1])).sum((-1, -2))
            loss_22 = torch.where(mask_pair[:, L1:, L1:], d_ij_clamped[:, L1:, L1:],
                                  torch.zeros_like(d_ij_clamped[:, L1:, L1:])).sum((-1, -2))
            loss_12 = torch.where(mask_pair[:, :L1, L1:], d_ij[:, :L1, L1:],
                                  torch.zeros_like(d_ij_clamped[:, :L1, L1:])).sum((-1, -2))
            loss_21 = torch.where(mask_pair[:, L1:, :L1], d_ij[:, L1:, :L1],
                                  torch.zeros_like(d_ij_clamped[:, L1:, :L1])).sum((-1, -2))
            loss_self = (loss_11 + loss_22) / \
                        (mask_pair[:, :L1, :L1].sum((-1, -2)) + mask_pair[:, L1:, L1:].sum((-1, -2)))
            loss_cross = (loss_12 + loss_21) / \
                         (mask_pair[:, :L1, L1:].sum((-1, -2)) + mask_pair[:, L1:, :L1].sum((-1, -2)))
            loss = (loss_self + loss_cross).mean() * 0.5 / self.scale
        return loss


class FullFrameAlignmentLoss(Module):
    EPSILON = 1e-4

    def __init__(self, d_clamp=10, scale=1.0, clamp_prob=0.9, clamp_mode='pair'):
        super().__init__()
        self.d_clamp = d_clamp
        self.scale = scale
        self.clamp_prob = clamp_prob
        assert clamp_mode in ('pair', 'data',)
        self.clamp_mode = clamp_mode

    def forward(self, R_pred, t_pred, pos14_pred, R_true, t_true, pos14_true, frame_mask, pos14_mask):
        """
        Args:
            R_pred: Predicted frame basis, (N, L, 6, 3, 3).
            t_pred: Predicted frame translation, (N, L, 6, 3).
            pos14_pred: (N, L, 14, 3).
            R_true: (N, L, 6, 3, 3).
            t_true: (N, L, 6, 3).
            pos14_true: (N, L, 14, 3).
            frame_mask: (N, L, 6).
            pos14_mask: (N, L, 14).
        """
        N, L, _, _ = t_pred.size()
        p_pred_repeat = pos14_pred.view(N, -1, 3).unsqueeze(1).repeat(1, L * 6, 1, 1) # (N, L * 6, L * 14, 3)
        p_pred_local_ij = global_to_local(R_pred.view(N, -1, 3, 3), t_pred.view(N, -1, 3), p_pred_repeat)

        p_true_repeat = pos14_true.view(N, -1, 3).unsqueeze(1).repeat(1, L * 6, 1, 1)
        p_true_local_ij = global_to_local(R_true.view(N, -1, 3, 3), t_true.view(N, -1, 3), p_true_repeat)

        d_ij = torch.sqrt(((p_pred_local_ij - p_true_local_ij) ** 2).sum(dim=-1) + self.EPSILON)  # (N, L, L)

        if self.clamp_mode == 'pair':
            clamp_flag = torch.bernoulli(torch.full_like(d_ij, fill_value=self.clamp_prob)).bool()  # (N, L, L)
        elif self.clamp_mode == 'data':
            clamp_flag = torch.bernoulli(torch.full_like(d_ij, fill_value=self.clamp_prob)).bool()
            clamp_flag = clamp_flag[:, :1, :1].expand([N, L, L])  # (N, 1, 1) -> (N, L, L)
        clamp_flag = clamp_flag * (d_ij > self.d_clamp)
        d_ij_clamped = torch.where(clamp_flag, torch.full_like(d_ij, fill_value=self.d_clamp), d_ij)

        mask_pair = frame_mask.view(N, -1).unsqueeze(-1) * pos14_mask.view(N, -1).unsqueeze(-2)

        loss = torch.where(mask_pair, d_ij_clamped, torch.zeros_like(d_ij_clamped)).sum((-1, -2)) / \
               mask_pair.sum((-1, -2))
        loss = loss.mean() / self.scale
        return loss
