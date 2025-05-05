import math

import torch
from torch.nn import Module
from torch.nn import functional as F

from modules.common.mask import node_mask_to_pair_mask
from modules.common.layers import DistanceToBins


class ContactLoss(Module):

    def __init__(self, dist_min=0.0, dist_max=20.0, num_bins=64, use_onehot=False):
        super().__init__()
        self.dist_expand = DistanceToBins(dist_min, dist_max, num_bins, use_onehot=use_onehot)

    def forward(self, out, d_gt, mask_pair=None, reduction='mean'):
        """
        Args:
            out:    Non-normalized histograms, (N, num_bins, L, L).
            d_gt:   Ground truth distances, (N, 1, L, L)
            mask:   Nodewise masks, (N, L, L)
        """
        N, _, L1, L2 = out.size()
        # mask = mask.view(N, 1, L)   # (N, L) -> (N, 1, L)
        # mask_pair = node_mask_to_pair_mask(mask)
        p_target = self.dist_expand(d_gt, dim=1, normalize=True)     # (N, num_bins, L, L), normalized probability
        log_p_pred = F.log_softmax(out, dim=1)  # (N, num_bins, L, L), log-probability
        plogp = F.kl_div(log_p_pred, p_target, reduction='none', log_target=False)

        if mask_pair is not None:
            plogp = plogp * mask_pair[:, None, :, :]    # (N, n_bins, L, L)
            n = mask_pair.sum()
        else:
            n = N * L1 * L2
        
        # sum_plogp = plogp.sum(dim=1)    # (N, L, L)

        if reduction == 'mean':
            loss = plogp.sum() / n
        elif reduction == 'sum':
            loss = plogp.sum()
        elif reduction is None:
            return plogp.sum(dim=1)
        else:
            raise ValueError('Unknown reduction: %s' % reduction)
        return loss
