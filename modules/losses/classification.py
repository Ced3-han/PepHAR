import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.common.layers import DistanceToBins


class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_config):
        super(CrossEntropyLoss, self).__init__()
        self.loss_config = loss_config
        self.dist_expand = DistanceToBins(loss_config.energy_min, loss_config.energy_max, loss_config.num_bins,
                                          use_onehot=loss_config.use_onehot)

    def forward(self, pred, gt):
        """
        Args:
            pred: (N, _, bins)
            gt: (N, _)
        """
        N, _ = gt.size()

        p_target = self.dist_expand(gt.unsqueeze(-1), dim=-1, normalize=True)
        log_p_pred = F.log_softmax(pred, dim=-1)  # (N, _, num_bins), log-probability
        loss = F.kl_div(log_p_pred, p_target, reduction='none', log_target=False).sum(-1).mean(-1)

        return loss.mean()