import torch
import torch.nn as nn
from torch.nn import functional as F


class L2Loss(nn.Module):
    def __init__(self, clamp_config):
        super(L2Loss, self).__init__()
        self.clamp_config = clamp_config

    def forward(self, pred, gt):
        """
        Args:
            pred: (N, _)
            gt: (N, _)
        """
        if self.clamp_config.clamp:
            gt = torch.clamp(gt, self.clamp_config.clamp_min, self.clamp_config.clamp_max)
        loss = (pred - gt) ** 2
        return loss.mean()


class L1Loss(nn.Module):
    def __init__(self, clamp_config):
        super(L1Loss, self).__init__()
        self.clamp_config = clamp_config

    def forward(self, pred, gt):
        """
        Args:
            pred: (N, _)
            gt: (N, _)
        """
        if self.clamp_config.clamp:
            gt = torch.clamp(gt, self.clamp_config.clamp_min, self.clamp_config.clamp_max)
        loss = (pred - gt).abs()
        return loss.mean()