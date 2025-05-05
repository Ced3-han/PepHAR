import torch
import torch.nn as nn
from torch.nn import functional as F


class IntersectionLoss(nn.Module):
    def __init__(self, radius = 1.0, sigma = 2.5, epsilon = 1e-12):
        super(IntersectionLoss, self).__init__()
        self.radius = radius
        self.sigma = sigma
        self.epsilon = epsilon

    def forward(self, t1, t2, mask1, mask2):
        """
        Args:
            t1: (N, L1, 3)
            t2: (N, L2, 3)
            mask1: (N, L1)
            mask2: (N, L2)
        """
        distance = - ((t2[:, :, None, :] - t1[:, None, :, :]) ** 2).sum(-1) / self.sigma # (N, L2, L1)
        distance = self.radius + self.sigma * ((distance.exp() * mask1.unsqueeze(1)).sum(-1)
                                               + self.epsilon).log() # (N, L2)
        distance_mask = distance < 0 # (N, L2)
        distance = torch.where(distance_mask, torch.zeros_like(distance), distance) # (N, L2)
        loss = (distance * mask2).sum(-1) / mask2.sum(-1) # (N)
        return loss