import torch
import torch.nn as nn
import math
from easydict import EasyDict

from modules.common.layers import DistanceToBins, LayerNorm
from modules.losses.contact import ContactLoss


class ContactMapOutput(nn.Module):

    def __init__(self, in_channels, dist_min=0.0, dist_max=20.0, num_bins=64, use_onehot = False):
        super().__init__()
        self.in_channels = in_channels
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.num_bins = num_bins

        c = in_channels
        self.net = nn.Sequential(  # 1x1 convolutions
            nn.Conv2d(c, c//2, 1), nn.ELU(),
            nn.Conv2d(c//2, c//2, 1), nn.ELU(),
            nn.Conv2d(c//2, self.num_bins, 1),   # with an additional overflow symbol
        )

        self.criterion = ContactLoss(
            dist_min=dist_min,
            dist_max=dist_max,
            num_bins=num_bins,
            use_onehot=use_onehot,
        )

        self.layer_norm = LayerNorm(in_channels)

    @property
    def out_channels(self):
        return self.num_bins + 1

    def forward(self, x):
        """
        Args:
            x:  Amino acid pair features , (N, L, L, C)
        Returns:
            (N, L, L, num_bins), histograms.
        """
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)   # (N,L,L,C) -> (N,C,L,L)
        # x_t = x.permute(0, 1, 3, 2)
        # x = (x + x_t) * 0.5
        hist = self.net(x)  # (N,B,L,L)
        hist = hist.permute(0, 2, 3, 1) # (N,B,L,L) -> (N,L,L,B)
        return hist

    def get_loss(self, x, dmap, mask=None, return_results=False, reduction='mean'):
        """
        Args:
            x:  Amino acid pair features , (N, L, L, C).
            dmap:  Ground truth distances, (N, L, L).
            mask_loss:  Nodewise mask, (N, L, L).
        """
        N, L = x.shape[:2]

        out = self(x)
        loss_contact = self.criterion(
            out = out.permute(0, 3, 1, 2),
            d_gt = dmap.unsqueeze(1),
            mask_pair = mask,
            reduction = reduction,
        )

        losses = EasyDict({
            'contact': loss_contact,
        })

        if return_results:
            return losses, out
        else:
            return losses


if __name__ == '__main__':
    dist = torch.FloatTensor([0.25, 1.0, 20.0, 100.0]).view(4, 1, 1, 1)
    dist_exp = DistanceToBins()
    print(dist_exp(dist, dim=1).squeeze())
