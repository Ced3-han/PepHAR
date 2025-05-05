import torch
import torch.nn as nn
from easydict import EasyDict

from modules.common.mask import mask_zero
from modules.losses.regression import L2Loss, L1Loss
from modules.losses.classification import CrossEntropyLoss


class MutationEnergyOutputs(nn.Module):

    def __init__(self, feat_dim, energy_list, loss_config):
        super().__init__()
        self.feat_dim = feat_dim
        self.energy_list = energy_list
        self.loss_config = loss_config

        self.mlp = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim), nn.ReLU(),
                                 nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))

        assert loss_config.type in ['L1', 'L2', 'CrossEntropy']
        self.loss_type = loss_config.type
        if loss_config.type == 'L1':
            self.loss = L1Loss(loss_config)
        elif loss_config.type == 'L2':
            self.loss = L2Loss(loss_config)
        elif loss_config.type == 'CrossEntropy':
            self.loss = CrossEntropyLoss(loss_config)

        self.regression = self.loss_type in ['L1', 'L2']
        if self.regression:
            self.linear_energy = nn.Linear(feat_dim, len(energy_list), bias=False)
        else:
            self.linear_energy = nn.Linear(feat_dim, len(energy_list) * loss_config.num_bins)

    def forward(self, feat_wt, feat_mut, mask=None, gt_info=None, get_loss=False):
        """
        Args:
            feat_wt: (N, L, d)
            feat_mut: (N, L, d)
            mask: None or (N, L)
            gt_info: None or Dict
            get_loss: Boolean
        """
        feat_mw = torch.cat([feat_mut, feat_wt], dim=-1)
        feat_wm = torch.cat([feat_wt, feat_mut], dim=-1)
        feat = self.mlp(feat_mw) - self.mlp(feat_wm)

        energy = self.linear_energy(feat)
        if mask is not None:
            energy = mask_zero(mask.unsqueeze(-1), energy)
        energy = energy.sum(dim=1)
        energy_dict = {term : energy[:, index] for index, term in enumerate(self.energy_list)}

        if get_loss:
            gt = torch.stack([gt_info[term] for term in self.energy_list], dim=-1)
            loss = self.loss(energy, gt)
            return energy_dict, loss
        else:
            return energy_dict