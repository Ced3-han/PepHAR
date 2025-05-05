import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import one_hot
from torch.nn import Linear
import torch.nn.functional as F

from models.ga import GAEncoder
from modules.common.geometry import construct_3d_basis

from utils.metrics import *

from ._base import register_model

@register_model("frontier_model")
class FrontierModel(nn.Module):
    def __init__(self, cfg):  #
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        ga_block_opt = {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }

        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers,
                                 ga_block_opt=ga_block_opt)

        self.node_feat_layer = Linear(21, self.node_feat_dim)
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)

        self.classifier_layer = Linear(self.node_feat_dim, 21)
        self.regression_layer = Linear(self.node_feat_dim, 9)


    def forward(self, batch):
        R, t, res_feature, pair_feature, mask = self._batch_to_feature(batch)
        new_res_feature = self.encoder(R, t, res_feature, pair_feature, mask) # (B,L,D)
        # use the first token embedding for clf and reg
        pred_logits = self.classifier_layer(new_res_feature[:,0]) # (B,21)
        pred_coords = self.regression_layer(new_res_feature[:,0]) # (B,9)
        return pred_logits, pred_coords

    def _batch_to_feature(self, batch):
        #print(batch['coord'])
        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']

        #print(batch['aa'])
        res_feature = one_hot(batch['aa'], num_classes=21).float() # find 21 in aa

        res_feature = self.node_feat_layer(res_feature)

        N, L = pos.shape[:2]
        distance_map = pos.unsqueeze(2) - pos.unsqueeze(1)  # (N, L, 1, 3) - (N, 1, L, 3) = (N, L, L, 3)
        distance_map = (distance_map ** 2).sum(dim=-1)  # (N, L, L)

        assert self.pair_feat_dim % 2 == 0
        div_term = torch.exp(torch.linspace(0, 4, self.pair_feat_dim // 2)).to(distance_map.device)
        distance_map_sin = torch.sin(distance_map.unsqueeze(3) / div_term.view(1, 1, 1, -1))
        distance_map_cos = torch.cos(distance_map.unsqueeze(3) / div_term.view(1, 1, 1, -1))

        pair_feature = torch.cat([
            distance_map_sin, distance_map_cos,
        ], dim=3)
        pair_feature = self.pair_feat_layer(pair_feature)

        return rot, pos, res_feature, pair_feature, mask

    def get_loss(self, batch):
        pred_logits, pred_coords = self(batch)  # (B,21), (B,9)
        clf_loss = F.cross_entropy(pred_logits,batch['label_type'].view(-1))
        acc = (torch.argmax(pred_logits,dim=-1)==batch['label_type'].view(-1)).float().mean()
        # coord_loss = ((pred_coords-batch['label_coord'])**2).mean()
        # coord_abs = (torch.abs(pred_coords-batch['label_coord'])).mean()
        coord_loss = ((pred_coords-batch['label_coord'])**2).sum(-1).mean()
        coord_abs = (torch.abs(pred_coords-batch['label_coord'])).sum(-1).mean()
        return {'clf_loss':clf_loss,'coord_loss':coord_loss}, coord_abs, acc

