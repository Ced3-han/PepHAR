import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import Linear

from models.ga import GAEncoder
from modules.common.geometry import construct_3d_basis

from utils.metrics import *

from ._base import register_model

@register_model("density_module_v1")
class DensityModule(nn.Module):
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




    def forward(self, batch):
        R, t, res_feature, pair_feature, mask, is_query = self._batch_to_feature(batch)
        new_res_feature = self.encoder(R, t, res_feature, pair_feature, mask, is_query)
        pred_logits = self.classifier_layer(new_res_feature)
        return pred_logits

    def _batch_to_feature(self, batch):
        #print(batch['coord'])
        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']
        is_query = batch['is_query']

        #print(batch['aa'])
        res_feature = one_hot(batch['aa'], num_classes=21).float() # find 21 in aa
        res_feature[is_query] = 0.0  # mask the aa type for queries
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

        return rot, pos, res_feature, pair_feature, mask, is_query

    def get_loss(self, batch):
        pred_logits = self(batch)  # (N, L, 21)
        loss = cross_entropy_with_mask(pred_logits, batch['aa'], batch['is_query'])
        accuracy = accuracy_with_mask(pred_logits, batch['aa'], batch['is_query'])
        return {"ce_loss":loss}, pred_logits, accuracy


@register_model("density_module_new")
class DensityModule_new(nn.Module):
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

        self.node_feat_layer = Linear(21+1, self.node_feat_dim) # add mask tag
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)

        self.classifier_layer = Linear(self.node_feat_dim, 21) # predict aa type and unk

        self.dropout = nn.Dropout(cfg.dropout)


    def forward(self, batch):
        R, t, res_feature, pair_feature, mask = self._batch_to_feature(batch)
        new_res_feature = self.dropout(self.encoder(R, t, res_feature, pair_feature, mask,
                                       max_rec_length=224, max_pep_length=64, is_query=True))
        pred_logits = self.classifier_layer(new_res_feature)
        return pred_logits

    def _batch_to_feature(self, batch):
        #print(batch['coord'])
        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']
        #is_query = batch['is_query']

        #print(batch['aa'])
        res_feature = one_hot(batch['aa'], num_classes=21+1).float() # find 21 in aa, and mask
        #res_feature[is_query] = 0.0  # mask the aa type for queries
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
        pred_logits = self(batch)  # (N, L, 21)
        loss = cross_entropy_with_mask(pred_logits, batch['label_types'], batch['is_query'])
        accuracy = accuracy_with_mask(pred_logits, batch['label_types'], batch['is_query'])
        return {"clf_loss":loss}, torch.tensor(0.).to(accuracy), accuracy


@register_model("density_module_v3")
class DensityModuleV3(nn.Module):
    NUM_AA_TYPES = 20
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

        self.node_feat_layer = Linear(self.NUM_AA_TYPES + 1, self.node_feat_dim) # add mask tag
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)

        self.classifier_layer = Linear(self.node_feat_dim, self.NUM_AA_TYPES + 1) # predict aa type and unk

        self.dropout = nn.Dropout(cfg.dropout)

    def _batch_to_feature(self, batch):
        
        batch['coord'] = torch.cat([batch['rec_coords'], batch['pep_coords']], dim=1)  # (B,L,3,3)
        batch['aa'] = torch.cat([batch['rec_aa'], batch['pep_aa']], dim=1)  # (B,L)
        # batch['aa'] = torch.cat([batch['rec_aa'], torch.full_like(batch['pep_aa'], 20)], dim=1)  # (B,L)
        batch['mask'] = torch.cat([batch['rec_mask'], batch['pep_mask']], dim=1)  # (B,L)

        #print(batch['coord'])
        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']

        res_feature = F.one_hot(batch['aa'], num_classes=self.NUM_AA_TYPES + 1).float() # find 21 in aa, and mask
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

    def forward(self, batch, max_rec_length=224, max_pep_length=64):
        R, t, res_feature, pair_feature, mask = self._batch_to_feature(batch)
        B, L, D = res_feature.shape
        embedding = self.encoder(R, t, res_feature, pair_feature, mask, max_rec_length, max_pep_length, is_query=True)
        embedding = self.dropout(embedding)
        assert embedding.shape == (B, L, D)
        pep_embedding = embedding[:, max_rec_length:, :]
        assert pep_embedding.shape == (B, max_pep_length, D)
        pred_logits = self.classifier_layer(pep_embedding)
        assert pred_logits.shape == (B, max_pep_length, self.NUM_AA_TYPES + 1)
        return pred_logits

    def get_loss(self, batch):
        logits = self(batch)  # (N, L, 21)
        
        label = batch['label']
        mask = batch['label_mask']

        logits, label, mask = logits.reshape(-1, logits.shape[-1]), label.reshape(-1), mask.reshape(-1)  # (N, C), (N, C), (N, C)

        loss_mat = F.cross_entropy(logits, label, reduction='none')
        loss = loss_mat[mask].mean()
        pred = logits.argmax(dim=-1)
        accuracy = (pred == label)[mask].float().mean()
        hist_pred, hist_label = pred[mask], label[mask]

        mask_neg = (label == 20) & mask
        mask_pos = (label != 20) & mask

        return {"loss_clf":loss}, {
            'accuracy': accuracy,
            'hist_label': hist_label,
            'hist_pred': hist_pred,
            'neg/cross_entropy': F.cross_entropy(logits[mask_neg], label[mask_neg]),
            'neg/accuracy': (pred == label)[mask_neg].float().mean(),
            'neg/hist_label': label[mask_neg],
            'neg/hist_pred': pred[mask_neg],
            'pos/cross_entropy': F.cross_entropy(logits[mask_pos], label[mask_pos]),
            'pos/accuracy': (pred == label)[mask_pos].float().mean(),
            'pos/hist_label': label[mask_pos],
            'pos/hist_pred': pred[mask_pos],
        }


@register_model("density_module_v4")
class DensityModuleV4(nn.Module):
    NUM_AA_TYPES = 20
    def __init__(self, cfg):  #
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        ga_block_opt = cfg.ga_block_opt if 'ga_block_opt' in cfg else {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }
        dropout_rate = cfg.dropout_rate
        
        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers, ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(self.NUM_AA_TYPES + 1 + 2, self.node_feat_dim) # add mask tag
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)
        self.classifier_layer = Linear(self.node_feat_dim, self.NUM_AA_TYPES) # predict aa type and unk
        self.dropout = nn.Dropout(dropout_rate)

    def _filter_batch(self, batch):
        return {k: batch[k] for k in ['rec_coord', 'pep_coord', 'qry_coord', 'rec_aa', 'pep_aa', 'qry_aa']}
    
    def _get_length_batch(self, batch):
        return batch['rec_aa'].shape[1], batch['pep_aa'].shape[1], batch['qry_aa'].shape[1]

    def _batch_to_feature(self, batch):
        batch = self._filter_batch(batch)
        coord = torch.cat([batch['rec_coord'], batch['pep_coord'], batch['qry_coord']], dim=1)  # (B,L,3,3)
        aa = torch.cat([batch['rec_aa'], batch['pep_aa'], torch.full_like(batch['qry_aa'], 20)], dim=1)  # (B,L)
        mask = torch.cat([batch['rec_aa'], batch['pep_aa'], batch['qry_aa']], dim=1) != 20  # (B,L)
        is_peptide = torch.cat([torch.zeros_like(batch['rec_aa']), torch.ones_like(batch['pep_aa']), torch.zeros_like(batch['qry_aa'])], dim=1)  # (B,L)
        is_query = torch.cat([torch.zeros_like(batch['rec_aa']), torch.zeros_like(batch['pep_aa']), torch.ones_like(batch['qry_aa'])], dim=1)  # (B,L)

        rot = construct_3d_basis(coord[:, :, 0, :], coord[:, :, 1, :], coord[:, :, 2, :])
        pos = coord[:, :, 0, :]

        res_feature = torch.cat([
            one_hot(aa, num_classes=21).float(),
            is_peptide.unsqueeze(-1),
            is_query.unsqueeze(-1),
        ], dim=-1)  # attach is peptide here
        res_feature = self.node_feat_layer(res_feature)

        assert self.pair_feat_dim % 2 == 0
        distance_map = pos.unsqueeze(2) - pos.unsqueeze(1)  # (N, L, 1, 3) - (N, 1, L, 3) = (N, L, L, 3)
        distance_map = (distance_map ** 2).sum(dim=-1)  # (N, L, L)
        div_term = torch.exp(torch.linspace(0, 4, self.pair_feat_dim // 2)).to(distance_map.device)
        distance_map_sin = torch.sin(distance_map.unsqueeze(3) / div_term.view(1, 1, 1, -1))
        distance_map_cos = torch.cos(distance_map.unsqueeze(3) / div_term.view(1, 1, 1, -1))
        pair_feature = torch.cat([
            distance_map_sin, distance_map_cos,
        ], dim=-1)
        pair_feature = self.pair_feat_layer(pair_feature)
        return rot, pos, res_feature, pair_feature, mask

    def forward(self, batch, qry_strategy='all'):
        batch = self._filter_batch(batch)
        rot, pos, res_feature, pair_feature, mask = self._batch_to_feature(batch)
        rec_length, pep_length, qry_length = self._get_length_batch(batch)
        B, L1, L2, D = rot.shape[0], rec_length + pep_length, qry_length, self.node_feat_dim
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, rec_length, pep_length, qry_length, qry_strategy=qry_strategy)
        embedding = self.dropout(embedding)
        assert embedding.shape == (B, L1 + L2, D)

        qry_embedding = embedding[:, L1:, :]
        assert qry_embedding.shape == (B, L2, D)

        pred_logits = self.classifier_layer(qry_embedding)
        assert pred_logits.shape == (B, L2, self.NUM_AA_TYPES)

        nce_logits = F.pad(pred_logits, (0, 1), 'constant', value=0.0)
        assert nce_logits.shape == (B, L2, self.NUM_AA_TYPES + 1)

        return nce_logits


    def get_loss(self, batch):
        logits = self(batch)  # (N, L, 21)
        
        B, L, _ = logits.shape
        label = batch['label_class']
        mask = batch['label_class_mask']
        assert logits.shape == (B, L, self.NUM_AA_TYPES + 1), f'{logits.shape} != {(B, L, self.NUM_AA_TYPES + 1)}'
        assert label.shape == (B, L), f'{label.shape} != {(B, L)}'
        assert mask.shape == (B, L), f'{mask.shape} != {(B, L)}'

        mask_neg = (label == 20)
        mask_pos = (label != 20)

        info_dict = {}

        loss_neg, _ = cross_entropy_with_mask(logits, label, mask & mask_neg)
        loss_pos, _ = cross_entropy_with_mask(logits, label, mask & mask_pos)

        acc_neg, info_neg = accuracy_with_mask(logits, label, mask & mask_neg)
        acc_pos, info_pos = accuracy_with_mask(logits, label, mask & mask_pos)

        info_dict['loss_neg'] = loss_neg
        info_dict['loss_pos'] = loss_pos
        info_dict['acc_neg'] = acc_neg
        info_dict['acc_pos'] = acc_pos
        info_dict.update({f'neg/{k}': v for k, v in info_neg.items()})
        info_dict.update({f'pos/{k}': v for k, v in info_pos.items()})

        return {"loss_clf": loss_neg + loss_pos}, info_dict
