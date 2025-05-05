import math
from select import select
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import one_hot
from torch.nn import Linear
import torch.nn.functional as F
from evaluate.geometry import rotation_matrix_to_quaternion

from models.ga import GAEncoder
from modules.common.geometry import construct_3d_basis, quaternion_to_rotation_matrix

from utils.metrics import *

from ._base import register_model


@register_model('prediction_module_v1')
class PredictionModule(nn.Module):
    NUM_AA_TYPES = 21

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        ga_block_opt = {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }
        self.encoder = GAEncoder(node_feat_dim=node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers,
                                 ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(21 + 1, node_feat_dim)
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)
        self.classification_layer = Linear(node_feat_dim, 21)  # 20+1
        self.regression_layer = Linear(node_feat_dim, 21 * 4)  # sin_psi, cos_psi, sin_theta, cos_theta

    def _select_by_index(self, x, index):
        '''
            Args:
                x: (N, L, ...)
                index: (N, ) \in \{0, 1, ..., L-1\}
        '''
        assert x.dim() >= 2
        assert index.shape == (x.shape[0],)
        index = index.reshape(-1, *([1] * (x.dim() - 1)))  # (N, 1, ...)
        # print('_select_by_index: index.shape', index.shape)
        index = index.expand(-1, 1, *x.shape[2:])  # (N, 1, ...)
        # print('_select_by_index', x.shape, index.shape)
        x_selected = x.gather(1, index).squeeze(1)  # (N, ...)
        return x_selected

    def _remove_invalid_data(self, batch):
        mask = batch.get('anchor') != -1
        mask = mask.reshape(-1)
        return {k: v[mask] for k, v in batch.items()}

    def _batch_to_feature(self, batch: dict):
        assert all([k in batch for k in ['aa', 'coord', 'is_peptide', 'mask']])
        anchor = batch['anchor']
        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']

        res_feature = torch.cat([
            one_hot(batch['aa'], num_classes=self.NUM_AA_TYPES).float(),
            batch['is_peptide'].unsqueeze(-1).float(),
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
        return anchor, rot, pos, res_feature, pair_feature, mask

    def _batch_to_label(self, batch):
        assert all([k in batch for k in ['label_type', 'label_angle']])
        label_type = batch['label_type']  # (B, )
        label_angle = batch['label_angle']  # (B, 4)
        return label_type, label_angle

    def forward(self, anchor, rot, pos, res_feature, pair_feature, mask):
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask,
                                 is_query=torch.zeros_like(mask))  # (B, L, D)
        embedding = self._select_by_index(embedding, anchor)  # (B, D)
        type_logits = self.classification_layer(embedding)  # (B, AA_TYPES)
        prediction = self.regression_layer(embedding).reshape(-1, self.NUM_AA_TYPES, 4)  # (B, AA_TYPES, 4)
        psi, phi = prediction[:, :, :2], prediction[:, :, 2:]
        type_angles = torch.cat([F.normalize(psi, dim=-1), F.normalize(phi, dim=-1)], dim=-1)
        return type_logits, type_angles

    def get_loss(self, batch):
        batch = self._remove_invalid_data(batch)
        B = batch['aa'].shape[0]
        if B == 0:
            return {}
        features = self._batch_to_feature(batch)
        label_type, label_angle = self._batch_to_label(batch)  # (B, ), (B, 4)
        pred_type_logits, pred_type_angles = self.forward(*features)  # (B, 21), (B, 21, 4)
        loss_class = F.cross_entropy(pred_type_logits, label_type)
        pred_angle = self._select_by_index(pred_type_angles, label_type)  # (B, 4)
        loss_regress = F.mse_loss(pred_angle, label_angle)

        angle_abs = F.l1_loss(pred_angle, label_angle)
        pred_acc = (torch.argmax(pred_type_logits, dim=-1) == label_type).float().mean()

        return {'clf_loss': loss_class, 'angle_loss': loss_regress}, angle_abs, pred_acc


@register_model('prediction_module_new')
class PredictionModuleV2(nn.Module):
    NUM_AA_TYPES = 21

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        ga_block_opt = {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }
        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim,
                                 num_layers=num_layers,
                                 ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(21 + 1, self.node_feat_dim)  # is_peptide
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)
        # first predict aa, then predict angles
        self.classification_layer = Linear(self.node_feat_dim, 21)  # 20+1
        self.regression_layer = Linear(self.node_feat_dim + 21,
                                       4)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax

    def _batch_to_feature(self, batch: dict):
        # assert all([k in batch for k in ['aa', 'coord', 'is_peptide', 'mask']])
        batch['coord'] = torch.cat([batch['rec_coords'], batch['pep_coords']], dim=1)  # (B,L,3,3)

        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']

        res_feature = torch.cat([
            one_hot(batch['aa'], num_classes=21).float(),
            batch['is_peptide'].unsqueeze(-1).float(),
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

    def forward(self, rot, pos, res_feature, pair_feature, mask, max_rec_length=224, max_pep_length=32):
        # embedding = self.encoder(rot, pos, res_feature, pair_feature, mask,
        #                          is_query=torch.zeros_like(mask))  # (B, L1+L2, D),L1=224,L2=32
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask,
                                 max_rec_length, max_pep_length)  # (B, L1+L2, D),L1=224,L2=32
        pep_embedding = embedding[:, max_rec_length:]  # (B,L2,D), L2 pad
        type_logits = self.classification_layer(pep_embedding)  # (B, L2, AA_TYPES)
        prediction = self.regression_layer(torch.cat([pep_embedding, type_logits], dim=-1))  # (B,L2,4)
        psi, phi = prediction[:,:,:2], prediction[:,:,2:] # [phi,psi]
        type_angles = torch.cat([F.normalize(psi, dim=-1), F.normalize(phi, dim=-1)], dim=-1)
        return type_logits, type_angles

    def get_loss(self, batch):
        features = self._batch_to_feature(batch)
        label_types, label_angles = batch['label_types'], batch['label_angles']
        label_types_mask, label_angles_mask = batch['label_types_mask'], batch['label_angles_mask']
        pred_type_logits, pred_type_angles = self.forward(*features)  # (B, L, 21), (B,L,4)
        loss_class = cross_entropy_with_mask(pred_type_logits, label_types, label_types_mask)
        loss_regress = ((label_angles - pred_type_angles) ** 2 * batch['label_angles_mask']).sum() / batch[
            'label_angles_mask'].sum()
        pred_acc = accuracy_with_mask(pred_type_logits, label_types, label_types_mask)
        pred_abs = (torch.abs(label_angles - pred_type_angles) * batch['label_angles_mask']).sum() / batch[
            'label_angles_mask'].sum()

        return {'clf_loss': loss_class, 'angle_loss': loss_regress}, pred_abs, pred_acc

    def sample(self, rot, pos, res_feature, pair_feature, mask, max_rec_length=224, max_pep_length=32, is_sample=True):
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask,
                                 max_rec_length, max_pep_length, is_sample)  # (B, L1+L2, D),L1=224,L2=32
        pep_embedding = embedding[:, -1]  # (B,D),
        type_logits = self.classification_layer(pep_embedding)  # (B, AA_TYPES)
        prediction = self.regression_layer(torch.cat([pep_embedding, type_logits], dim=-1))  # (B,4)
        psi, phi = prediction[:,:2], prediction[:,2:] # [phi,psi]
        type_angles = torch.cat([F.normalize(psi, dim=-1), F.normalize(phi, dim=-1)], dim=-1)
        return type_logits, type_angles


@register_model('prediction_module_v3')
class PredictionModuleV3(nn.Module):
    NUM_AA_TYPES = 21

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        ga_block_opt = {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }
        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers, ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(21 + 1, self.node_feat_dim)  # is_peptide
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)
        # first predict aa, then predict angles
        self.classification_layer = Linear(self.node_feat_dim, 2 * self.NUM_AA_TYPES)
        self.regression_layer = Linear(self.node_feat_dim, 2 * self.NUM_AA_TYPES * 4)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax

    def _select_by_index(self, x, index):
        '''
            Args:
                x: (d1, d2, ..., dk, L, ...)
                index: (d1, ..., dk) \in \{0, 1, ..., L-1\}
        '''
        DX, DI = x.dim(), index.dim()
        # reshape indices to the same length as x
        indices = index.reshape(*index.shape, *([1] * (DX - DI)))
        # expand indices as x except the last dim of indices
        indices = indices.expand(*index.shape, 1, *x.shape[DI + 1:])
        # select x by indices
        x_selected = x.gather(DI, indices).squeeze(DI)  # (N, ...)
        return x_selected


    def _batch_to_feature(self, batch: dict):
        # assert all([k in batch for k in ['aa', 'coord', 'is_peptide', 'mask']])
        batch['coord'] = torch.cat([batch['rec_coords'], batch['pep_coords']], dim=1)  # (B,L,3,3)

        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']

        res_feature = torch.cat([
            one_hot(batch['aa'], num_classes=21).float(),
            batch['is_peptide'].unsqueeze(-1).float(),
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

    def forward(self, rot, pos, res_feature, pair_feature, mask, max_rec_length=224, max_pep_length=32):
        # embedding = self.encoder(rot, pos, res_feature, pair_feature, mask,
        #                          is_query=torch.zeros_like(mask))  # (B, L1+L2, D),L1=224,L2=32
        B, L1, L2, D = rot.shape[0], max_rec_length, max_pep_length, self.node_feat_dim
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, max_rec_length, max_pep_length)  # (B, L1+L2, D),L1=224,L2=32
        assert embedding.shape == (B, L1 + L2, D)
        pep_embedding = embedding[:, max_rec_length:]  # (B, L2, D), L2 pad
        assert pep_embedding.shape == (B, L2, D)

        types_logits = self.classification_layer(pep_embedding).reshape(B, L2, 2, self.NUM_AA_TYPES)  # (B, L2, 2, AA_TYPES)
        assert types_logits.shape == (B, L2, 2, self.NUM_AA_TYPES)
        angles = self.regression_layer(pep_embedding).reshape(B, L2, 2, self.NUM_AA_TYPES, 4)  # (B, L2, 2, AA_TYPES, 4)
        assert angles.shape == (B, L2, 2, self.NUM_AA_TYPES, 4)
        angles = torch.cat([F.normalize(angles[..., :2], dim=-1), F.normalize(angles[..., 2:], dim=-1)], dim=-1)
        return types_logits, angles

    def get_loss(self, batch):
        features = self._batch_to_feature(batch)
        pred_types_logits, pred_angles = self.forward(*features)  # (B, L2, 2, AA_TYPES), (B, L2, 2, AA_TYPES, 4)
        
        label_types = batch['label_types']  # (B, L2, 2)
        label_angles = batch['label_angles']  # (B, L2, 2, 4)
        label_types_mask = batch['label_types_mask']  # (B, L2, 2)
        label_angles_mask = batch['label_angles_mask']  # (B, L2, 2, 4)

        loss_class = cross_entropy_with_mask(pred_types_logits, label_types, label_types_mask)
        pred_gt_angles = self._select_by_index(pred_angles, label_types)  # (B, L2, 2, 4)
        loss_reg = mse_with_mask(pred_gt_angles, label_angles, label_angles_mask)
        pred_acc = accuracy_with_mask(pred_types_logits, label_types, label_types_mask)
        pred_abs = l1_with_mask(pred_gt_angles, label_angles, label_angles_mask)
        return {'clf_loss': loss_class, 'angle_loss': loss_reg}, pred_abs, pred_acc


    def sample(self, rot, pos, res_feature, pair_feature, mask, max_rec_length=224, max_pep_length=32, is_sample=True):
        raise NotImplementedError
        B = rot.shape[0]
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, max_rec_length, max_pep_length, is_sample)  # (B, L1+L2, D),L1=224,L2=32
        pep_embedding = embedding[:, -1]  # (B, D)
        types_logits = self.classification_layer(pep_embedding).reshape(B, 2, self.NUM_AA_TYPES)  # (B, 2, AA_TYPES)
        angles = self.regression_layer(pep_embedding).reshape(B, 2, self.NUM_AA_TYPES, 4)  # (B, 2, AA_TYPES, 4)
        angles = torch.cat([F.normalize(angles[..., :2], dim=-1), F.normalize(angles[..., 2:], dim=-1)], dim=-1)
        return types_logits, angles



@register_model('prediction_module_v4')
class PredictionModuleV4(nn.Module):
    NUM_AA_TYPES = 21

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        
        self.normalize_angle = cfg.normalize_angle if hasattr(cfg, 'normalize_angle') else False

        ga_block_opt = {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }
        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers, ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(21 + 1, self.node_feat_dim)  # is_peptide
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)

        # first predict aa, then predict angles
        # self.classification_layer = Linear(self.node_feat_dim, self.NUM_AA_TYPES)
        self.regression_layer = Linear(self.node_feat_dim, 2 * 2 * 2)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax
        
    def _batch_to_feature(self, batch: dict):
        # assert all([k in batch for k in ['aa', 'coord', 'is_peptide', 'mask']])
        batch['coord'] = torch.cat([batch['rec_coords'], batch['pep_coords']], dim=1)  # (B,L,3,3)
        batch['aa'] = torch.cat([batch['rec_aa'], batch['pep_aa']], dim=1)  # (B,L)
        # batch['aa'] = torch.cat([batch['rec_aa'], torch.full_like(batch['pep_aa'], 20)], dim=1)  # (B,L)
        batch['mask'] = (batch['aa'] != 20)
        batch['is_peptide'] = torch.cat([torch.zeros_like(batch['rec_aa']), torch.ones_like(batch['pep_aa'])], dim=1)  # (B,L)

        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']

        res_feature = torch.cat([
            one_hot(batch['aa'], num_classes=21).float(),
            batch['is_peptide'].unsqueeze(-1).float(),
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

    def forward(self, rot, pos, res_feature, pair_feature, mask, max_rec_length=224, max_pep_length=32):
        # embedding = self.encoder(rot, pos, res_feature, pair_feature, mask,
        #                          is_query=torch.zeros_like(mask))  # (B, L1+L2, D),L1=224,L2=32
        B, L1, L2, D = rot.shape[0], max_rec_length, max_pep_length, self.node_feat_dim

        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, max_rec_length, max_pep_length)

        assert embedding.shape == (B, L1 + L2, D)
        pep_embedding = embedding[:, max_rec_length:]  # (B, L2, D), L2 pad
        assert pep_embedding.shape == (B, L2, D)

        # types_logits = self.classification_layer(pep_embedding).reshape(B, L2, 2, self.NUM_AA_TYPES)  # (B, L2, 2, AA_TYPES)
        # assert types_logits.shape == (B, L2, 2, self.NUM_AA_TYPES)
        angles = self.regression_layer(pep_embedding).reshape(B, L2, 2, 2, 2)  # (B, L2, 2, 2, 2)
        assert angles.shape == (B, L2, 2, 2, 2)
        if self.normalize_angle:
            angles = F.normalize(angles, dim=-1)
        return angles

    def get_loss(self, batch):
        features = self._batch_to_feature(batch)
        pred_angles_eu = self.forward(*features)  # (B, L2, 4)
        
        B, L, _, _, _ = pred_angles_eu.shape
        # load label
        label_angles_eu = batch['label_angles'].reshape(B, L, 2, 2, 2)
        label_angles_eu_mask = batch['label_angles_mask'].reshape(B, L, 2, 2, 2)

        assert pred_angles_eu.shape == (B, L, 2, 2, 2)
        assert label_angles_eu.shape == (B, L, 2, 2, 2)
        assert label_angles_eu_mask.shape == (B, L, 2, 2, 2)
        
        
        loss_reg = mse_with_mask(pred_angles_eu, label_angles_eu, label_angles_eu_mask)
        
        def angle_diff_with_mask(pred, label, mask):
            pred, label, mask = pred.reshape(-1, 2), label.reshape(-1, 2), mask.reshape(-1, 2)  # (N, C), (N, C), (N, C)
            pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
            label_angle = torch.atan2(label[:, 1], label[:, 0])
            mask = mask[:, 0] & mask[:, 1]
            angle_diff = torch.abs(pred_angle - label_angle)
            angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)  # (B, L2, 2)
            l1_dist = (angle_diff * mask).sum() / mask.sum()
            return l1_dist, angle_diff[mask], pred_angle[mask], label_angle[mask]
        
        psi_angle_dist, psi_angle_hist, psi_pred_hist, psi_label_hist = angle_diff_with_mask(pred_angles_eu[..., 0, :], label_angles_eu[..., 0, :], label_angles_eu_mask[..., 0, :])
        phi_angle_dist, phi_angle_hist, phi_pred_hist, phi_label_hist = angle_diff_with_mask(pred_angles_eu[..., 1, :], label_angles_eu[..., 1, :], label_angles_eu_mask[..., 1, :])
        
        return {'loss_reg': loss_reg}, {
            'psi/angle_diff': psi_angle_dist, 'psi/hist_angle_diff': psi_angle_hist, 'psi/hist_pred': psi_pred_hist, 'psi/hist_label': psi_label_hist,
            'phi/angle_diff': phi_angle_dist, 'phi/hist_angle_diff': phi_angle_hist, 'phi/hist_pred': phi_pred_hist, 'phi/hist_label': phi_label_hist,
        }


@register_model('prediction_module_d1')
class PredictionModuleD1(nn.Module):
    NUM_AA_TYPES = 21

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        self.lam_kai1 = cfg.lam_kai1
        self.lam_kai2 = cfg.lam_kai2

        ga_block_opt = {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }
        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers, ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(21 + 1, self.node_feat_dim)  # is_peptide
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)

        # first predict aa, then predict angles
        # self.classification_layer = Linear(self.node_feat_dim, self.NUM_AA_TYPES)
        self.regression_layer = Linear(self.node_feat_dim, 2 * 2 * 2)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax
        self.kai_layer = Linear(self.node_feat_dim, 2 * 2)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax
    
    def _filter_batch(self, batch):
        return {k: batch[k] for k in ['rec_coords', 'pep_coords', 'rec_aa', 'pep_aa']}

    def _batch_to_feature(self, batch: dict):
        batch = self._filter_batch(batch)
        batch['coord'] = torch.cat([batch['rec_coords'], batch['pep_coords']], dim=1)  # (B,L,3,3)
        batch['aa'] = torch.cat([batch['rec_aa'], batch['pep_aa']], dim=1)  # (B,L)
        # batch['aa'] = torch.cat([batch['rec_aa'], torch.full_like(batch['pep_aa'], 20)], dim=1)  # (B,L)
        batch['mask'] = (batch['aa'] != 20)
        batch['is_peptide'] = torch.cat([torch.zeros_like(batch['rec_aa']), torch.ones_like(batch['pep_aa'])], dim=1)  # (B,L)

        rot = construct_3d_basis(
            batch['coord'][:, :, 0, :],
            batch['coord'][:, :, 1, :],
            batch['coord'][:, :, 2, :],
        )
        pos = batch['coord'][:, :, 0, :]
        mask = batch['mask']

        res_feature = torch.cat([
            one_hot(batch['aa'], num_classes=21).float(),
            batch['is_peptide'].unsqueeze(-1).float(),
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

    def forward(self, batch: dict):
        rot, pos, res_feature, pair_feature, mask = self._batch_to_feature(batch)
        max_rec_length = 224
        max_pep_length = 32
        B, L1, L2, D = rot.shape[0], max_rec_length, max_pep_length, self.node_feat_dim
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, max_rec_length, max_pep_length, is_query=True)

        assert embedding.shape == (B, L1 + L2, D)
        pep_embedding = embedding[:, max_rec_length:]  # (B, L2, D), L2 pad
        assert pep_embedding.shape == (B, L2, D)

        # types_logits = self.classification_layer(pep_embedding).reshape(B, L2, 2, self.NUM_AA_TYPES)  # (B, L2, 2, AA_TYPES)
        # assert types_logits.shape == (B, L2, 2, self.NUM_AA_TYPES)
        mu_eu = self.regression_layer(pep_embedding).reshape(B, L2, 2, 2, 2)  # (B, L2, 2, 2, 2)
        log_kai = self.kai_layer(pep_embedding).reshape(B, L2, 2, 2)
        assert mu_eu.shape == (B, L2, 2, 2, 2)
        assert log_kai.shape == (B, L2, 2, 2)
        mu_eu = F.normalize(mu_eu, dim=-1)
        return mu_eu, log_kai

    def get_loss(self, batch):
        pred_angles_eu, log_kai = self.forward(batch)  # (B, L2, 4)
        
        B, L, _, _, _ = pred_angles_eu.shape
        # load label
        label_angles_eu = batch['label_angles'].reshape(B, L, 2, 2, 2)
        label_angles_eu_mask = batch['label_angles_mask'].reshape(B, L, 2, 2, 2)

        assert pred_angles_eu.shape == (B, L, 2, 2, 2)
        assert label_angles_eu.shape == (B, L, 2, 2, 2)
        assert label_angles_eu_mask.shape == (B, L, 2, 2, 2)
        
        # loss_reg = mse_with_mask(pred_angles_eu, label_angles_eu, label_angles_eu_mask)
        
        def von_mises_mle_with_mask(mu_eu, log_kai, label_eu, mask):
            mu_eu, log_kai, label_eu, mask = mu_eu.reshape(-1, 2), log_kai.reshape(-1), label_eu.reshape(-1, 2), mask.reshape(-1)
            mu_eu, log_kai, label_eu = mu_eu[mask], log_kai[mask], label_eu[mask]
            kai = log_kai.exp()
            cossim = (mu_eu * label_eu).sum(dim=-1)
            loss_part1 = -(kai * cossim).mean()
            loss_part2 = torch.log(torch.i0(kai)).mean()
            loss = loss_part1 + loss_part2
            loss_kai = (self.lam_kai2 * log_kai ** 2 + self.lam_kai1 * log_kai).mean()
            return loss, loss_kai, cossim.mean(), loss_part1, loss_part2, log_kai
        
        psi_loss, psi_loss_kai, psi_cossim, psi_loss_part1, psi_loss_part2, psi_log_kai_hist = von_mises_mle_with_mask(pred_angles_eu[..., 0, :], log_kai[..., 0], label_angles_eu[..., 0, :], label_angles_eu_mask[..., 0, 0])
        phi_loss, phi_loss_kai, phi_cossim, phi_loss_part1, phi_loss_part2, phi_log_kai_hist = von_mises_mle_with_mask(pred_angles_eu[..., 1, :], log_kai[..., 1], label_angles_eu[..., 1, :], label_angles_eu_mask[..., 1, 0])
        
        def angle_diff_with_mask(pred, label, mask):
            pred, label, mask = pred.reshape(-1, 2), label.reshape(-1, 2), mask.reshape(-1, 2)  # (N, C), (N, C), (N, C)
            pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
            label_angle = torch.atan2(label[:, 1], label[:, 0])
            mask = mask[:, 0] & mask[:, 1]
            angle_diff = torch.abs(pred_angle - label_angle)
            angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)  # (B, L2, 2)
            l1_dist = (angle_diff * mask).sum() / mask.sum()
            return l1_dist, angle_diff[mask], pred_angle[mask], label_angle[mask]
        
        psi_angle_dist, psi_angle_hist, psi_pred_hist, psi_label_hist = angle_diff_with_mask(pred_angles_eu[..., 0, :], label_angles_eu[..., 0, :], label_angles_eu_mask[..., 0, :])
        phi_angle_dist, phi_angle_hist, phi_pred_hist, phi_label_hist = angle_diff_with_mask(pred_angles_eu[..., 1, :], label_angles_eu[..., 1, :], label_angles_eu_mask[..., 1, :])
        
        return {'loss': psi_loss + phi_loss, 'loss_kai': psi_loss_kai + phi_loss_kai}, {
            'psi/angle_diff': psi_angle_dist, 'psi/hist_angle_diff': psi_angle_hist, 'psi/hist_pred': psi_pred_hist, 'psi/hist_label': psi_label_hist,
            'phi/angle_diff': phi_angle_dist, 'phi/hist_angle_diff': phi_angle_hist, 'phi/hist_pred': phi_pred_hist, 'phi/hist_label': phi_label_hist,
            'psi/cossin': psi_cossim, 'psi/loss_part1': psi_loss_part1, 'psi/loss_part2': psi_loss_part2, 'psi/hist_log_kai': psi_log_kai_hist,
            'phi/cossin': phi_cossim, 'phi/loss_part1': phi_loss_part1, 'phi/loss_part2': phi_loss_part2, 'phi/hist_log_kai': phi_log_kai_hist,
        }

@register_model('prediction_module_d2')
class PredictionModuleD2(nn.Module):
    NUM_AA_TYPES = 21
    MAX_LOG_KAI = 6.0

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        dropout_rate = cfg.dropout_rate
        ga_block_opt = cfg.ga_block_opt if 'ga_block_opt' in cfg else {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }

        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers, ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(self.NUM_AA_TYPES + 2, self.node_feat_dim)  # is_peptide
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)
        self.regression_layer = Linear(self.node_feat_dim, 2 * 2 * 2)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax
        self.kai_layer = Linear(self.node_feat_dim, 2 * 2)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax
        self.dropout = nn.Dropout(dropout_rate)
    
    def _filter_batch(self, batch):
        return {k: batch[k] for k in ['rec_coord', 'pep_coord', 'qry_coord', 'rec_aa', 'pep_aa', 'qry_aa']}
    
    def _get_length_batch(self, batch):
        return batch['rec_aa'].shape[1], batch['pep_aa'].shape[1], batch['qry_aa'].shape[1]

    def _batch_to_feature(self, batch: dict):
        coord = torch.cat([batch['rec_coord'], batch['pep_coord'], batch['qry_coord']], dim=1)  # (B,L,3,3)
        # print({k: v.shape for k, v in batch.items()})
        aa = torch.cat([batch['rec_aa'], batch['pep_aa'], batch['qry_aa']], dim=1)  # (B,L)
        mask = (aa != 20)
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

    def forward(self, batch: dict, qry_strategy='all'):
        batch = self._filter_batch(batch)
        rot, pos, res_feature, pair_feature, mask = self._batch_to_feature(batch)
        rec_length, pep_length, qry_length = self._get_length_batch(batch)
        B, L1, L2, D = rot.shape[0], rec_length + pep_length, qry_length, self.node_feat_dim
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, rec_length, pep_length, qry_length, qry_strategy=qry_strategy)
        embedding = self.dropout(embedding)
        assert embedding.shape == (B, L1 + L2, D)
        
        qry_embedding = embedding[:, L1:]  # (B, L2, D), L2 pad
        assert qry_embedding.shape == (B, L2, D)

        mu_eu = self.regression_layer(qry_embedding).reshape(B, L2, 2, 2, 2)  # (B, L2, 2, 2, 2)
        mu = torch.atan2(mu_eu[..., 1], mu_eu[..., 0])  # (B, L2, 2, 2)
        log_kai = (self.MAX_LOG_KAI * F.tanh(self.kai_layer(qry_embedding))).reshape(B, L2, 2, 2)
        assert mu.shape == (B, L2, 2, 2)
        assert log_kai.shape == (B, L2, 2, 2)
        return mu, log_kai

    def get_loss(self, batch):
        mu, log_kai = self.forward(batch)  # (B, L2, 4)
        
        # load label
        B, L, _, _, = mu.shape
        label_angles = batch['label_angle']
        label_angles_mask = batch['label_angle_mask']
        label_known = batch['label_known']
        assert mu.shape == (B, L, 2, 2)
        assert label_angles.shape == (B, L, 2, 2)
        assert label_angles_mask.shape == (B, L, 2, 2)
        assert label_known.shape == (B, L, 2, 2)
        
        loss_dict = {}
        info_dict = {}

        for angle_type in [0, 1]:
            for unknown_type in [False, True]:
                angle_tag = 'psi' if angle_type == 0 else 'phi'
                known_tag = 'unknown' if unknown_type else 'known'
                loss, info = von_mises_mle_with_mask(mu[..., angle_type], log_kai[..., angle_type], label_angles[..., angle_type], 
                    label_angles_mask[..., angle_type] & (label_known[..., angle_type] ^ unknown_type))
                loss_dict[f'loss_{angle_tag}_{known_tag}'] = loss
                info_dict[f'{angle_tag}/{known_tag}/loss'] = loss
                info_dict.update({f'{angle_tag}/{known_tag}/{k}': v for k, v in info.items()})
                angle_dist, info = angle_diff_with_mask(mu[..., angle_type], label_angles[..., angle_type], 
                    label_angles_mask[..., angle_type] & (label_known[..., angle_type] ^ unknown_type))
                info_dict[f'{angle_tag}/{known_tag}/angle_diff'] = angle_dist
                info_dict.update({f'{angle_tag}/{known_tag}/{k}': v for k, v in info.items()})
                
        return {'loss_mle': sum(loss_dict.values()) / 4}, info_dict


@register_model('prediction_module_abl_pos')
class PredictionModuleAblPos(nn.Module):
    NUM_AA_TYPES = 21
    MAX_LOG_KAI = 6.0

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        dropout_rate = cfg.dropout_rate
        ga_block_opt = cfg.ga_block_opt if 'ga_block_opt' in cfg else {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }

        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers, ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(self.NUM_AA_TYPES + 2, self.node_feat_dim)  # is_peptide
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)
        self.regression_layer = Linear(self.node_feat_dim, 2 * 7)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax
        self.dropout = nn.Dropout(dropout_rate)
    
    def _filter_batch(self, batch):
        return {k: batch[k] for k in ['rec_coord', 'pep_coord', 'qry_coord', 'rec_aa', 'pep_aa', 'qry_aa']}
    
    def _get_length_batch(self, batch):
        return batch['rec_aa'].shape[1], batch['pep_aa'].shape[1], batch['qry_aa'].shape[1]

    def _batch_to_feature(self, batch: dict):
        coord = torch.cat([batch['rec_coord'], batch['pep_coord'], batch['qry_coord']], dim=1)  # (B,L,3,3)
        # print({k: v.shape for k, v in batch.items()})
        aa = torch.cat([batch['rec_aa'], batch['pep_aa'], batch['qry_aa']], dim=1)  # (B,L)
        mask = (aa != 20)
        is_peptide = torch.cat([torch.zeros_like(batch['rec_aa']), torch.ones_like(batch['pep_aa']), torch.zeros_like(batch['qry_aa'])], dim=1)  # (B,L)
        is_query = torch.cat([torch.zeros_like(batch['rec_aa']), torch.zeros_like(batch['pep_aa']), torch.ones_like(batch['qry_aa'])], dim=1)  # (B,L)

        rot = construct_3d_basis(coord[:, :, 0, :], coord[:, :, 1, :], coord[:, :, 2, :])
        pos = coord[:, :, 0, :]
        B, L, _, _ = rot.shape
        assert aa.shape == (B, L)
        assert rot.shape == (B, L, 3, 3)
        unit_mat = torch.eye(3).to(rot.device).unsqueeze(0).unsqueeze(1).expand(B, L, 3, 3)
        mask_mat = mask.unsqueeze(2).unsqueeze(3).expand(B, L, 3, 3)
        rot = torch.where(mask_mat, rot, unit_mat)

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

    def forward(self, batch: dict, qry_strategy='all'):
        batch = self._filter_batch(batch)
        rot, pos, res_feature, pair_feature, mask = self._batch_to_feature(batch)
        rec_length, pep_length, qry_length = self._get_length_batch(batch)
        B, L1, L2, D = rot.shape[0], rec_length + pep_length, qry_length, self.node_feat_dim
        embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, rec_length, pep_length, qry_length, qry_strategy=qry_strategy)
        embedding = self.dropout(embedding)
        assert embedding.shape == (B, L1 + L2, D)
        
        qry_embedding = embedding[:, L1:]  # (B, L2, D), L2 pad
        assert qry_embedding.shape == (B, L2, D)

        xo = self.regression_layer(qry_embedding).reshape(B, L2, 2, 7)  # (B, L2, 2, 7)
        x_local, o_local = xo[..., :3], F.normalize(xo[..., 3:], dim=-1)
        assert x_local.shape == (B, L2, 2, 3), o_local.shape == (B, L2, 2, 4)
        rot_qry = rot[:, L1:].unsqueeze(2).expand(B, L2, 2, 3, 3)
        pos_qry = pos[:, L1:].unsqueeze(2).expand(B, L2, 2, 3)
        assert rot_qry.shape == (B, L2, 2, 3, 3) and pos_qry.shape == (B, L2, 2, 3)
        x_global = torch.matmul(rot_qry, x_local.unsqueeze(-1)).squeeze(-1) + pos_qry

        def quaternion_multiply(q0, q1):
            assert q0.shape == q1.shape
            w0, x0, y0, z0 = torch.unbind(q0, -1)
            w1, x1, y1, z1 = torch.unbind(q1, -1)
            o = torch.stack([
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
            ], dim=-1)
            return o

        # mat_local = quaternion_to_rotation_matrix(o_local)
        # mat_global = torch.matmul(rot_qry, mat_local)
        # o_global = rotation_matrix_to_quaternion(mat_global)
        # for i in range(2):
        #     for j in range(32):
        #         for k in range(2):
        #             print(i, j, k)
        #             print(o_local[i, j, k])
        #             print(o_local[i, j, k].norm(dim=0))
        #             print(rot_qry[i, j, k])
        #             print(rot_qry[i, j, k].norm(dim=0))
        #             print(mat_local[i, j, k])
        #             print(mat_local[i, j, k].norm(dim=0))
        #             print(mat_global[i, j, k])
        #             print(mat_global[i, j, k].norm(dim=0))

        rot_quater = rotation_matrix_to_quaternion(rot_qry)
        o_global = quaternion_multiply(rot_quater, o_local)
        return torch.concat([x_global, o_global], dim=-1)

    def get_loss(self, batch):
        xo = self.forward(batch)  # (B, L2, 4)
        
        # load label
        B, L, _, _, = xo.shape
        label_angles = batch['label_angle']
        label_angles_mask = batch['label_angle_mask']
        label_known = batch['label_known']
        assert xo.shape == (B, L, 2, 7)
        assert label_angles.shape == (B, L, 2, 7)
        assert label_angles_mask.shape == (B, L, 2, 7)
        assert label_known.shape == (B, L, 2)
        
        loss_dict = {}
        info_dict = {}

        for i in range(2):
            for j in range(2):
                print(i, j, label_angles[i, j].detach().cpu().numpy())
                print(i, j, label_angles_mask[i, j].detach().cpu().numpy())
                print(i, j, xo[i, j].detach().cpu().numpy())
                print()

        for unknown_type in [False, True]:
            known_tag = 'unknown' if unknown_type else 'known'

            mask = label_angles_mask & ((label_known ^ unknown_type).unsqueeze(-1))
            assert mask.dtype == torch.bool
            pred = xo.reshape(-1)[mask.reshape(-1)]
            label = label_angles.reshape(-1)[mask.reshape(-1)]
            loss = ((pred - label) ** 2).mean()
            loss_dict[f'loss_{known_tag}'] = loss
            info_dict[f'{known_tag}/loss'] = loss
        
        print(loss_dict)
                
        return {'loss_mle': sum(loss_dict.values()) / 4}, info_dict
