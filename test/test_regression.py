
import sys, os
sys.path.append(f"/home/chentong/ws/pep-design")
# importing required libraries
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from utils.train import get_optimizer, get_scheduler, recursive_to
from utils.misc import BlackHole
from utils.protein.constants import resindex_to_ressymb
from datasets import get_dataset
import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.nn import Linear

from models.ga import GAEncoder
from modules.common.geometry import construct_3d_basis

from utils.metrics import *
import datetime
import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import wandb
import math
import torch.nn.functional as F


class AngleRegressionModel(nn.Module):
    NUM_AA_TYPES = 21

    def __init__(self, cfg):
        super().__init__()
        num_layers = cfg.num_layers
        self.node_feat_dim, self.pair_feat_dim = cfg.node_feat_dim, cfg.pair_feat_dim
        
        self.predict_target = cfg.predict_target if hasattr(cfg, 'predict_target') else 'this'
        self.mask_type = cfg.mask_type if hasattr(cfg, 'mask_type') else 'single'  # single, query, triangle
        self.normalize_angle = cfg.normalize_angle if hasattr(cfg, 'normalize_angle') else False


        assert self.predict_target in ['this', 'next', 'prev']
        ga_block_opt = {
            'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
            'num_value_points': 8, 'num_heads': 12, 'bias': False,
        }
        self.encoder = GAEncoder(node_feat_dim=self.node_feat_dim, pair_feat_dim=self.pair_feat_dim, num_layers=num_layers, ga_block_opt=ga_block_opt)
        self.node_feat_layer = Linear(21 + 1, self.node_feat_dim)  # is_peptide
        self.pair_feat_layer = Linear(self.pair_feat_dim, self.pair_feat_dim)

        # first predict aa, then predict angles
        # self.classification_layer = Linear(self.node_feat_dim, self.NUM_AA_TYPES)
        self.regression_layer = Linear(self.node_feat_dim, 4)  # sin_psi, cos_psi, sin_theta, cos_theta, another way is to use gumbal softmax
        
    def _batch_to_feature(self, batch: dict):
        # assert all([k in batch for k in ['aa', 'coord', 'is_peptide', 'mask']])
        batch['coord'] = torch.cat([batch['rec_coords'], batch['pep_coords']], dim=1)  # (B,L,3,3)
        batch['aa'] = torch.cat([batch['rec_aa'], batch['pep_aa']], dim=1)  # (B,L)
        # batch['aa'] = torch.cat([batch['rec_aa'], torch.full_like(batch['pep_aa'], 20)], dim=1)  # (B,L)
        batch['mask'] = (batch['aa'] != 20)

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

        if self.mask_type == 'single' or self.mask_type == 'triangle':
            embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, max_rec_length, max_pep_length)  # (B, L1+L2, D),L1=224,L2=32
        elif self.mask_type == 'query':
            embedding = self.encoder(rot, pos, res_feature, pair_feature, mask, max_rec_length, max_pep_length, is_query=True)

        assert embedding.shape == (B, L1 + L2, D)
        pep_embedding = embedding[:, max_rec_length:]  # (B, L2, D), L2 pad
        assert pep_embedding.shape == (B, L2, D)

        # types_logits = self.classification_layer(pep_embedding).reshape(B, L2, 2, self.NUM_AA_TYPES)  # (B, L2, 2, AA_TYPES)
        # assert types_logits.shape == (B, L2, 2, self.NUM_AA_TYPES)
        angles = self.regression_layer(pep_embedding).reshape(B, L2, 2, 2)  # (B, L2, 2, 2)
        assert angles.shape == (B, L2, 2, 2)
        if self.normalize_angle:
            angles = F.normalize(angles, dim=-1)
        return angles

    def get_loss(self, batch):
        features = self._batch_to_feature(batch)
        pred_angles_eu = self.forward(*features)  # (B, L2, 4)
        
        # load label
        if self.predict_target == 'this':
            label_angles = batch['pep_dihed']  # (B, L2, 3)
            label_angles_mask = batch['pep_dihed_mask']  # (B, L2, 3)
            label_angles_eu = torch.stack([torch.cos(label_angles[:, :, 2]), torch.sin(label_angles[:, :, 2]), torch.cos(label_angles[:, :, 1]), torch.sin(label_angles[:, :, 1])], dim=-1)
            label_angles_eu_mask = torch.stack([label_angles_mask[:, :, 2], label_angles_mask[:, :, 2], label_angles_mask[:, :, 1], label_angles_mask[:, :, 1]], dim=-1)
        elif self.predict_target == 'next':
            label_angles_eu = batch['label_angles'][:, :, 1, :]
            label_angles_eu_mask = batch['label_angles_mask'][:, :, 1, :]
        elif self.predict_target == 'prev':
            label_angles_eu = batch['label_angles'][:, :, 0, :]
            label_angles_eu_mask = batch['label_angles_mask'][:, :, 0, :]
        B, L, _, _ = pred_angles_eu.shape
        label_angles_eu = label_angles_eu.reshape(B, L, 2, 2)
        label_angles_eu_mask = label_angles_eu_mask.reshape(B, L, 2, 2)

        assert pred_angles_eu.shape == (B, L, 2, 2)
        assert label_angles_eu.shape == (B, L, 2, 2)
        assert label_angles_eu_mask.shape == (B, L, 2, 2)
        
        if self.mask_type == 'single':
            # only use the first position
            pred_angles_eu = pred_angles_eu[:, :1]
            label_angles_eu = label_angles_eu[:, :1]
            label_angles_eu_mask = label_angles_eu_mask[:, :1]
        
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
        
        psi_angle_dist, psi_angle_hist, psi_pred_hist, psi_label_hist = angle_diff_with_mask(pred_angles_eu[:, :, 0], label_angles_eu[:, :, 0], label_angles_eu_mask[:, :, 0])
        phi_angle_dist, phi_angle_hist, phi_pred_hist, phi_label_hist = angle_diff_with_mask(pred_angles_eu[:, :, 1], label_angles_eu[:, :, 1], label_angles_eu_mask[:, :, 1])
        
        return {'loss_reg': loss_reg}, {
            'psi/angle_diff': psi_angle_dist, 'psi/hist_angle_diff': psi_angle_hist, 'psi/hist_pred': psi_pred_hist, 'psi/hist_label': psi_label_hist,
            'phi/angle_diff': phi_angle_dist, 'phi/hist_angle_diff': phi_angle_hist, 'phi/hist_pred': phi_pred_hist, 'phi/hist_label': phi_label_hist,
        }


def fit(model, train_loader, val_loader, config):
    # Init wandb
    wandb.watch(model)
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    train_iterator = inf_iterator(train_loader)

    def train():
        model.train()
        batch = recursive_to(next(train_iterator), device)
        loss_dict, info_dict = model.get_loss(batch)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        wandb.log({f'train/{k}': wandb.Histogram(v.detach().cpu()) if v.dim() > 0 else v.item() for k, v in {**loss_dict, **info_dict}.items()})

    def valid():
        model.eval()
        tape = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = recursive_to(batch, device)
                loss_dict, info_dict = model.get_loss(batch)
                for k, v in {**loss_dict, **info_dict}.items():
                    tape[k].append(v.detach().cpu().reshape(-1) if v.dim() > 0 else v.item())
        wandb.log({f'valid/{k}': wandb.Histogram(torch.cat(v)) if isinstance(v[-1], torch.Tensor) else np.mean(v) for k, v in tape.items()})

    try:
        for it in tqdm(range(config.train.max_iters)):
            train()
            if it % config.train.val_freq == 0:
                valid()
    except KeyboardInterrupt:
        print('Interrupted by user')
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', type=str)
    parser.add_argument('--predict_target', type=str, default='this')
    parser.add_argument('--mask_type', type=str, default='single')
    # parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=-1)
    # parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--tag', type=str, default='')
    # parser.add_argument('--resume', type=str, default=None)
    # parser.add_argument('--postfix', type=str, default=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args = parser.parse_args()

    

    # config, config_name = load_config("./config_predict_dihedral.yml")
    config, config_name = load_config(args.configs)
    config.model.predict_target = args.predict_target
    config.model.mask_type = args.mask_type

    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    wandb.init(project='PepDesign', name=f'test_regression_{args.predict_target}_{args.mask_type}' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), config=config)
    wandb.config.update({'device': device})

    train_dataset = get_dataset(config.data.train)
    val_dataset = get_dataset(config.data.val)

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=8)

    model = AngleRegressionModel(config.model).to(device)
    fit(model, train_loader, val_loader, config)
