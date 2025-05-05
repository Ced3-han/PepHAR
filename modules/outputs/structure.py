import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict

from modules.common.layers import angstrom_to_nm, nm_to_angstrom, LayerNorm
from modules.common.geometry import compose_rotation_and_translation, quaternion_to_rotation_matrix, \
    repr_6d_to_rotation_matrix, global_to_local, local_to_global, normalize_vector
from modules.common.mask import mask_zero
from modules.losses.fape import FrameAlignmentLoss


def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)      # (N, L, *, *)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)     # (N, L, L, *)
    
    logits = torch.where(mask_pair, logits, logits-inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class IPABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=16, query_key_dim=16, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=bias)
        self.proj_key   = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim*num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points*num_heads*3, bias=bias)
        self.proj_key_point   = nn.Linear(node_feat_dim, num_query_points*num_heads*3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points*num_heads*3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features = (num_heads*pair_feat_dim) + (num_heads*value_dim) + (num_heads*num_value_points*(3+3+1)),
            out_features = node_feat_dim,
        )
        self.layer_norm = LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)    # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)      # (N, L, n_heads, qk_ch)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)    # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()
        # Query
        query_points = _heads(self.proj_query_point(x), self.num_heads*self.num_query_points,
                              3) # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points) # Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1) # (N, L, n_heads, n_pnts*3)
        # Key
        key_points = _heads(self.proj_key_point(x), self.num_heads*self.num_query_points,
                            3) # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points) # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1) # (N, L, n_heads, n_pnts*3)
        # Q-K Product
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1) # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))
                                        / 2) # (N, L, L, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2) # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2) # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim) # (N, L, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1) # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2) # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads*self.num_value_points,
                              3) # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3)) # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) *\
                      value_points.unsqueeze(1) # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2) # (N, L, n_heads, n_pnts, 3)

        feat_points = global_to_local(R, t, aggr_points) # (N, L, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1) # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4) # (N, L, n_heads, n_pnts, 3)
        
        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1), 
            feat_distance.reshape(N, L, -1), 
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1)) # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm(x + feat_all)
        return x_updated


class RotationTranslationPrediction(nn.Module):

    def __init__(self, feat_dim, rot_repr, nn_type='mlp'):
        super().__init__()
        assert rot_repr in ('quaternion', '6d')
        self.rot_repr = rot_repr
        if rot_repr == 'quaternion':
            out_dim = 3 + 3
        elif rot_repr == '6d':
            out_dim = 6 + 3
        self.nn = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        y = self.nn(x)  # (..., d+3)
        if self.rot_repr == 'quaternion':
            quaternion = torch.cat([torch.ones_like(y[..., :1]), y[..., 0:3]], dim=-1)
            R_delta = quaternion_to_rotation_matrix(quaternion)
            t_delta = y[..., 3:6]
            return R_delta, t_delta
        elif self.rot_repr == '6d':
            R_delta = repr_6d_to_rotation_matrix(y[..., 0:6])
            t_delta = y[..., 6:9]
            return R_delta, t_delta


class RigidUpdate(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=16, query_key_dim=16, num_query_points=4,
                 num_value_points=4, num_heads=12, rot_repr='quaternion', rot_tran_nn_type='mlp', bias=False):
        super().__init__()
        self.attention = IPABlock(node_feat_dim, pair_feat_dim, value_dim, query_key_dim, num_query_points,
                                  num_value_points, num_heads, bias)

        self.transition_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim),
        )
        self.transition_layer_norm = LayerNorm(node_feat_dim)

        self.rot_tran = RotationTranslationPrediction(node_feat_dim, rot_repr, nn_type=rot_tran_nn_type)
    
    def forward(self, R, t, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3). Unit: Angstrom.
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
            noise_R:  Standard deviation of R.
            noise_t:  Standard deviation of translation t, Unit: Angstrom.
        Returns:
            R': Updated basis matrices, (N, L, 3, 3_index).
            t': Updated coordinates, (N, L, 3).
            x': Updated node-wise features, (N, L, F).
        """
        t = angstrom_to_nm(t)   # Angstrom -> nanometer
        x = self.attention(R, t, x, z, mask)
        x = self.transition_layer_norm(x + self.transition_mlp(x))

        R_delta, t_delta = self.rot_tran(x) # (N, L, 3, 3), (N, L, 3)
        R_new, t_new = compose_rotation_and_translation(R, t, R_delta, t_delta)
        t_new = nm_to_angstrom(t_new)
        return R_new, t_new, x


class IterativeSharedRigidUpdate(nn.Module):

    def __init__(self, num_iters, node_feat_dim, pair_feat_dim, value_dim=16, query_key_dim=16, num_query_points=4,
                 num_value_points=4, num_heads=12, rot_repr='6d', rot_tran_nn_type='mlp', clamp_prob=0.0, bias=False):
        super().__init__()

        self.node_norm = LayerNorm(node_feat_dim)
        self.pair_norm = LayerNorm(pair_feat_dim)
        self.node_init = nn.Linear(node_feat_dim, node_feat_dim)

        self.num_structure_iters = num_iters
        self.structure_module = RigidUpdate(
            node_feat_dim = node_feat_dim,
            pair_feat_dim = pair_feat_dim,
            rot_repr = rot_repr,
            value_dim = value_dim,
            query_key_dim = query_key_dim,
            num_query_points = num_query_points,
            num_value_points = num_value_points,
            num_heads = num_heads,
            rot_tran_nn_type = rot_tran_nn_type,
            bias=bias,
        )
        self.fa_loss = FrameAlignmentLoss(clamp_prob=clamp_prob)
        self.aux_loss = FrameAlignmentLoss(clamp_prob=clamp_prob)

    def forward(self, node_feat, pair_feat, mask_seq, R_init=None, t_init=None):
        N, L, _ = node_feat.size()
        if R_init is None: R_init = torch.eye(3)[None, None, :, :].repeat(N, L, 1, 1).to(node_feat)
        if t_init is None: t_init = torch.zeros([N, L, 3]).to(node_feat)

        R, t = R_init, t_init
        R_traj, t_traj = [], []
        for l in range(self.num_structure_iters):
            R = R.clone().detach()
            R, t, node_feat = self.structure_module(R, t, node_feat, pair_feat, mask_seq)
            R_traj.append(R)
            t_traj.append(t)

        return EasyDict({
            'R': R,
            't': t,
            'R_traj': R_traj,
            't_traj': t_traj,
            'node_feat': node_feat,
        })

    def get_loss(self, node_feat, pair_feat, R_true, t_true, mask_seq, mask_loss, return_results=False, L1=None):
        """
        Args:
            R_init: Initial frame basis, (N, L, 3, 3).
            t_init: Initial frame translation, (N, L, 3).
            R_true: Ground truth frame basis, (N, L, 3, 3).
            t_true: Ground truth frame translation, (N, L, 3).
            p_true: Ground truth external coordinates of atoms, (N, L, A, 3).
        """
        node_feat = self.node_init(self.node_norm(node_feat))
        pair_feat = self.pair_norm(pair_feat)

        N, L, _ = node_feat.size()

        out = self(node_feat, pair_feat, mask_seq)

        loss_aux = 0
        for R, t in zip(out['R_traj'], out['t_traj']):
            loss_aux = loss_aux + self.aux_loss(R, t, R_true, t_true, mask_loss, L1)
        loss_aux = loss_aux / self.num_structure_iters

        loss_fape = self.fa_loss(out['R'], out['t'], R_true, t_true, mask_loss, L1)
        
        losses = EasyDict({
            'fape': loss_fape,
            'aux': loss_aux,
        })

        if return_results:
            return losses, out
        else:
            return losses
