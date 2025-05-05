import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


inf=1e5


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)

def angstrom_to_nm(x):
    return x / 10


def nm_to_angstrom(x):
    return x * 10


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)    # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def _alpha_from_logits(logits, mask, mask_pair):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
        mask_pair:  Masks, (L_i, L_j).
    Returns:
        alpha:  Attention weights.
    """

    N, L, _, _ = logits.size()

    mask_row = mask.view(N, L, 1, 1).expand_as(logits)  # (N, L, *, *)
    mask_pair_padding = mask_row * mask_row.permute(0, 2, 1, 3)  # (N, L, L, *)
    # mask_pair_query = ~is_query.view(N, 1, L, 1)  # (N, *, L, *)  a_{ij} = 0, if j is a query point
    # encoder decoder mask
    # from https://github.com/huggingface/transformers/issues/9366

    mask_pair = mask_pair.view(1, L, L, 1)

    # final mask_pair
    mask_pair = mask_pair & mask_pair_padding # (N, L, L, *), broad cast

    logits = torch.where(mask_pair, logits, logits - inf)
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


class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=32, query_key_dim=32, num_query_points=8,
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
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = nn.LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = nn.LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)  # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        # Query
        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points,
                              3)  # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points)  # Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        # Key
        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points,
                            3)  # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)  # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        # Q-K Product
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)  # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))
                                        / 2)  # (N, L, L, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)  # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points,
                              3)  # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3))  # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * \
                      value_points.unsqueeze(1)  # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2)  # (N, L, n_heads, n_pnts, 3)

        feat_points = global_to_local(R, t, aggr_points)  # (N, L, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)  # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)  # (N, L, n_heads, n_pnts, 3)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, mask, mask_pair):
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
        # mask encoder-decoder causal mask here
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask, mask_pair)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated


class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt={}):
        super(GAEncoder, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.blocks = nn.ModuleList([
            GABlock(node_feat_dim, pair_feat_dim, **ga_block_opt)
            for _ in range(num_layers)
        ])

    def _get_mask_pair(self, rec_length, pep_length, qry_length, qry_strategy):
        if qry_strategy == 'all':
            return torch.cat([
                torch.cat([torch.ones(rec_length, rec_length), torch.zeros(rec_length, pep_length), torch.zeros(rec_length, qry_length)], dim=1),
                torch.cat([torch.ones(pep_length, rec_length), torch.tril(torch.ones(pep_length, pep_length)), torch.zeros(pep_length, qry_length)], dim=1),
                torch.cat([torch.ones(qry_length, rec_length), torch.ones(qry_length, pep_length), torch.diag(torch.ones(qry_length))], dim=1),
            ], dim=0).bool()
        elif qry_strategy.startswith('prefix'):
            assert pep_length == qry_length, 'pep_length and qry_length must be equal when qry_strategy is prefix'
            delta = int(qry_strategy[len('prefix'):])
            return torch.cat([
                torch.cat([torch.ones(rec_length, rec_length), torch.zeros(rec_length, pep_length), torch.zeros(rec_length, qry_length)], dim=1),
                torch.cat([torch.ones(pep_length, rec_length), torch.tril(torch.ones(pep_length, pep_length)), torch.zeros(pep_length, qry_length)], dim=1),
                torch.cat([torch.ones(qry_length, rec_length), torch.tril(torch.ones(qry_length, pep_length), diagonal=-delta), torch.diag(torch.ones(qry_length))], dim=1),
            ], dim=0).bool()
        
        elif qry_strategy.startswith('suffix'):
            assert pep_length == qry_length, 'pep_length and qry_length must be equal when qry_strategy is prefix'
            delta = int(qry_strategy[len('suffix'):])
            return torch.cat([
                torch.cat([torch.ones(rec_length, rec_length), torch.zeros(rec_length, pep_length), torch.zeros(rec_length, qry_length)], dim=1),
                torch.cat([torch.ones(pep_length, rec_length), torch.triu(torch.ones(pep_length, pep_length)), torch.zeros(pep_length, qry_length)], dim=1),
                torch.cat([torch.ones(qry_length, rec_length), torch.triu(torch.ones(qry_length, pep_length), diagonal=delta), torch.diag(torch.ones(qry_length))], dim=1),
            ], dim=0).bool()
        else:
            raise NotImplementedError

    def forward(self, R, t, res_feat, pair_feat, mask, rec_length, pep_length, qry_length, qry_strategy='all'):
        N, L, _, _ = R.shape
        assert R.shape == (N, L, 3, 3), f"R.shape = {R.shape}"
        assert t.shape == (N, L, 3), f"t.shape = {t.shape}"
        assert res_feat.shape == (N, L, self.node_feat_dim), f"res_feat.shape = {res_feat.shape}"
        assert pair_feat.shape == (N, L, L, self.pair_feat_dim), f"pair_feat.shape = {pair_feat.shape}"
        assert L == rec_length + pep_length + qry_length, f"L = {L}, rec_length = {rec_length}, pep_length = {pep_length}, qry_length = {qry_length}"
        
        mask_pair = self._get_mask_pair(rec_length, pep_length, qry_length, qry_strategy).to(mask.device)
        assert mask_pair.shape == (L, L), f"mask_pair.shape = {mask_pair.shape}"

        for i, block in enumerate(self.blocks):
            res_feat = block(R, angstrom_to_nm(t), res_feat, pair_feat, mask, mask_pair)
        return res_feat