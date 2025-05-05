import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from modules.common.layers import PositionalEncoding, LayerNorm
from modules.common.geometry import construct_3d_basis, global_to_local, safe_norm, pairwise_distances, knn_gather
from modules.common.mask import mask_coordinates_zero, mask_frames_identity
from utils.protein import ATOM_C, ATOM_CA, ATOM_N


class PointEncodingBlock(nn.Module):

    def __init__(self, n_knn, feat_dim):
        super().__init__()
        self.n_knn = n_knn
        self.feat_dim = feat_dim
        self.atom_embed = nn.Embedding(21 * 14, feat_dim)
        self.position_encode = PositionalEncoding()

        self.point_mlp = nn.Sequential(
            nn.Linear(self.position_encode.get_out_dim(3 + 1) + feat_dim * 2, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

        self.residue_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        self.layernorm = LayerNorm(feat_dim)

    def forward(self, pos14, res_feat, aa, mask_atom):
        """
        Args:
            R:  (N, L, 3, 3).
            t:  (N, L, 3).
            pos14:  (N, L, 14, 3)
            res_feat:   (N, L, F)
            aa: (N, L)
            mask_atom:   (N, L, 14)
        Returns:
            (N, L, F)
        """
        N, L = aa.size()
        mask_res = mask_atom[:, :, ATOM_CA]  # (N, L), Residue mask
        mask_flat = mask_atom.reshape(N, L * 14)

        # Construct residue frames
        R = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])
        R = mask_frames_identity(R, mask_res)
        t = pos14[:, :, ATOM_CA]
        t = mask_coordinates_zero(t, mask_res)

        atom_type = aa.unsqueeze(-1) * 14 + torch.arange(14).to(aa).reshape(1, 1, -1)  # (N, L, 14)
        atom_feat = torch.cat([
            res_feat.unsqueeze(2).repeat(1, 1, 14, 1),
            self.atom_embed(atom_type)
        ], dim=2)  # (N, L, 14, 2F)
        atom_feat = atom_feat.reshape(N, L * 14, -1)  # (N, L*14, 2F)

        pos_ca = pos14[:, :, ATOM_CA, :]  # (N, L, 3), 1 for CA
        pos_all = pos14.reshape(N, L * 14, 3)  # (N, L*14, 3)

        d = pairwise_distances(pos_ca, pos_all)  # (N, L, L*14)
        d = torch.where(mask_flat[:, None, :].expand_as(d), d, torch.full_like(d, fill_value=float('+inf')))
        n_knn = min(mask_flat.sum(dim=-1).min().item(), self.n_knn)
        _, knn_idx = d.topk(n_knn, dim=-1, largest=False)  # (N, L, K)

        # Aggregate neighbors and project to local frames
        pos_j = knn_gather(knn_idx, pos_all)  # (N, L, K, 3)
        pos_ij = global_to_local(R, t, pos_j)
        dist_ij = safe_norm(pos_ij, dim=-1, keepdim=True)  # (N, L, K, 1)
        spatial_ij = self.position_encode(torch.cat([pos_ij, dist_ij], dim=-1))
        # (N, L, K, P), Spatial relation encoding

        # Aggregate features
        feat_j = knn_gather(knn_idx, atom_feat)  # (N, L, K, 2F)

        # Transform atomwise features
        h_j = torch.cat([spatial_ij, feat_j], dim=-1)
        h_j = self.point_mlp(h_j)  # (N, L, K, F)

        # Pool to residue
        h_pool = torch.cat([h_j.mean(2), h_j.max(2)[0]], dim=-1)  # (N, L, 2F)
        h_res = self.layernorm(res_feat + self.residue_mlp(h_pool))  # (N, L, F) !!RESIDUAL CONNECTION!!
        h_res = h_res * mask_res.unsqueeze(-1)

        return h_res


class PointEncoder(nn.Module):

    def __init__(self, n_knn, feat_dim, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([
            PointEncodingBlock(n_knn, feat_dim)
            for _ in range(num_layers)
        ])

    def forward(self, pos14, res_feat, aa, mask_atom):
        h = res_feat
        for block in self.blocks:
            h = checkpoint(block, pos14, h, aa, mask_atom)  # Residual connection is included in the module
        return h