import torch
import torch.nn as nn
from torch.nn import functional as F

from utils._protein_constants import restype_rigid_group_rotation, restype_rigid_group_translation, \
    restype_atom14_rigid_group_positions, restype_atom14_to_rigid_group, \
    PSI_FRAME, CHI1_FRAME, CHI2_FRAME, CHI3_FRAME, CHI4_FRAME
from modules.common.geometry import compose_chain, safe_norm


class TorsionAnglePrediction(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        self.lin = nn.Linear(feat_dim, feat_dim)
        self.lin_ini = nn.Linear(feat_dim, feat_dim)
        self.mlp_0 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.mlp_1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.lin_out = nn.Linear(feat_dim, 5 * 2) # (Psi, Chi1, Chi2, Chi3, Chi4) * (sine, cosine)

    def forward(self, x, x_ini = None):
        """
        Args:
            x:  (N, L, feat_dim)
            x_ini: (N, L, feat_dim)
        Returns:
            (N, L, 5, 2), un-normalized sine and cosine of Psi, Chi1, Chi2, Chi3, and Chi4.
        """
        N, L, _ = x.size()
        if x_ini is None:
            x_ini = torch.zeros_like(x)
        h = self.lin(x) + self.lin_ini(x_ini)
        h = h + self.mlp_0(h)
        h = h + self.mlp_1(h)
        y = self.lin_out(F.relu(h)).reshape(N, L, 5, 2)
        return y


def _make_psi_chi_rotation_matrices(alpha):
    """
    Args:
        alpha: (N, L, 5, 2), un-normalized sine and cosine of psi and chi1-4 angles.
    Returns:
        (N, L, 5, 3, 3)
    """
    N, L = alpha.shape[:2]
    sine, cosine = torch.unbind(alpha / safe_norm(alpha, dim=-1, keepdim=True), dim=-1)  # (N, L, 5), (N, L, 5)
    sine = sine.reshape(N, L, -1, 1, 1)
    cosine = cosine.reshape(N, L, -1, 1, 1)
    zero = torch.zeros_like(sine)
    one = torch.ones_like(sine)

    row1 = torch.cat([one, zero, zero], dim=-1)  # (N, L, 5, 1, 3)
    row2 = torch.cat([zero, cosine, -sine], dim=-1)  # (N, L, 5, 1, 3)
    row3 = torch.cat([zero, sine, cosine], dim=-1)  # (N, L, 5, 1, 3)
    R = torch.cat([row1, row2, row3], dim=-2)  # (N, L, 5, 3, 3)

    return R


class FullAtomReconstruction(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('restype_rigid_group_rotation', restype_rigid_group_rotation)  # (21, 8, 3, 3)
        self.register_buffer('restype_rigid_group_translation', restype_rigid_group_translation)  # (21, 8, 3)
        self.register_buffer('restype_atom14_to_rigid_group', restype_atom14_to_rigid_group)  # (21, 14)
        self.register_buffer('restype_atom14_rigid_group_positions',
                             restype_atom14_rigid_group_positions)  # (21, 14, 3)

    def _get_rigid_group(self, aa):
        """
        Args:
            aa: (N, L).
        """
        N, L = aa.size()
        aa = aa.flatten()
        rotation = self.restype_rigid_group_rotation[aa].reshape(N, L, 8, 3, 3)
        translation = self.restype_rigid_group_translation[aa].reshape(N, L, 8, 3)
        atom14_group = self.restype_atom14_to_rigid_group[aa].reshape(N, L, 14)
        atom14_position = self.restype_atom14_rigid_group_positions[aa].reshape(N, L, 14, 3)
        return rotation, translation, atom14_group, atom14_position

    def get_frame(self, R_bb, t_bb, alpha, aa):
        """
        Args:
            R_bb:   (N, L, 3, 3)
            t_bb:   (N, L, 3)
            alpha:  (N, L, 5, 2), un-normalized sine and cosine.
            aa:     (N, L)
        """
        N, L = aa.size()

        rot_psi, rot_chi1, rot_chi2, rot_chi3, rot_chi4 = _make_psi_chi_rotation_matrices(alpha).unbind(dim=2)
        # (N, L, 3, 3)
        zeros = torch.zeros_like(t_bb)

        rigid_rotation, rigid_translation, atom14_group, atom14_position = self._get_rigid_group(aa)

        R_psi, t_psi = compose_chain([
            (R_bb, t_bb),
            (rigid_rotation[:, :, PSI_FRAME], rigid_translation[:, :, PSI_FRAME]),
            (rot_psi, zeros),
        ])

        R_chi1, t_chi1 = compose_chain([
            (R_bb, t_bb),
            (rigid_rotation[:, :, CHI1_FRAME], rigid_translation[:, :, CHI1_FRAME]),
            (rot_chi1, zeros),
        ])

        R_chi2, t_chi2 = compose_chain([
            (R_chi1, t_chi1),
            (rigid_rotation[:, :, CHI2_FRAME], rigid_translation[:, :, CHI2_FRAME]),
            (rot_chi2, zeros),
        ])

        R_chi3, t_chi3 = compose_chain([
            (R_chi2, t_chi2),
            (rigid_rotation[:, :, CHI3_FRAME], rigid_translation[:, :, CHI3_FRAME]),
            (rot_chi3, zeros),
        ])

        R_chi4, t_chi4 = compose_chain([
            (R_chi3, t_chi3),
            (rigid_rotation[:, :, CHI4_FRAME], rigid_translation[:, :, CHI4_FRAME]),
            (rot_chi4, zeros),
        ])

        # Return Frame
        R_ret = torch.stack([R_bb, R_psi, R_chi1, R_chi2, R_chi3, R_chi4], dim=2)
        t_ret = torch.stack([t_bb, t_psi, t_chi1, t_chi2, t_chi3, t_chi4], dim=2)

        # Backbone, Omega, Phi, Psi, Chi1,2,3,4
        R_all = torch.stack([R_bb, R_bb, R_bb, R_psi, R_chi1, R_chi2, R_chi3, R_chi4], dim=2)  # (N, L, 8, 3, 3)
        t_all = torch.stack([t_bb, t_bb, t_bb, t_psi, t_chi1, t_chi2, t_chi3, t_chi4], dim=2)  # (N, L, 8, 3)

        index_R = atom14_group.reshape(N, L, 14, 1, 1).repeat(1, 1, 1, 3, 3)  # (N, L, 14, 3, 3)
        index_t = atom14_group.reshape(N, L, 14, 1).repeat(1, 1, 1, 3)  # (N, L, 14, 3)

        R_atom = torch.gather(R_all, dim=2, index=index_R)  # (N, L, 14, 3, 3)
        t_atom = torch.gather(t_all, dim=2, index=index_t)  # (N, L, 14, 3)
        p_atom = atom14_position  # (N, L, 14, 3)

        pos14 = torch.matmul(R_atom, p_atom.unsqueeze(-1)).squeeze(-1) + t_atom
        return pos14, R_ret, t_ret

    def forward(self, R_bb, t_bb, alpha, aa):
        pos14, _, _ = self.get_frame(R_bb, t_bb, alpha, aa)
        return pos14
