import torch
import torch.nn as nn

from modules.common.geometry import safe_norm


class TorsionAngleLoss(nn.Module):

    def __init__(self):
        super(TorsionAngleLoss, self).__init__()

    def get_alpha_pred(self, alpha_pred, gt, alt, mask):
        """
        Args:
            alpha_pred: (N, L, T, 2), un-normalized sine and cosine of torsional angles.
            gt:         (N, L, T), Ground truth torsional angles.
            alt:        (N, L, T), Alternative ground truth torsional angles.
            mask:       (N, L, T).
        """
        alpha_pred_norm = alpha_pred / safe_norm(alpha_pred, dim=-1, keepdim=True)
        alpha_gt = torch.stack([gt.sin(), gt.cos()], dim=-1)
        alpha_alt = torch.stack([alt.sin(), alt.cos()], dim=-1)

        l2_gt = ((alpha_pred_norm - alpha_gt) ** 2).sum(-1)
        l2_alt = ((alpha_pred_norm - alpha_alt) ** 2).sum(-1)

        alpha_mask = (mask * (l2_gt > l2_alt)).unsqueeze(-1).repeat(1, 1, 1, 2)
        alpha_pred = torch.where(alpha_mask, -alpha_pred, alpha_pred)

        l2_loss = torch.where(l2_gt < l2_alt, l2_gt, l2_alt)
        l2_loss = torch.where(mask, l2_loss, torch.zeros_like(l2_loss))
        mask = mask.sum((-1, -2))
        mask = torch.where(mask == 0, torch.ones_like(mask), mask)
        loss = (l2_loss.sum((-1, -2)) / mask).mean()

        return loss, alpha_pred

    def forward(self, alpha_pred, gt, alt, mask):
        loss, _ = self.get_alpha_pred(alpha_pred, gt, alt, mask)
        return loss


class AngleNormLoss(nn.Module):

    def __init__(self):
        super(AngleNormLoss, self).__init__()

    def forward(self, alpha_pred, mask):
        """
        Args:
            alpha_pred: (N, L, T, 2), un-normalized sine and cosine of torsional angles.
            mask:       (N, L, T).
        """
        loss = (safe_norm(alpha_pred, dim=-1) - 1).abs()
        loss = torch.where(mask, loss, torch.zeros_like(loss))

        mask = mask.sum((-1, -2))
        mask = torch.where(mask == 0, torch.ones_like(mask), mask)
        loss = (loss.sum((-1, -2)) / mask).mean()
        return loss
