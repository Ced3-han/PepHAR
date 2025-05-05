import math
import torch
from torch.nn.functional import one_hot, cross_entropy
from torch.distributions.von_mises import VonMises

def cross_entropy_with_mask(logits, label, mask):
    """
        logits: (B, ..., C)
    """
    assert logits.shape[:-1] == label.shape, 'logits.shape = {}, label.shape = {}'.format(logits.shape, label.shape)
    assert label.shape == mask.shape , 'label.shape = {}, mask.shape = {}'.format(label.shape, mask.shape)
    assert label.max() < logits.shape[-1], 'label.max() = {}, logits.shape[-1] = {}'.format(label.max(), logits.shape[-1])

    logits, label, mask = logits.reshape(-1, logits.shape[-1]), label.reshape(-1), mask.reshape(-1)  # (N, C), (N,), (N,)
    logits, label = logits[mask], label[mask]
    loss = cross_entropy(logits, label)
    return loss, {}


def accuracy_with_mask(logits, label, mask):
    """
        logits: (B, ..., 20)
    """
    assert logits.shape[:-1] == label.shape, 'logits.shape = {}, label.shape = {}'.format(logits.shape, label.shape)
    assert label.shape == mask.shape , 'label.shape = {}, mask.shape = {}'.format(label.shape, mask.shape)
    
    logits, label, mask = logits.reshape(-1, logits.shape[-1]), label.reshape(-1), mask.reshape(-1)  # (N, C), (N,), (N,)
    logits, label = logits[mask], label[mask]
    pred = torch.argmax(logits, dim=-1)  # (N,)
    accuracy = (pred == label).float().mean()
    return accuracy, {'pred': pred, 'label': label}


def mse_with_mask(pred, label, mask):
    """
        pred: torch.Tensor
    """
    assert pred.shape == label.shape, 'pred.shape = {}, label.shape = {}'.format(pred.shape, label.shape)
    assert label.shape == mask.shape , 'label.shape = {}, mask.shape = {}'.format(label.shape, mask.shape)
    return ((pred - label) ** 2 * mask).sum() / mask.sum()


def l1_with_mask(pred, label, mask):
    """

    """
    assert pred.shape == label.shape, 'pred.shape = {}, label.shape = {}'.format(pred.shape, label.shape)
    assert label.shape == mask.shape , 'label.shape = {}, mask.shape = {}'.format(label.shape, mask.shape)
    return (torch.abs(pred - label) * mask).sum() / mask.sum()


def von_mises_mle_with_mask(mu, log_kai, label, mask):
    assert mu.shape == label.shape, 'mu.shape = {}, label.shape = {}'.format(mu.shape, label.shape)
    assert log_kai.shape == mu.shape, 'log_kai.shape = {}, mu.shape = {}'.format(log_kai.shape, mu.shape)
    assert mask.shape == mu.shape, 'mask.shape = {}, mu.shape = {}'.format(mask.shape, mu.shape)
    if mask.sum() == 0:
        return torch.tensor(0.0), {}
    mu, log_kai, label, mask = mu.reshape(-1), log_kai.reshape(-1), label.reshape(-1), mask.reshape(-1)
    mu, log_kai, label = mu[mask], log_kai[mask], label[mask]
    loss = -VonMises(mu, log_kai.exp()).log_prob(label).mean()
    cossim = (mu - label).cos().mean()
    return loss, {'sim': cossim, 'hist_log_kai': log_kai}


def angle_diff_with_mask(pred, label, mask):
    assert pred.shape == label.shape, 'pred.shape = {}, label.shape = {}'.format(pred.shape, label.shape)
    assert pred.shape == mask.shape, 'pred.shape = {}, mask.shape = {}'.format(pred.shape, mask.shape)
    if mask.sum() == 0:
        return torch.tensor(0.0), {}
    pred, label, mask = pred.reshape(-1), label.reshape(-1), mask.reshape(-1)  # (N, 2), (N, 2), (N)
    pred, label = pred[mask], label[mask]
    angle_diff = torch.abs(pred - label)
    angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)  # (N, )
    l1_dist = angle_diff.mean()
    return l1_dist, {'hist_anglediff': angle_diff, 'hist_pred_angle': pred, 'hist_label_angle': label}
