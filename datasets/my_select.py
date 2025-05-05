import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import math


def select(v, index):
    if isinstance(v, list):
        return [v[i] for i in index]
    elif isinstance(v, torch.Tensor):
        return v[index]
    else:
        raise NotImplementedError

def masked_select(v, mask):
    if isinstance(v, str):
        return ''.join([s for i, s in enumerate(v) if mask[i]])
    elif isinstance(v, list):
        return [s for i, s in enumerate(v) if mask[i]]
    elif isinstance(v, torch.Tensor):
        return v[mask]
    else:
        raise NotImplementedError

def _index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


def _index_select_data(data, index):
    return {
        k: _index_select(v, index, data['aa'].size(0))
        for k, v in data.items()
    }


def _mask_select(v, mask):
    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
        return v[mask]
    elif isinstance(v, list) and len(v) == mask.size(0):
        return [v[i] for i, b in enumerate(mask) if b]
    else:
        return v


def _mask_select_data(data, mask):
    return {
        k: _mask_select(v, mask)
        for k, v in data.items()
    }


