import copy
import torch
from torchvision.transforms import Compose


_TRANSFORM_DICT = {}


def register_transform(name):
    # 注意：带参数的装饰器会被立刻执行
    def decorator(cls):
        _TRANSFORM_DICT[name] = cls
        return cls
    return decorator


def get_transform(cfg):
    # get_transform(cfg.transform)
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')] # object name
        tfms.append(cls(**t_dict)) # initialize object with defined keys
    return Compose(tfms)


def _index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


