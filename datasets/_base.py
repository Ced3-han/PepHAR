from .transforms import get_transform


_DATASET_DICT = {}


def register_dataset(name):
    def decorator(cls):
        _DATASET_DICT[name] = cls
        return cls
    return decorator


def get_dataset(cfg):
    transform = get_transform(cfg.transform) if 'transform' in cfg else None
    return _DATASET_DICT[cfg.type](cfg, transform=transform)