import torch
from utils.convert import _unzip

def test_unzip_last():
    x = torch.randn([2, 3, 4])
    a = _unzip(x)
    assert len(a) == 4
    assert a[0].shape == (2, 3)

if __name__ == '__main__':
    test_unzip_last()
