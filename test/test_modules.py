import sys, os

from models.prediction_module import PredictionModule
sys.path.append(os.curdir)

import torch
from models.density_module import DensityModule
from torchinfo import summary
from argparse import Namespace

def _to(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}


def test_f_forward():
    gpu_id = 5
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    model = DensityModule().to(device)
    N, L = 4, 10
    batch = {
        'aa': torch.randint(0, 21, (N, L)),
        'coord': torch.randn(N, L, 15, 3),
        'mask': torch.rand(N, L) < 0.9,
        'is_query': torch.rand(N, L) < 0.1,
    }
    batch = _to(batch, device)
    model.training_step(batch)

def test_g_forward():
    print('test_g_forward')
    gpu_id = 0
    print('create model')
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    model = PredictionModule().to(device)
    print('create data')
    N, L = 16, 10
    batch = {
        'aa': torch.randint(0, 21, (N, L)),
        'coord': torch.randn(N, L, 3, 3),
        'mask': torch.rand(N, L) < 0.9,
        'is_peptide': torch.rand(N, L) < 0.1,
        'anchor': torch.randint(-1, L, (N, )),
        'label_angle': torch.rand(N, 4),
        'label_type': torch.randint(0, 20, (N,) ),
    }
    batch = _to(batch, device)
    print('forward')
    model.training_step(batch)

if __name__ == '__main__':
    # test_protein_peptide_dataset()
    test_g_forward()
