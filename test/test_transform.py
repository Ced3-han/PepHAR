import sys, os

sys.path.append(os.curdir)

from datasets.aa_density_data_module import ToAADensityTask
from datasets.sample_negative_type_transform import SampleNegativeType, random_rotation_matrix
from datasets.truncate_protein_transform import TruncateProtein
from utils.assertion import assert_eq

from torch.utils.data import DataLoader, random_split
import torch
import time

def test_negative_sampling():
    start_time = time.process_time()
    R = random_rotation_matrix(100000)
    max_error = (((R ** 2).sum(dim=2)) - 1).abs().max()
    print(f'Random rotation matrix error: {max_error}')
    assert max_error < 3e-6
    end_time = time.process_time()
    print(f'Random rotation matrix generation time: {end_time - start_time}s')

    transform = SampleNegativeType(ratio=1.0)
    L = 100
    def _generate_data(L):
        return {
            'aa': torch.randint(0, 20, (L, )),
            "pos_coord": torch.randn(L, 3, 3), 
            "is_query": torch.rand(L) < 0.1,
        }
    data = _generate_data(100)
    new_data = transform(data)
    target_size = L + data['is_query'].sum().item()
    assert new_data['aa'].shape == (target_size, )
    assert new_data['coord'].shape == (target_size, 3, 3)
    assert new_data['is_query'].shape == (target_size, )

def test_truncation():
    transform = TruncateProtein(max_length=128, reserve_ratio=1.0)
    L = 100
    def _generate_data(L):
        return {
            'aa': torch.randint(0, 20, (L, )),
            "pos_heavyatom": torch.randn(L, 3, 15), 
            "is_peptide": torch.rand(L) < 0.1,
        }
    data100 = _generate_data(100)
    new_data100 = transform(data100)
    assert_eq(new_data100, data100)
    data200 = _generate_data(200)
    new_data200 = transform(data200)
    target_size = 128 - data200['is_peptide'].sum().item()
    # print(new_data200['aa'].shape, (target_size , ))
    assert new_data200['aa'].shape == (target_size, )
    assert new_data200['pos_heavyatom'].shape == (target_size, 3, 15)
    assert new_data200['is_peptide'].shape == (target_size, )


def test_aa_density_task_transform():
    transform = ToAADensityTask()
    L = 100
    data = {
        'aa': torch.randint(0, 20, (L, )),
        "pos_heavyatom": torch.randn(L, 15, 3), 
        "is_peptide": torch.rand(L) < 0.1,
    }
    data = transform(data)
    assert set(data.keys()) == set(['aa', 'coord', 'is_query'])

if __name__ == '__main__':
    test_negative_sampling()
