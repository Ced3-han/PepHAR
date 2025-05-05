import sys, os
sys.path.append(os.curdir)
from tqdm import tqdm
from datasets.aa_density_data_module import AADensityDataModule
from models.density_module import DensityModule
from datasets.protein_peptide_dataset import ProteinPeptideDataset
from utils.assertion import assert_eq

from torch.utils.data import DataLoader, random_split
import torch

def test_dataset_reproducibility():
	root = './data/protein_peptide_tensor'
	batch_size = 32
	def _get_data_once():
		data_both = ProteinPeptideDataset(root=root, train=True)
		data_test = ProteinPeptideDataset(root=root, train=False)
		valid_size = int(len(data_both) * 0.2)
		train_size = len(data_both) - valid_size
		data_train, data_valid = random_split(data_both, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
		# data_train = data_both
		# data_valid = data_both
		data1 = next(iter(data_train))
		data2 = next(iter(data_valid))
		data3 = next(iter(data_test))
		return data1, data2, data3
	print('Loading 1st time')
	data1, data2, data3 = _get_data_once()
	print('Loading 2nd time')
	data4, data5, data6 = _get_data_once()
	assert_eq(data1, data4)
	assert_eq(data2, data5)
	assert_eq(data3, data6)

def _to_device(data: dict, device) -> dict:
	return {k: v.to(device) for k, v in data.items()}

def test_data_module():
	print('Loading data...')
	root = './data/protein_peptide_tensor'
	batch_size = 16
	valid_ratio = 0.2
	num_workers = 4
	# max_length = 128
	max_length = 384
	negative_sampling_ratio = 1.0
	data_module = AADensityDataModule(
		root, batch_size, valid_ratio=valid_ratio, num_workers=num_workers,
		max_length=max_length, negative_sampling_ratio=negative_sampling_ratio,
	)
	print('Building model...')
	num_layers = 8
	learning_rate = 1e-4
	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
	model = DensityModule(
		num_layers=num_layers, learning_rate=learning_rate,
	).to(device)

	print('Training...')
	loader_valid = data_module.val_dataloader()
	with torch.no_grad():
		for batch in tqdm(loader_valid, desc="valid", leave=False):
			batch = _to_device(batch, device)
			model.test_step(batch)

if __name__ == '__main__':
    # test_dataset_reproducibility()
	test_data_module()