import sys, os
sys.path.append(os.curdir)
from datasets.protein_peptide_dataset import ProteinPeptideDataset
from datasets.padding_collate import PaddingCollate
from datasets.aa_density_data_module import ToAADensityTask

from torch.utils.data import DataLoader, random_split
import torch

def test_protein_peptide_dataset():
	root = './data/protein_peptide_tensor'
	data_train = ProteinPeptideDataset(root=root, train=True)
	print('#items:', len(data_train))
	N = len(data_train)
	assert isinstance(data_train[0], dict)
	assert isinstance(data_train[N - 1], dict)
	try :
		wrong = data_train[N]
	except IndexError:
		print('IndexError passed')
	else:
		raise AssertionError('dataset[N] should not be accessible')
	
	data_test_wrong = ProteinPeptideDataset(root=root, train=False, split_ratio=0.7)
	assert (set(data_train.name_list) & set(data_test_wrong.name_list)) != set()

	data_test = ProteinPeptideDataset(root=root, train=False)
	assert (set(data_train.name_list) & set(data_test.name_list)) == set()
	assert data_test.name_list == sorted(data_test.name_list)
	# print(data_test.name_list)


def test_protein_peptide_dataloader():
	root = './data/protein_peptide_tensor'
	data = ProteinPeptideDataset(root=root, train=True, transform=ToAADensityTask())
	train_loader = DataLoader(data, batch_size=32, shuffle=False, collate_fn=PaddingCollate())
	batch = next(iter(train_loader))
	assert isinstance(batch, dict)
	for key, value in batch.items():
		print(key)
		print(value.shape)
	# print(batch['mask'][0, :])
	# print(batch['mask'][1, :])

	## aa should not be unknown in the unmasked area
	# flag = ((batch['mask'] == False) | ((0 <= batch['aa']) & (batch['aa'] < 20)))
	# for i in range(flag.shape[0]):
	#     for j in range(flag.shape[1]):
	#         if flag[i, j] == False:
	#             print(i, j, batch['aa'][i, j], batch['mask'][i, j], batch['is_query'][i, j])
	# print(batch['mask'])
	# print(batch['aa'])
	assert (
		(batch['mask'] == False) | 
		(batch['is_query'] & (0 <= batch['aa']) & (batch['aa'] <= 20)) | 
		((0 <= batch['aa']) & (batch['aa'] < 20))
	).all()

	## minimal padding length should be 0-7
	print('padding length: ', [(batch['mask'][i, :] == False).sum().item() for i in range(batch['mask'].shape[0])])
	assert min((batch['mask'][i, :] == False).sum() for i in range(batch['mask'].shape[0])) < 8

	## in all cases, length of peptide should larger than 0
	print('query length: ', [batch['is_query'][i, :].sum().item() for i in range(batch['is_query'].shape[0])])
	assert min(batch['is_query'][i, :].sum() for i in range(batch['is_query'].shape[0])) > 0



if __name__ == '__main__':
	test_protein_peptide_dataset()
	# test_protein_peptide_dataloader()
	# test_transform_f()
	# test_transform_f_with_dataloader()
	# test_dataset_reproducibility()
