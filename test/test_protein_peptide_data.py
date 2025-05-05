import argparse
import pickle
import torch
import pickle

config_length = 10081
config_attribute_type_shape = {
	"aa": (torch.int64, (None, )), 
	"pos_heavyatom": (torch.float32, (None, 15, 3)), 
	"is_peptide": (torch.bool, (None, )),
}

def test_protein_peptide_data(data_root='./data/protein_peptide_tensor'):
	with open(f"{data_root}/data.pkl", 'rb') as file:
		data = pickle.load(file)

	data_type = type(data)
	print(f'data_type: {data_type}')
	assert isinstance(data, dict)

	data_len = len(data)
	print(f'data_len: {data_len}')
	assert data_len == config_length

	min_len_case = min(len(case['aa']) for case in data.values())
	max_len_case = max(len(case['aa']) for case in data.values())
	print(f'min_len_case: {min_len_case}')
	print(f'max_len_case: {max_len_case}')

	min_aa_value = min(case['aa'].min() for case in data.values())
	max_aa_value = max(case['aa'].max() for case in data.values())
	print(f'min_aa_value: {min_aa_value}')
	print(f'max_aa_value: {max_aa_value}')


	min_peptide_length = min(case['is_peptide'].sum() for case in data.values())
	max_peptide_length = max(case['is_peptide'].sum() for case in data.values())
	print(f'min_peptide_length: {min_peptide_length}')
	print(f'max_peptide_length: {max_peptide_length}')

	print(f'4 keys: {list(data.keys())[:4]}')

	for key, value in data.items():
		assert isinstance(value, dict)
		for attr, x in value.items():
			assert isinstance(x, torch.Tensor)
			L = x.shape[0]
			config_shape = tuple(x if x is not None else L for x in config_attribute_type_shape[attr][1])
			config_type = config_attribute_type_shape[attr][0]
			# print(attr, x.dtype, x.shape, config_shape)
			assert x.dtype == config_type
			assert x.shape == config_shape
	print('Feature Test Passed')

	print('Pass!')

if __name__ == '__main__':
	test_protein_peptide_data()