from tqdm import tqdm
from datasets.aa_prediction_data_module import AAPredictionDataModule, ToAAPredicitonTask
from datasets.protein_peptide_dataset import ProteinPeptideDataset
import torch.nn.functional as F
from datasets.aa_prediction_data_module import ToAAPredicitonTask
from datasets.protein_peptide_dataset import ProteinPeptideDataset
from datasets.truncate_protein_transform import TruncateProtein
from utils.tools import get_strucutre, get_next_structure
import torch
from utils.common import construct_3d_basis
from torchvision import transforms

def test_prediction_task_transform():
	transfrom = ToAAPredicitonTask()
	root = './data/protein_peptide_tensor'
	dataset = ProteinPeptideDataset(root=root, train=True, transform=transfrom)
	N = len(dataset)
	N = 20
	for i in tqdm(range(N)):
		# print(i)
		data = dataset[i]
	print('data info', {k: (v.shape, v.dtype) for k, v in data.items()})
	# print(data)

def test_data_module():
	root = '/home/chentong/ws/pep-design/data/protein_peptide_tensor'
	data_module = AAPredictionDataModule(
		root, 16, num_workers=16, max_length=256,
	)
	loader = data_module.train_dataloader()
	cnt = 0
	for i in range(10):
		for data in tqdm(loader):
			cnt += data.get('aa').shape[0]

def test_anchor_and_angle():
	root = '/home/chentong/ws/pep-design/data/protein_peptide_tensor'

	def _get_ground_truth(id):
		data = get_strucutre(root, id)[1]
		is_peptide = data['is_query']
		pos_coord = data['coord'][is_peptide]
		return pos_coord
	
	def _get_new(id):
		transfrom = transforms.Compose([
			TruncateProtein(256),
			ToAAPredicitonTask(),
		])
		dataset = ProteinPeptideDataset(root=root, train=True, transform=transfrom)
		# N = len(dataset)
		# N = 20
		# for i in tqdm(range(N)):
		#     d = dataset[i]
		#     print(i, d.get('is_peptide').sum(), d.get('anchor'))
		data = dataset[0]
		# print('data info', {k: v.shape for k, v in data.items()})
		# print(data)
		data = dataset[id]
		anchor = data.get('anchor')
		if anchor < 0:
			return None
		label = data.get('label').reshape(1, -1)
		# print(anchor, label)
		pos_coord = data.get('coord')
		x_last = pos_coord[anchor, 0, :]  # (K, 3)
		# print(pos_coord[3])
		# print(x_last)
		mat = construct_3d_basis(pos_coord[anchor, 0, :], pos_coord[anchor, 1, :], pos_coord[anchor, 2, :])  # (K, 3, 3)
		structure = get_next_structure(label[:, 0], label[:, 1], label[:, 2], label[:, 3])   # (K, 3)
		new_x0 = x_last + torch.matmul(mat, structure['CA2'].unsqueeze(-1)).squeeze(-1)
		new_x1 = x_last + torch.matmul(mat, structure['C2'].unsqueeze(-1)).squeeze(-1)
		new_x2 = x_last + torch.matmul(mat, structure['N2'].unsqueeze(-1)).squeeze(-1)
		new_pos_coord = torch.stack([new_x0, new_x1, new_x2], dim=-2)  # (K, 3, 3)
		return new_pos_coord
	
	for id in range(10):
		new_pos_coord = _get_new(id)  # (3, 3)
		if new_pos_coord is None:
			continue
		peptide_pos_coord = _get_ground_truth(id)  # (L, 3, 3)
		diff = ((new_pos_coord.reshape(1, 3, 3) - peptide_pos_coord) ** 2).mean(dim=-1).mean(dim=-1)
		print(diff)
		print('pass', id)
		assert diff.min() < 1e-1

if __name__ == '__main__':
	test_prediction_task_transform()
	# test_anchor_and_angle()
	# test_data_module()
