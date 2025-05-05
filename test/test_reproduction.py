import torch
import torch.nn.functional as F
from models.density_module import DensityModule
from utils.tools import concatenate_data, data_to_batch, generate_position_grid, get_strucutre, load_module, to_device

def test_load_density_module():
	device = 'cuda:0'
	id = 9
	
	def _run_model(model, data):
		batch_data = to_device(data_to_batch(data), device)
		with torch.no_grad():
			logits = model(batch_data).squeeze(0).detach().cpu()
		pred = torch.argmax(logits, dim=1)
		log_prob = F.log_softmax(logits, dim=1)
		# print(log_prob)
		return log_prob

	def _run_all():
		model = load_module('/home/chentong/ws/pep-design/log/20220625_layer8_trunc256_sig5_noise02/model_100.pt', DensityModule, device=device)
		data_root = '/home/chentong/ws/pep-design/data/protein_peptide_tensor'
		data = get_strucutre(data_root, id)[1]
		print(list(data.keys()))
		data_grid, _, _ = generate_position_grid(data.get('coord')[data.get('is_query')][0], -10.0, 10.0, -10.0, 10.0, 20)
		data_grid_all = concatenate_data(data_grid, data)
		return _run_model(model, data_grid_all)

	res1 = _run_all()
	res2 = _run_all()
	print(res1)
	print(res2)
	assert (res1 == res2).all()


if __name__ == '__main__':
	test_load_density_module()
