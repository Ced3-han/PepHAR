from matplotlib import colors, pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

from evaluate.tools import concatenate_data, coord_from_xm, data_to_batch, to_device, to_numpy
from evaluate.geometry import eular_to_rotation_matrix, rotation_matrix_to_quaternion, construct_3d_basis

color_list = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    # Ligher colors
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]


# def plot_protein(data, show_negative=False, subplots_config=None, use_legend=True):
#     # creating figure
#     # plt.close('all')
#     data = to_numpy(data)

#     is_query = data.get('is_query') if 'is_query' in data else data.get('is_peptide')
#     is_peptide = is_query & (data['aa'] < 20)
#     is_negative = is_query & (data['aa'] == 20)
#     is_protein = ~is_query
#     center = data['coord'][:, 0, :]
#     u1 = data['coord'][:, 1, :] - center
#     u2 = data['coord'][:, 2, :] - center

#     if subplots_config:
#         ax = plt.subplot(*subplots_config, projection='3d')
#     else:
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')

#     ## Plot peptide
#     xs, ys, zs = center[is_peptide].T
#     cs = data['aa'][is_peptide]
#     for c in range(20):
#         idx = cs == c
#         if idx.sum() == 0:
#             continue
#         ax.scatter(xs[idx], ys[idx], zs[idx], color=color_list[c], label=f'type {c}')
#     quiver_length = 1.0
#     xu1, yu1, zu1 = u1[is_peptide].T
#     ax.quiver(xs, ys, zs, xu1, yu1, zu1, length=quiver_length, color='b')
#     xu2, yu2, zu2 = u2[is_peptide].T
#     ax.quiver(xs, ys, zs, xu2, yu2, zu2, length=quiver_length, color='g')

#     ## Plot receptor
#     xs, ys, zs = center[is_protein].T
#     ax.scatter(xs, ys, zs, color='gray', alpha=0.1, label=f'receptor')

#     ## Plot Negative
#     if show_negative:
#         xs, ys, zs = center[is_negative].T
#         ax.scatter(xs, ys, zs, color='cyan', alpha=0.2, label=f'x')
#         quiver_length = 1.0
#         xu1, yu1, zu1 = u1[is_negative].T
#         ax.quiver(xs, ys, zs, xu1, yu1, zu1, length=quiver_length, color='b', alpha=0.2)
#         xu2, yu2, zu2 = u2[is_negative].T
#         ax.quiver(xs, ys, zs, xu2, yu2, zu2, length=quiver_length, color='g', alpha=0.2)

#     # setting title and labels
#     # ax.set_title("3D plot")
#     ax.set_xlabel('x-axis')
#     ax.set_ylabel('y-axis')
#     ax.set_zlabel('z-axis')
#     if use_legend:
#         ax.legend(loc='right', bbox_to_anchor=(0.0, 0.5))



def generate_position_grid(anchor_pos_coord, x_min, x_max, y_min, y_max, num):
	X, Y = torch.meshgrid(torch.linspace(x_min, x_max, num), torch.linspace(y_min, y_max, num))
	K = X.shape[0] * X.shape[1]
	generated_pos_coord = torch.tile(anchor_pos_coord.unsqueeze(0), (K, 1, 1))
	generated_pos_coord[:, :, 0] += X.reshape(K, 1)
	generated_pos_coord[:, :, 1] += Y.reshape(K, 1)

	data_grid = {
		'aa': torch.tensor([20] * K),
		'coord': generated_pos_coord,
		'is_query': torch.tensor([True] * K),
	}
	return data_grid, X, Y


def generate_orintation_grid(anchor_pos_coord, x_min, x_max, y_min, y_max, num, type='xy'):
	assert anchor_pos_coord.shape == (3, 3)
	X, Y = torch.meshgrid(torch.linspace(x_min, x_max, num), torch.linspace(y_min, y_max, num))
	K = X.shape[0] * X.shape[1]

	anchor_x = anchor_pos_coord[0, :]  # (3, )
	anchor_mat = construct_3d_basis(anchor_pos_coord[0], anchor_pos_coord[1], anchor_pos_coord[2])  # (3, 3)
	if type == 'xy':
		x_array, y_array, z_array = X.reshape(-1), Y.reshape(-1), torch.zeros_like(X.reshape(-1))
	elif type == 'yz':
		x_array, y_array, z_array = torch.zeros_like(X.reshape(-1)), X.reshape(-1), Y.reshape(-1)
	elif type == 'zx':
		x_array, y_array, z_array = Y.reshape(-1), torch.zeros_like(X.reshape(-1)), X.reshape(-1)
	else:
		raise NotImplementedError
	rotation_mat = eular_to_rotation_matrix(x_array, y_array, z_array)  # (K, 3, 3)
	generate_mat = torch.matmul(rotation_mat, anchor_mat)  # (K, 3, 3)
	generated_pos_coord = coord_from_xm(anchor_x.tile(K, 1), generate_mat)

	data_grid = {
		'aa': torch.tensor([20] * K),
		'coord': generated_pos_coord,
		'is_query': torch.tensor([True] * K),
	}
	return data_grid, X, Y


def plot_protein(data, density_model=None, density_model_type=20, device='cpu'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot Peptide
    xs, ys, zs = data['pep_coord'][:, 0, :].T
    box = np.array([xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()]).reshape(3, 2)
    sz = (box[:, 1] - box[:, 0]).max()
    md = (box[:, 1] + box[:, 0]) / 2
    box = np.stack([md - sz / 2, md + sz / 2], axis=1)
    ax.set_xlim(box[0, 0], box[0, 1])
    ax.set_ylim(box[1, 0], box[1, 1])
    ax.set_zlim(box[2, 0], box[2, 1])

    cs = data['pep_aa']
    for c in range(20):
        idx = cs == c
        if idx.sum() == 0:
            continue
        ax.scatter(xs[idx], ys[idx], zs[idx], label=f'type {c}')
    for i in range(xs.shape[0] - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs[i], zs[i + 1]], color='black', linestyle='--', alpha=0.5)

    ## Plot peptide framework
    quiver_length = 1.5
    u1 = data['pep_coord'][:, 1, :] - data['pep_coord'][:, 0, :]
    u2 = data['pep_coord'][:, 2, :] - data['pep_coord'][:, 0, :]
    xu1, yu1, zu1 = u1.T
    ax.quiver(xs, ys, zs, xu1, yu1, zu1, length=quiver_length, color='b')
    xu2, yu2, zu2 = u2.T
    ax.quiver(xs, ys, zs, xu2, yu2, zu2, length=quiver_length, color='g')

    ## Plot receptor
    xs, ys, zs = data['rec_coord'][:, 0, :].T
    ax.scatter(xs, ys, zs, color='gray', alpha=0.05, label=f'receptor', marker='s')

    ## Plot query
    if 'qry_coord' in data and density_model is not None:
        xs, ys, zs = data['qry_coord'][:, 0, :].T

        # ax.scatter(xs, ys, zs, color='black', alpha=0.05, label=f'query', marker='s')
        # quiver_length = 1.5
        # u1 = data['qry_coord'][:, 1, :] - data['qry_coord'][:, 0, :]
        # u2 = data['qry_coord'][:, 2, :] - data['qry_coord'][:, 0, :]
        # xu1, yu1, zu1 = u1.T
        # ax.quiver(xs, ys, zs, xu1, yu1, zu1, length=quiver_length, color='b')
        # xu2, yu2, zu2 = u2.T
        # ax.quiver(xs, ys, zs, xu2, yu2, zu2, length=quiver_length, color='g')

        query_data = {
            'rec_coord': data['rec_coord'],
            'qry_coord': data['qry_coord'],
            'pep_coord': data['pep_coord'][:0],
            'pep_aa': data['pep_aa'][:0],
            'rec_aa': data['rec_aa'],
            'qry_aa': data['qry_aa'],
        }
        batch_data = to_device(data_to_batch(query_data), device)
        with torch.no_grad():
            logits = density_model(batch_data).squeeze(0).detach().cpu()
        prob = F.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        levels = np.linspace(0.0, 1.0, 11)
        for j in range(len(levels) - 1):
            idx = (levels[j] <= prob[:, density_model_type]) & (prob[:, density_model_type] < levels[j + 1])
            # print(idx)
            ax.scatter(xs[idx], ys[idx], zs[idx], color='black', alpha=levels[j], label=f'level-{levels[j]:.2f}', marker='s')

        for c in range(20):
            idx = pred == c
            if idx.sum() == 0:
                continue
            ax.scatter(xs[idx], ys[idx], zs[idx], label=f'type {c}')

    ax.set_title("3D plot")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.legend(loc='right', bbox_to_anchor=(0.0, 0.5))
    # displaying the plot 


def plot_prediction_contour(data, X, Y, density_model=None, density_model_type=20, device='cpu', xlabel='x', ylabel='y'):

    K = X.shape[0] * X.shape[1]
    query_data = {
        'rec_coord': data['rec_coord'],
        'qry_coord': data['qry_coord'],
        'pep_coord': data['pep_coord'][:0],
        'pep_aa': data['pep_aa'][:0],
        'rec_aa': data['rec_aa'],
        'qry_aa': data['qry_aa'],
    }
    batch_data = to_device(data_to_batch(query_data), device)
    with torch.no_grad():
        logits = density_model(batch_data).squeeze(0).detach().cpu()
    prob = F.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1)

    Z = prob[:, density_model_type].reshape(X.shape[0], X.shape[1])

    # if target_type < 0:
    #     Z = torch.log(1.0 - torch.exp(log_prob[:, abs(target_type)])).reshape(X.shape[0], X.shape[1])
    # else:
    #     Z = log_prob[:, target_type].reshape(X.shape[0], X.shape[1])
    print(X.shape, Y.shape, Z.shape)

    fig = plt.figure()
    ax = plt.axes()
    clev = np.concatenate([np.arange(0.0, 1.0 + 1e-6, .01)]) #Adjust the .001 to get finer gradient
    norm = colors.BoundaryNorm(boundaries=clev, ncolors=256, extend='both')
    cset = plt.contourf(X, Y, Z, clev, norm=norm, cmap=plt.cm.coolwarm)
    contour = plt.contour(X, Y, Z, [-3.0, -2.0, -1.0], alpha=0.5)
    plt.clabel(contour, fontsize=10, inline=1)
    plt.colorbar(cset)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.imshow(Z)

    