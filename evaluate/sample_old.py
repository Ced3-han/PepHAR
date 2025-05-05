import math
import torch
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from .geometry import *
from .tools import coord_from, data_to_batch, generate_next_coord, to_device

structure_type = {
    'aa': torch.LongTensor,
    'coord': torch.Tensor,
    'is_peptide': torch.BoolTensor,
}
density_task_type = {
    'aa': torch.LongTensor,
    'coord': torch.Tensor,
    'is_query': torch.BoolTensor,
}
pridiction_task_type = {
    'aa': torch.LongTensor,
    'coord': torch.Tensor,
    'is_peptide': torch.BoolTensor,
    'anchor': torch.LongTensor,
    'label': torch.Tensor,
}


def _sample_seeds_by_maximizing_density(density_model, device, receptor, x_init,
                                        o_init, target_type, n_steps=1000,epsilon=1e-2,
                                        verbose=False, epsilon_strategy='none',
                                        x_strategy='none'):
    """
    :param x_init: (N,
    :param o_init: (N,
    :return:
    """
    assert x_init.shape[0] == o_init.shape[0]
    K = x_init.shape[0] # peptide length
    x = nn.Parameter(x_init.clone(), requires_grad=True if x_strategy != 'fixed' else False)
    o = nn.Parameter(o_init.clone(), requires_grad=True)
    coord_list = []
    prob_list = []
    aa_list = []
    optimizer = SGD([x, o], lr=epsilon)
    for t in tqdm(range(n_steps), desc="seed", leave=False):
        new_coord = coord_from(x, o)
        data_query = {
            'aa': torch.cat([torch.tensor([20] * K), receptor.get('aa')], dim=0),
            'coord': torch.cat([new_coord, receptor.get('coord')], dim=0),
            'is_query': torch.cat([torch.tensor([True] * K), receptor.get('is_peptide')], dim=0),
        }
        batch_data = to_device(data_to_batch(data_query), device)
        logits = density_model(batch_data).squeeze(0)
        log_prob = nn.functional.log_softmax(logits, dim=-1)  # (L, 21)
        target_score = log_prob[:K, target_type] if target_type >= 0 else torch.log(
            1.0 - torch.exp(log_prob[:K, abs(target_type)]))
        gd_loss = -target_score.mean()
        optimizer.zero_grad()
        gd_loss.backward()
        optimizer.step()
        noise_epsilon = math.sqrt(2 * epsilon) if epsilon_strategy != 'deterministic' else 0.0
        if x_strategy != 'fixed':
            x_noise = torch.randn(x.shape)
            x.data.add_(noise_epsilon * x_noise)
        o_noise = torch.randn(o.shape)
        o.data.add_(noise_epsilon * o_noise)
        o.data.div_(torch.sqrt((o ** 2).sum(dim=-1).unsqueeze(-1)))
        coord_list.append(new_coord.detach().cpu())
        prob_list.append(target_score.detach().cpu())
        aa_list.append(log_prob[:K, :].argmax(dim=-1).detach().cpu())
        if verbose and t % (max(1, n_steps // 10)) == 0:
            print(f'step:{t}, loss:{gd_loss}, log_prob:{log_prob[:K, target_type].detach().cpu().numpy()}')
    return coord_list, aa_list, prob_list


class Sampler():
    def __init__(self, data, *args, **kwargs) -> None:
        assert all([k in data for k in ['aa', 'coord', 'is_peptide']])
        is_peptide = data['is_peptide']
        is_protein = ~is_peptide
        self.receptor = {k: v[is_protein] for k, v in data.items()}
        if is_peptide.any():
            self.pocket = torch.stack(
                [data['coord'][is_peptide, 0, :].min(dim=0)[0], data['coord'][is_peptide, 0, :].max(dim=0)[0]], dim=0)
            self.coord_lib = [data['coord'][is_peptide, :, :][i] for i in range(is_peptide.sum())]
            self.aa_lib = [data['aa'][is_peptide][i] for i in range(is_peptide.sum())]
        else:
            raise ValueError('No peptide found in the data')

    def sample(N):
        raise NotImplemented


class AutoregressiveSampler(Sampler):
    def __init__(self, receptor, density_model, prediction_model, device='cpu') -> None:
        super().__init__(receptor)
        self.prediction_model = prediction_model
        self.density_model = density_model
        self.device = device

    def _sample_next(self, coord_all, type_all):
        """
            :param coord_all: (L, 3, 3)
            :param type_all: (L)
        """
        assert coord_all.dim() == 3

        L = len(coord_all)
        data_query = {
            'aa': torch.cat([type_all, self.receptor.get('aa')], dim=0),
            'coord': torch.cat([coord_all, self.receptor.get('coord')], dim=0),
            'is_peptide': torch.cat([torch.tensor([True] * L), self.receptor.get('is_peptide')], dim=0),
            'anchor': torch.tensor(L - 1),
        }
        batch_data = to_device(data_to_batch(data_query), self.device) # single can also
        with torch.no_grad():
            features = self.prediction_model._batch_to_feature(batch_data)
            type_logits, type_angle = self.prediction_model(*features)  # (1, 21), (1, 21, 4)
        type_logits = type_logits.squeeze(0).cpu()
        type_angle = type_angle.squeeze(0).cpu()
        assert type_logits.shape == (21,)
        assert type_angle.shape == (21, 4)
        prob = F.softmax(type_logits, dim=-1)  # (21, )
        # type_new = prob.multinomial(1).squeeze(-1), may have UNK here
        type_new = prob[:20].multinomial(1).squeeze(-1)
        angle = type_angle[type_new]  # (4, )

        coord_new = generate_next_coord(coord_all[-1], angle)  # (3, 3)
        #return coord_new, type_new, torch.log(prob[type_new])
        return coord_new, type_new, [type_logits]

    def _check_termination(self, new_coord, new_type):
        """
            :param coord_last: (3, 3)
            :param type_last: ()
        """
        assert new_coord.shape == (3, 3)
        assert new_type.shape == ()
        data_query = {
            'aa': torch.cat([torch.tensor([20]), self.receptor.get('aa')], dim=0),
            'coord': torch.cat([new_coord.reshape(1, 3, 3), self.receptor.get('coord')], dim=0),
            'is_query': torch.cat([torch.tensor([True]), self.receptor.get('is_peptide')], dim=0),
        }
        batch_data = to_device(data_to_batch(data_query), self.device)
        # print(batch_data)
        with torch.no_grad():
            pred_logits = self.density_model(batch_data)  # (1, L, 21)
        target_logits = pred_logits[0, 0]
        # print(target_logits.shape)
        assert target_logits.shape == (21,)
        prob = F.softmax(target_logits, dim=-1)
        prob_negative_type = prob[20].item()
        return True if prob_negative_type > 0.5 else False

    def _sample_peptide(self, L, x_init, o_init, a_init):
        """
            :param x_init: (3,)
            :param o_init: (4,)
            :param a_init: (1,)
        """
        assert x_init.shape == (3,)
        assert o_init.shape == (4,)
        assert a_init.shape == (1,)
        for l in range(0, L): # delete tqdm here
            if l == 0:
                coord_all = coord_from(x_init, o_init).reshape(1, 3, 3)
                type_all = a_init.reshape(1)
                # prob_all = torch.zeros(1)
                #prob_all = torch.ones(1)
                logits = [F.one_hot(a_init,num_classes=21).view(-1)] # (21)
                continue
            # new_coord, new_type, new_prob = self._sample_next(coord_all, type_all) # use prediction model
            new_coord, new_type, new_logits = self._sample_next(coord_all, type_all)  # use prediction model
            termination_flag = self._check_termination(new_coord, new_type)
            if termination_flag:
                break
            coord_all = torch.cat([coord_all, new_coord.reshape(-1, 3, 3)], dim=0)  # (L + 1, 3, 3)
            type_all = torch.cat([type_all, new_type.reshape(-1)], dim=0)  # (L + 1, )
            logits += new_logits
            if 20 in type_all:
                print(type_all)
                raise ValueError
            # prob_all = prob_all + new_prob
        return coord_all, type_all, torch.stack(logits,dim=0)

    def sample(self, N, L, seed_strategy='random'):
        # sample N peptides with length L
        #print(self.pocket)
        peptide_list = []
        prob_list = [] # logit list infact

        if seed_strategy == 'random':
            ## Random + Optimization
            x_init = self.pocket.mean(dim=0).tile(N, 1)
            o_init = F.normalize(torch.randn(4, ), dim=-1).tile(N, 1)
            coord_seed_list, aa_seed_list, prob_seed_list = _sample_seeds_by_maximizing_density(
                self.density_model, self.device, self.receptor,
                x_init.reshape(-1, 3), o_init.reshape(-1, 4), target_type=-20,
                n_steps=100, epsilon=1e-2)
            idx_seed = torch.argmax(torch.stack(prob_seed_list, dim=0), dim=0)  # (N, )
            assert idx_seed.shape == (N,)
            # print(prob_seed_list)
            # print(coord_seed_list)
            coord_seed_tensor = torch.stack(coord_seed_list, dim=0)  # (T, N, 3, 3)
            aa_seed_tensor = torch.stack(aa_seed_list, dim=0)  # (T, N,)
            prob_seed_tensor = torch.stack(prob_seed_list, dim=0)  # (T, N, )
            coord_seed_all = coord_seed_tensor.gather(0, idx_seed.reshape(1, N, 1, 1).expand(1, N, 3, 3)).squeeze(
                0)  # (N, 3, 3)
            a_seed_all = aa_seed_tensor.gather(0, idx_seed.reshape(1, N).expand(1, N)).squeeze(0)  # (N, )
            prob_seed_all = prob_seed_tensor.gather(0, idx_seed.reshape(1, N).expand(1, N)).squeeze(0)  # (N, )
            x_seed_all = coord_seed_all[:, 0]
            o_seed_all = rotation_matrix_to_quaternion(
                construct_3d_basis(coord_seed_all[:, 0], coord_seed_all[:, 1], coord_seed_all[:, 2]))

        for i in range(N):
            if seed_strategy == 'random':
                a_seed, x_seed, o_seed = a_seed_all[i], x_seed_all[i], o_seed_all[i]
                print(i, a_seed, prob_seed_all[i], x_seed, o_seed)

            elif seed_strategy == 'ground_truth':
                ## Use Ground-truth
                # self.aa_lib = [data['aa'][is_peptide][i] for i in range(is_peptide.sum())]
                # rand_idx = torch.randint(len(self.aa_lib), (1,)), start from the first
                rand_idx = 0  # TODO: TEMPORARY
                coord_seed = self.coord_lib[rand_idx]
                a_seed = self.aa_lib[rand_idx].reshape(-1)
                x_seed = coord_seed[0]
                o_seed = rotation_matrix_to_quaternion(construct_3d_basis(coord_seed[0], coord_seed[1], coord_seed[2]))
            else:
                raise NotImplementedError

            # Start generation
            coord_all, type_all, prob_all = self._sample_peptide(L, x_seed.reshape(3), o_seed.reshape(4),
                                                                 a_seed.reshape(1))
            peptide_list.append({
                'aa': type_all,
                'coord': coord_all,
                'is_peptide': torch.tensor([True] * len(type_all)),
            })
            prob_list.append(prob_all)
        return peptide_list, torch.stack(prob_list, dim=0)
