from evaluate.tools import data_to_batch, generate_next_coord, generate_prev_coord
from utils.misc import load_config
import torch
from models import get_model
from Bio.SVDSuperimposer import SVDSuperimposer
import torch.nn.functional as F
from utils.train import recursive_to
from modules.common.geometry import dihedral_from_four_points
from torch.optim import SGD, Adam
from evaluate.geometry import rotation_matrix_to_quaternion, construct_3d_basis
from evaluate.tools import coord_from
from torch.distributions.von_mises import VonMises

class AnchorBasedSampler():
    def __init__(self, args):
        self._init_models(
            density_config_path=args.density_config_path,
            density_param_path=args.density_param_path,
            prediction_config_path=args.prediction_config_path,
            prediction_param_path=args.prediction_param_path,
            device=args.device,
        )


    def _init_models(self, density_config_path, density_param_path, prediction_config_path, prediction_param_path, device):
        self.device = device

        density_config,_ = load_config(density_config_path)
        density_model = get_model(density_config.model).to(device)
        density_model.load_state_dict(torch.load(density_param_path, map_location=device)['model'])
        density_model.eval()
        self.density_model = density_model
        
        prediction_config,_ = load_config(prediction_config_path)
        prediction_model = get_model(prediction_config.model).to(device)
        prediction_model.load_state_dict(torch.load(prediction_param_path, map_location=device)['model'])
        prediction_model.eval()
        self.prediction_model = prediction_model

    def _sample_anchor(self, data, coord_init, aa_init, n_steps=100, lr=3e-2, noise_eps=1e-2, verbose=False):
        assert coord_init.shape == (3, 3)
        x_init = coord_init[0]
        o_init = rotation_matrix_to_quaternion(construct_3d_basis(coord_init[0], coord_init[1], coord_init[2]))
        par_x = torch.tensor(x_init, requires_grad=True).to(self.device)  # L, 3
        par_o = torch.tensor(o_init, requires_grad=True).to(self.device)  # L, 4
        par_cls = torch.tensor(torch.zeros(20).float().to(self.device), requires_grad=True)
        par_T = torch.tensor(torch.zeros(1).to(self.device), requires_grad=True)
        optimizer = Adam([par_x, par_o, par_cls, par_T], lr=lr)

        for t in range(n_steps):
            new_coord = coord_from(par_x, par_o)
            data_query = {
                'rec_coord': data['rec_coord'],
                'rec_aa': data['rec_aa'],
                'pep_coord': data['pep_coord'][:0],
                'pep_aa': data['pep_aa'][:0],
                'qry_coord': torch.cat([new_coord.unsqueeze(0)]),
                'qry_aa': torch.tensor([0]),
            }
            logits = self.density_model(recursive_to(data_to_batch(data_query, collate_type='default'), self.device))  # (B, L2, 4)
            logits = logits[0, 0]  # (21, )
            # print(logits)
            prob = F.softmax(logits, -1)
            assert prob.shape == (21, )

            obj_cls = -torch.log((prob[:20] * F.softmax(par_cls / par_T.exp())).sum(-1))
            obj = obj_cls
            if t % (n_steps // 5) == 0 and verbose:
                print(f'{t}/{n_steps}, obj: {obj.item():.3f}, x: {par_x.detach().cpu().numpy()}, o: {par_o.detach().cpu().numpy()}, '
                    f'par_T_exp: {par_T.exp().item():.3f}, cls:{par_cls.argmax(-1).item()}, max_cls: {F.softmax(par_cls / par_T.exp()).max().item():.3f}', )
                # print(prob)

            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
            
            par_x.data.add_(noise_eps * torch.randn_like(par_x))
            par_o.data.add_(noise_eps * torch.randn_like(par_o))
            par_o.data.div_(torch.sqrt((par_o ** 2).sum(dim=-1).unsqueeze(-1)))

        coord = coord_from(par_x, par_o).detach()
        aa = par_cls.argmax(-1).detach()
        return coord, aa


    def _get_next(self, data, frag_list, dist_list, extend_strategy='random', verbose=False):
        # print(frag_list)
        # print([aa for coords, aa in frag_list])
        pep_coord = torch.cat([coords for coords, aa in frag_list], dim=0)
        # print(pep_coord.shape)
        pep_aa = torch.cat([aa for coords, aa in frag_list], dim=0)
        pep_length = pep_aa.shape[0]
        assert pep_coord.shape == (pep_length, 3, 3)
        
        with torch.no_grad():
            data_query = {
                'rec_coord': data['rec_coord'],
                'rec_aa': data['rec_aa'],
                'pep_coord': pep_coord,
                'pep_aa': pep_aa,
                'qry_coord': pep_coord,
                'qry_aa': pep_aa,
            }
            # print({k: v.shape for k, v in data_query.items()})
            angle_new, log_kai = self.prediction_model(recursive_to(data_to_batch(data_query, collate_type='default'), self.device))  # (B, L2, 4)
            angle_new = angle_new[0]  # (L2, 2， 2)
            log_kai = log_kai[0]  # (L2, 2)
        
        it = sum([coords.shape[0] for coords, aa in frag_list])
        while True:
            frag_index = it // 2 % len(frag_list)
            dir_index = it % 2
            if dist_list[frag_index + dir_index] > 0:
                break
            it += 1
        assert dist_list[frag_index + dir_index] > 0

        pos_index = sum([coords.shape[0] for coords, aa in frag_list[:frag_index]]) + (0 if dir_index == 0 else frag_list[frag_index][0].shape[0] - 1)
        # print(frag_index, dir_index, pos_index)
        generate_coord_fn = generate_prev_coord if dir_index == 0 else generate_next_coord

        selected_angle = angle_new[pos_index, dir_index]  # (2, 2)
        mu_psi, mu_phi = selected_angle[0], selected_angle[1]
        log_kai_psi, log_kai_phi = log_kai[pos_index, dir_index]  # (2, )

        if extend_strategy == 'sto':
            # print(log_kai.shape)
            psi = VonMises(mu_psi, log_kai_psi.exp()).sample()
            phi = VonMises(mu_phi, log_kai_phi.exp()).sample()
            if verbose:
                print('psi', mu_psi.item(), log_kai_psi.exp().item(), psi.item(), 'phi', mu_phi.item(), log_kai_phi.exp().item(), phi.item())
        elif extend_strategy == 'det':
            psi = mu_psi
            phi = mu_phi
        else:
            raise NotImplementedError

        sample_angle_eu = torch.stack([psi.cos(), psi.sin(), phi.cos(), phi.sin()]).reshape(2, 2)
        coord_new = generate_coord_fn(pep_coord[pos_index], sample_angle_eu.reshape(-1))  # (3, 3), (4,)

        with torch.no_grad():
            data_query = {
                'rec_coord': data['rec_coord'],
                'rec_aa': data['rec_aa'],
                'pep_coord': pep_coord,
                'pep_aa': pep_aa,
                'qry_coord': torch.cat([coord_new.unsqueeze(0)]),
                'qry_aa': torch.tensor([0]),
            }
            logits = self.density_model(recursive_to(data_to_batch(data_query, collate_type='default'), self.device))  # (B, L2, 4)
            logits = logits[0, 0]
            aa_new = logits[:20].argmax()
        
        if dir_index == 0:
            frag_list[frag_index][0] = torch.cat([coord_new.unsqueeze(0), frag_list[frag_index][0]], dim=0)
            frag_list[frag_index][1] = torch.cat([aa_new.unsqueeze(0), frag_list[frag_index][1]], dim=0)
        else:
            frag_list[frag_index][0] = torch.cat([frag_list[frag_index][0], coord_new.unsqueeze(0)], dim=0)
            frag_list[frag_index][1] = torch.cat([frag_list[frag_index][1], aa_new.unsqueeze(0)], dim=0)

        dist_list[frag_index + dir_index] -= 1

        # print(dist_list)
        # print(frag_list)

        return frag_list, dist_list


    def _get_data_query(self, data, pep_coord, pep_cls, type):
        L = data['pep_coord'].shape[0]
        pep_aa = pep_cls.argmax(-1)
        return {
            'rec_coord': data['rec_coord'],
            'rec_aa': data['rec_aa'],
            'pep_coord': pep_coord.detach(),
            'pep_aa': pep_aa.detach(),  # (L, )
            'qry_coord': pep_coord,  # (L, 3, 3,)
            'qry_aa': pep_aa if type == 'angle' else torch.zeros(L, ).long(),  # (L, )
        }


    def _finetune(self, data, pep_coord, pep_aa, n_steps=100, lr=1e-1, finetune_strategy='full', verbose=False):
        assert finetune_strategy in ['full', 'ablation']
        pep_length = pep_aa.shape[0]
        # pep_length = pep_coord.shape[0]
        # dihedral_from_four_points(N1, CA1, C1, N2)
        psi_init = dihedral_from_four_points(pep_coord[:-1, 2], pep_coord[:-1, 0], pep_coord[:-1, 1], pep_coord[1:, 2])
        # dihedral_from_four_points(C1, N2, CA2, C2)
        phi_init = dihedral_from_four_points(pep_coord[:-1, 1], pep_coord[1:, 2], pep_coord[1:, 0], pep_coord[1:, 1])

        x_init = pep_coord[:, 0]
        o_init = rotation_matrix_to_quaternion(construct_3d_basis(pep_coord[:, 0], pep_coord[:, 1], pep_coord[:, 2]))

        # Define parameters
        par_x = torch.tensor(x_init, requires_grad=True)  # L, 3
        par_o = torch.tensor(o_init, requires_grad=True)  # L, 4
        par_psi = torch.tensor(psi_init, requires_grad=True)  # L - 1
        par_phi = torch.tensor(phi_init, requires_grad=True)  # L - 1
        # par_cls = torch.tensor(F.one_hot(pep_aa, 20).float(), requires_grad=True)
        par_cls = torch.tensor(torch.zeros(pep_aa.shape[0], 20).float().to(self.device), requires_grad=True)
        par_T = torch.tensor(0.0 * torch.ones(1).to(self.device), requires_grad=True)

        # optimizer = SGD([par_x, par_o, par_dihed, par_cls], lr=1e-2)
        optimizer = Adam([par_x, par_o, par_psi, par_phi, par_cls, par_T], lr=lr)
        # optimizer = Adam([par_x, par_o, par_dihed], lr=1e-2)

        for t in range(n_steps):
            # Calculate structural difference
            this_coord = coord_from(par_x, par_o)

            # structure next
            data_angle = self._get_data_query(data, this_coord, par_cls, type='angle')
            pred_angles, log_kai = self.prediction_model(recursive_to(data_to_batch(data_angle, collate_type='default'), self.device), 'prefix0')  # (B, L2, 4)
            pred_angles = pred_angles[0, :]  # (pep_length, 2, 2, 2)
            log_kai = log_kai.squeeze(0)  # (pep_length, 2, 2)
            assert pred_angles.shape == (pep_length, 2, 2), f'{pred_angles.shape} != {(pep_length, 2, 2)}'
            assert log_kai.shape == (pep_length, 2, 2), f'{log_kai.shape} != {(pep_length, 2, 2)}'
            diheral_eu = torch.stack([par_psi.cos(), par_psi.sin(), par_phi.cos(), par_phi.sin()], dim=-1)  # (L - 1, 4)
            next_coord = generate_next_coord(this_coord[:-1], diheral_eu)  # (3, 3), (4,)
            obj_str_next = ((next_coord - this_coord[1:]) ** 2).sum(-1).mean()
            obj_ang_next = (
                - VonMises(pred_angles[:-1, 1, 0], log_kai[:-1, 1, 0].exp()).log_prob(par_psi).mean() 
                - VonMises(pred_angles[:-1, 1, 1], log_kai[:-1, 1, 1].exp()).log_prob(par_phi).mean()
            )

            # structure next
            pred_angles, log_kai = self.prediction_model(recursive_to(data_to_batch(data_angle, collate_type='default'), self.device), 'suffix0')  # (B, L2, 4)
            pred_angles = pred_angles[0, :]  # (pep_length, 2, 2, 2)
            log_kai = log_kai.squeeze(0)  # (pep_length, 2, 2)
            assert pred_angles.shape == (pep_length, 2, 2), f'{pred_angles.shape} != {(pep_length, 2, 2)}'
            assert log_kai.shape == (pep_length, 2, 2), f'{log_kai.shape} != {(pep_length, 2, 2)}'
            diheral_eu = torch.stack([par_psi.cos(), par_psi.sin(), par_phi.cos(), par_phi.sin()], dim=-1)  # (L - 1, 4)
            prev_coord = generate_prev_coord(this_coord[1:], diheral_eu)  # (3, 3), (4,)
            obj_str_prev = ((prev_coord - this_coord[:-1]) ** 2).sum(-1).mean() 
            obj_ang_prev = (
                - VonMises(pred_angles[1:, 0, 0], log_kai[1:, 0, 0].exp()).log_prob(par_psi).mean()
                - VonMises(pred_angles[1:, 0, 1], log_kai[1:, 0, 1].exp()).log_prob(par_phi).mean()
            )

            # class next
            data_class = self._get_data_query(data, this_coord, par_cls, type='class')
            logits = self.density_model(recursive_to(data_to_batch(data_class, collate_type='default'), self.device), 'prefix0')  # (B, L2, 4)
            logits = logits.squeeze(0)
            assert logits.shape == (pep_length, 21)
            prob = F.softmax(logits, -1)
            if finetune_strategy == "ablation":
                prob = prob.detach()
            obj_cls_next = -torch.log((prob[:, :20] * F.softmax(par_cls / par_T.exp())).sum(-1)).mean()

            # class prev
            logits = self.density_model(recursive_to(data_to_batch(data_class, collate_type='default'), self.device), 'suffix0')  # (B, L2, 4)
            logits = logits.squeeze(0)
            assert logits.shape == (pep_length, 21)
            prob = F.softmax(logits, -1)
            if finetune_strategy == "ablation":
                prob = prob.detach()
            obj_cls_prev = -torch.log((prob[:, :20] * F.softmax(par_cls / par_T.exp())).sum(-1)).mean()

            obj = obj_str_next + obj_str_prev + obj_cls_next + obj_cls_prev + obj_ang_next + obj_ang_prev
            
            if t % (n_steps // 10) == 0 and verbose:
                print(f'{t}/{n_steps}', obj_str_next.item(), obj_str_prev.item(), obj_ang_next.item(), obj_ang_prev.item(), obj_cls_next.item(), obj_cls_prev.item(), par_T.exp().item())

                # if t % 10 == 0:
                #     for j in range(10):
                #         print(j)
                #         for k in range(21):
                #             print(f'{prob1[j, k].item():.2f}|', end="")
                #             if k % 5 == 0:
                #                 print('|', end="")
                #         print()
                #         for k in range(21):
                #             print(f'{prob2[j, k].item():.2f}|', end="")
                #             if k % 5 == 0:
                #                 print('|', end="")
                #         print()
            
                # print(par_T, F.softmax(par_cls / par_T.exp()))
                # print(par_cls._grad)
                # print(par_cls)
                # print(par_cls)
                # print(par_coord)
                # print(((next_coord - par_coord[1:]) ** 2).sum(-1))
                # print(((prev_coord - par_coord[:L - 1]) ** 2).sum(-1))

            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
            
            par_o.data.div_(torch.sqrt((par_o ** 2).sum(dim=-1).unsqueeze(-1)))

        pep_coord = coord_from(par_x, par_o).detach()
        pep_aa = par_cls.argmax(-1).detach()
        # pep_aa = logits[:, :20].argmax(-1).detach()
        # print(logits)
        # print(par_T.exp(), F.softmax(par_cls / par_T.exp())[:4])
        return pep_coord, pep_aa

    def _metrics(self, gen, gt):
        gen_pos = gen['pep_coord'][:, 0]
        gen_aa = gen['pep_aa'][:, ]
        gt_pos = gt['pep_coord'][:, 0]
        gt_aa = gt['pep_aa'][:, ]

        # seq recovery
        recovery = (gen_aa == gt_aa).float().mean().item()

        # rmsd
        sup = SVDSuperimposer()
        sup.set(gt_pos.view(-1, 3).clone().detach().numpy(), gen_pos.view(-1, 3).clone().detach().numpy())
        sup.run()
        rmsd = float(sup.get_rms())

        valid_flag = ((gen_pos[:-1] - gen_pos[1:]).norm(p=2, dim=-1) < 4.5).all().float().item()

        return {'recovery': recovery, 'rmsd': rmsd, 'valid': valid_flag}

    def _frag_to_seq(self, frag_list):
        pep_coord = torch.cat([coords for coords, aa in frag_list], dim=0)
        pep_aa = torch.cat([aa for coords, aa in frag_list], dim=0)
        return pep_coord, pep_aa

    def _rand_anchors(self, coord, aa, x_sig=1.0, o_sig=0.2):
        x_init = coord[0]
        o_init = rotation_matrix_to_quaternion(construct_3d_basis(coord[0], coord[1], coord[2]))
        x_init = x_init + x_sig * torch.randn_like(x_init)
        o_init = o_init + o_sig * torch.randn_like(o_init)
        o_init = F.normalize(o_init, p=2, dim=-1)
        coord = coord_from(x_init, o_init)
        aa = torch.randint(0, 20, (1,)).long()[0].to(aa.device)
        return coord, aa

    def _get_anchors(self, data, dist_list, anchor_steps=100, anchor_strategy='rand', verbose=False):
        anchor_index = [sum(dist_list[:i + 1]) + i for i in range(len(dist_list) - 1)]
        # print(anchor_index)
        # assert anchor_index[0] == dist_list[0]
        # assert len(anchor_index) < 2 or anchor_index[1] == dist_list[0] + 1 + dist_list[1]
        # print(anchor_index)
        # print(data['pep_coord'])
        # frag_list = [
        #     [data['pep_coord'][anchor].unsqueeze(0), data['pep_aa'][anchor].unsqueeze(0)] for anchor in anchor_index
        # ]
        print()
        frag_list = []
        for anchor in anchor_index:
            coord, aa = data['pep_coord'][anchor], data['pep_aa'][anchor]
            if anchor_strategy == 'gt':
                pass
            elif anchor_strategy == 'ebm':
                coord, aa = self._rand_anchors(coord, aa)
                if anchor_steps > 0:
                    coord, aa = self._sample_anchor(data, coord, aa, n_steps=anchor_steps, verbose=verbose)
            elif anchor_strategy == 'rand':
                coord, aa = self._rand_anchors(coord, aa, x_sig=2.0, o_sig=0.5)
            else:
                raise NotImplementedError
            frag_list.append([coord.unsqueeze(0), aa.unsqueeze(0)])
        
        # print(anchor_index)
        # print(frag_list)
        return frag_list


    def _get_anchor_index(self, L, anchor_nums):
        base_dist = L // (anchor_nums + 1)
        anchor_index = [base_dist * (i + 1) for i in range(anchor_nums)]
        return anchor_index
        
    def _get_dist_list(self, L, dist_strategy):
        if dist_strategy == 'multi':
            if L <= 8:
                L1 = L // 2
                L2 = L - L1 - 1
                return [L1, L2]
            elif L <= 16:
                L1, L3 = L // 4, L // 4
                L2 = L - L1 - L3 - 2
                return [L1, L2, L3]
            elif L <= 24:
                L1, L4 = L // 6, L // 6
                L2 = (L - L1 - L4 - 2) // 2
                L3 = L - L1 - L2 - L4 - 3
                return [L1, L2, L3, L4]
            elif L < 32:
                L1, L5 = L // 8, L // 8
                L2, L4 = (L - L1 - L5 - 2) // 3, (L - L1 - L5 - 2) // 3
                L3 = L - L1 - L2 - L4 - L5 - 4
                return [L1, L2, L3, L4, L5]
            else:
                raise NotImplementedError
        elif dist_strategy == 'single':
            L1 = L // 2
            L2 = L - L1 - 1
            return [L1, L2]
        raise NotImplementedError
    
    def reverse_engineer_dist_list(self, anchor_index, total_length):
        dist_list = []
        previous_index = 0
        
        for i in range(len(anchor_index)):
            current_index = anchor_index[i]
            dist_list.append(current_index - previous_index)
            previous_index = current_index + 1
        
        # 计算最后一个段的长度
        dist_list.append(total_length - previous_index)
        
        return dist_list

    def sample(self, data, anchor_steps=100, finetune_steps=100, 
               dist_strategy='multi', anchor_strategy='ebm', extend_strategy='sto', 
               finetune_strategy='full', verbose=False,anchor_nums=1):
        """
            Args:
                data: dict
                anchor_steps: int
                finetune_steps: int
                dist_strategy: str, 'single' or 'multi'
                anchor_strategy: str, 'rand', 'gt', or 'ebm'
                extend_strategy: str, 'sto', 'det'
        """
        assert (data['pep_aa'] != 20).all()
        pep_length = data['pep_aa'].shape[0]
        assert 2 <= pep_length <= 32
        original_data = data
        data = recursive_to(data, self.device)
        
        # dist_list = self._get_dist_list(pep_length, dist_strategy)
        anchor_index = self._get_anchor_index(pep_length, anchor_nums)
        dist_list = self.reverse_engineer_dist_list(anchor_index, pep_length)
        frag_list = self._get_anchors(data, dist_list, anchor_steps=anchor_steps, anchor_strategy=anchor_strategy, verbose=verbose)

        assert len(dist_list) == len(frag_list) + 1
        while(sum(dist_list) > 0):
            frag_list, dist_list = self._get_next(data, frag_list, dist_list, extend_strategy=extend_strategy, verbose=verbose)
        pep_coord, pep_aa = self._frag_to_seq(frag_list)
        if finetune_steps > 0:
            pep_coord, pep_aa = self._finetune(data, pep_coord, pep_aa, n_steps=finetune_steps, finetune_strategy=finetune_strategy, verbose=verbose)

        gen = {'pep_coord': pep_coord.cpu(), 'pep_aa': pep_aa.cpu()}

        return gen, self._metrics(gen, original_data)

