import sys, os
sys.path.append(os.curdir)

import torch
from models.ga import GAEncoder
from torchinfo import summary

def test_ga_init():
    N, L = 16, 500
    node_feat_dim, pair_feat_dim, num_layers = 128, 16, 8
    ga_block_opt = {
        'value_dim': 32, 'query_key_dim': 32, 'num_query_points': 8,
        'num_value_points': 8, 'num_heads': 12, 'bias': False,
    }
    model = GAEncoder(node_feat_dim, pair_feat_dim, num_layers, ga_block_opt)
    R = torch.randn(N, L, 3, 3)
    t = torch.randn(N, L, 3)
    res_feat = torch.randn(N, L, node_feat_dim)
    pair_feat = torch.randn(N, L, L, pair_feat_dim)
    mask = torch.rand(N, L) < 0.9
    is_query = torch.rand(N, L) < 0.1
    # new_res_feat = model(R, t, res_feat, pair_feat, mask, is_query)
    # print(new_res_feat.shape)

    gpu_id = 4
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    summary(model, input_data=(R, t, res_feat, pair_feat, mask, is_query), device=device)

def test_ga_query_mask_leakage():
    # TODO: test query mask leakage
    pass

if __name__ == '__main__':
    # test_protein_peptide_dataset()
    test_ga_init()
