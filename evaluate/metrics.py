from utils.protein import parsers
from utils.protein import constants
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm_notebook as tqdm
from Bio.SVDSuperimposer import SVDSuperimposer
import yaml

def retrive_pep(data):
    counts = {}
    for chain in data['chain_nb']:
        #print(chain)
        if chain.item() not in counts:
            counts[chain.item()] = 1
        else:
            counts[chain.item()] += 1
    pep_length = counts[chain.item()]
    pep_chain = chain.item()
    for chain in counts:
        if counts[chain]<pep_length:
            pep_length = counts[chain]
            pep_chain = chain
    index = (data.chain_nb==pep_chain)
    aa = data['aa'][index]
    pos_heavyatom = data['pos_heavyatom'][index]
    return {'aa':aa,'pos_heavyatom':pos_heavyatom,'length':pep_length}

def calc_metric(gt, pred, clip=False):
    # init
    gt_pos = gt['pos_heavyatom'].clone().detach()
    gt_aa = gt['aa'].clone().detach()
    pred_pos = pred['pos_heavyatom'].clone().detach()
    pred_aa = pred['aa'].clone().detach()

    if clip == False:
        if gt['length'] >= pred['length']:
            delta = gt['length'] - pred['length']
            pred_aa = torch.cat([pred_aa, torch.ones(delta, ) * -1], dim=0)
            pred_pos = torch.concat([pred_pos, torch.zeros(delta, 15, 3)], dim=0)
        else:
            delta = pred['length'] - gt['length']
            gt_aa = torch.cat([gt_aa, torch.ones(delta, ) * -1], dim=0)
            gt_pos = torch.concat([gt_pos, torch.zeros(delta, 15, 3)], dim=0)
    else:
        # clip
        if gt['length'] >= pred['length']:
            gt_aa = gt_aa[:pred['length']]
            gt_pos = gt_pos[:pred['length']]
        else:
            pred_aa = pred_aa[:gt['length']]
            pred_pos = pred_pos[:gt['length']]

    # seq recovery
    recovery = (pred_aa == gt_aa).float().mean().item()

    # rmsd
    sup = SVDSuperimposer()
    sup.set(gt_pos.view(-1, 3).numpy(), pred_pos.view(-1, 3).numpy())
    sup.run()
    rmsd = sup.get_rms()
    return recovery, float(rmsd)