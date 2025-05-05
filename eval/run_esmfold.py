import os
import pandas as pd
import subprocess
import torch
import esm
import numpy as np
import shutil
from tqdm import tqdm

from joblib import delayed, Parallel

import warnings
from Bio import BiopythonWarning, SeqIO

from geometry import *

# 忽略PDBConstructionWarning
warnings.filterwarnings('ignore', category=BiopythonWarning)

# input_dir="./Data/Baselines_new/Tests"
# output_dir="/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Baselines_new/Codesign"

model = esm.pretrained.esmfold_v1()
model = model.eval().to('cuda:0')

def process_rf(name='1aze_B'):
    # input_dir=".Data/Baselines_new/Tests"
    # output_dir=".Data/Baselines_new/Codesign"
    struct_dir = os.path.join("/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3",name,'rfs_refold')
    seq_dir = os.path.join("/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3",name,'mpnns','seqs')
    os.makedirs(struct_dir,exist_ok=True)
    seqs = {}
    for seq_path in os.listdir(seq_dir):
        tmp_seqs = []
        if seq_path.endswith('.fasta'):
            for record in SeqIO.parse(os.path.join(seq_dir,seq_path), "fasta"):
                tmp_seqs.append(str(record.seq))
        seqs[seq_path.split('.')[0]] = tmp_seqs[-1]
    for seq_name,seq in seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        with open(os.path.join(struct_dir,seq_name+'.pdb'),'w') as f:
            f.write(output)

def process_pg(name='1aze_B',chain_id='A'):
    input_dir=".Data/Baselines_new/Tests"
    output_dir=".Data/Baselines_new/Codesign"
    struct_dir = os.path.join(output_dir,name,'pgs_refold')
    seq_dir = os.path.join(output_dir,name,'pgs')
    os.makedirs(struct_dir,exist_ok=True)
    seqs = {}
    for seq_path in os.listdir(seq_dir):
        if seq_path.endswith('.pdb'):
            seqs[seq_path.split('.')[0]] = get_seq(os.path.join(seq_dir,seq_path),chain_id)
    for seq_name,seq in seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        with open(os.path.join(struct_dir,seq_name+'.pdb'),'w') as f:
            f.write(output)

def process_ar(name='1aze_B',root_dir="/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/result",sample_dir='rfs'):
    # input_dir=".Data/Baselines_new/Tests"
    # output_dir=".Data/Baselines_new/Codesign"
    struct_dir = os.path.join(root_dir,sample_dir,name,'refold')
    seq_dir = os.path.join(root_dir,sample_dir,name,'seqs','seqs')
    os.makedirs(struct_dir,exist_ok=True)
    seqs = {}
    for seq_path in os.listdir(seq_dir):
        tmp_seqs = []
        if seq_path.endswith('.fasta'):
            for record in SeqIO.parse(os.path.join(seq_dir,seq_path), "fasta"):
                tmp_seqs.append(str(record.seq))
        seqs[seq_path.split('.')[0]] = tmp_seqs[-1]
    for seq_name,seq in seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        with open(os.path.join(struct_dir,seq_name+'.pdb'),'w') as f:
            f.write(output)
            
def refold(name,chain_id):
    raw_dir = os.path.join('/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/pf_3','pdbs')
    refold_dir = os.path.join('/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/pf_3','pdbs_refold')
    os.makedirs(os.path.join(refold_dir,name),exist_ok=True)
    seqs = {}
    for seq_path in os.listdir(os.path.join(raw_dir,name)):
        if seq_path.endswith('.pdb'):
            seqs[seq_path.split('.')[0]] = get_seq(os.path.join(raw_dir,name,seq_path),chain_id)
    for seq_name,seq in seqs.items():
        with torch.no_grad():
            output = model.infer_pdb(seq)
        with open(os.path.join(refold_dir,name,seq_name+'.pdb'),'w') as f:
            f.write(output)
            
def get_seq_from_fasta(fasta_path):
    with open(fasta_path, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            return str(record.seq)

if __name__ == '__main__':
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/pdb/rf_2"
    # for name in tqdm(os.listdir(root_dir)):
    #     try:
    #         if os.path.isdir(os.path.join(root_dir,name)):
    #             refold(name,name.split('_')[-1])
    #     except:
    #         continue
    # for name in tqdm(os.listdir('/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/pf_3/pdbs')):
    #     refold(name,name.split('_')[-1])
    
    # for name in tqdm(os.listdir('/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3')):
    #     try:
    #         process_rf(name)
    #     except:
    #         continue
    
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/result"
    # sample_dirs = [
    #     "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_ebm_sto_1",
    #     "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_ebm_sto_2",
    #     "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_gt_sto_3",
    #     "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_rand_sto_2"
    # ]
    # for sample_dir in tqdm(sample_dirs):
    #     if os.path.isdir(os.path.join(root_dir,sample_dir)):
    #         for name in tqdm(os.listdir(os.path.join(root_dir,sample_dir))):
    #             if os.path.isdir(os.path.join(root_dir,sample_dir,name)):
    #                 try:
    #                     process_ar(name,root_dir,sample_dir)
    #                 except:
    #                     continue
    
    sub_dirs = ["ar3",'ar2','ar1','gt1','gt2','gt3','rf1','rf2','rf3','pf1','pf2','pf3']
    
    names = os.listdir('/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/af2_seqs/gt3_test')
    names = [name[:6] for name in names]
    raw_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/af2_seqs/"

    for sub_dir in sub_dirs:
        out_dir = os.path.join('/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/esm_seqs',sub_dir)
        os.makedirs(out_dir,exist_ok=True)
        seqs = {}
        for seq_path in os.listdir(os.path.join(raw_dir,sub_dir)):
            if seq_path.endswith('.fasta') and seq_path[:6] in names:
                seqs[seq_path.split('.')[0]] = get_seq_from_fasta(os.path.join(raw_dir, sub_dir, seq_path))

        # print(seqs)
        for seq_name,seq in tqdm(seqs.items()):
            with torch.no_grad():
                output = model.infer_pdb(seq)
            with open(os.path.join(out_dir,seq_name+'.pdb'),'w') as f:
                f.write(output)