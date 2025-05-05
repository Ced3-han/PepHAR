# from utils import *
# from geometry import *

from Bio.PDB import PDBParser, PDBIO, Select, is_aa

import os
import pandas as pd
import subprocess
import torch
import numpy as np
import shutil
from tqdm import tqdm

from joblib import delayed, Parallel

import logging

input_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pepflowww/Data/Baselines_new/Tests"
output_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3"


RF = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/Tools/RFdiffusion/scripts/run_inference.py"
PROGEN="/datapool/data2/home/ruihan/data/jiahan/ResProjj/Tools/protein_generator/inference.py"


def get_anchor_index(L, anchor_nums):
    base_dist = L // (anchor_nums + 1)
    anchor_index = [base_dist * (i + 1) for i in range(anchor_nums)]
    return anchor_index

def reverse_engineer_dist_list(anchor_index, total_length):
    dist_list = []
    previous_index = 0
    
    for i in range(len(anchor_index)):
        current_index = anchor_index[i]
        dist_list.append(current_index - previous_index)
        previous_index = current_index + 1
    
    # 计算最后一个段的长度
    dist_list.append(total_length - previous_index)
    
    return dist_list

def get_chain_dic(input_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)

    chain_dic = {}

    for model in structure:
        for chain in model:
            chain_dic[chain.id] = len([res for res in chain if is_aa(res) and res.has_id('CA')])

    return chain_dic

def process_one_item_rf(name='1a1m_C',num_samples=8):
    if not os.path.exists(os.path.join(output_dir,name,'rfs')):
        os.makedirs(os.path.join(output_dir,name,'rfs'))
    chain_dic = get_chain_dic(os.path.join(input_dir,name,'pocket_merge_renum.pdb'))

    # rfdiffusion
    contigs = []
    # hotspot settings
    with open(os.path.join(input_dir,name,'seq.fasta'),'r') as f:
        pep_len = len(f.readlines()[1].strip())
    pep_chain = name.split('_')[-1]
    # acnhor nums
    anchor_nums = 3
    # anchor nums
    if pep_len <= 5:
        anchor_index = [pep_len // 2]
        dist_list = reverse_engineer_dist_list(anchor_index, pep_len)  
    else:
        anchor_index = get_anchor_index(pep_len, anchor_nums)
        dist_list = reverse_engineer_dist_list(anchor_index, pep_len)
    pep_str = ""
    for i in range(len(anchor_index)):
        pep_str += f'{dist_list[i]}-{dist_list[i]}/'
        pep_str += f'{pep_chain}{anchor_index[i]+1}-{anchor_index[i]+1}/'
    pep_str += f'{dist_list[-1]}-{dist_list[-1]}/0'
    contigs.append(pep_str)
    
    # contigs.append(f'{pep_len}-{pep_len}')
    
    for chain,chain_len in chain_dic.items():
        if chain != pep_chain:
            contigs.append(f'{chain}1-{chain_len}/0')
    
    contigs = " ".join(contigs)
    # print(contigs)
    # raise ValueError
    
    command = [
    "python", RF,
    f"inference.output_prefix='{os.path.join(output_dir,name,'rfs','sample')}'",
    f"inference.input_pdb='{os.path.join(input_dir,name,'pocket_merge_renum.pdb')}'",
    f"contigmap.contigs=[{contigs}]",
    f"inference.num_designs={num_samples}",
]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # logging.info("Command output: %s", result.stdout)
        return name
    except subprocess.CalledProcessError as e:
        logging.error("Command failed with return code: %d", e.returncode)
        logging.error("Command output: %s", e.output)
        logging.error("Command stderr: %s", e.stderr)
        return None
    # print(command)
    # try:
    #     result = subprocess.run(command, check=True, capture_output=True, text=True)
    #     print(result)
    #     return name
    # except:
    #     return None
    
def process_one_item_pg(name='1a1m_C',num_samples=10):
    if not os.path.exists(os.path.join(output_dir,name,'pgs')):
        os.makedirs(os.path.join(output_dir,name,'pgs'))
    os.makedirs(os.path.join(output_dir,name,'pgs'),exist_ok=True)
    chain_dic = get_chain_dic(os.path.join(input_dir,name,'pocket_merge_renum.pdb'))
    
    # protein_generator settings
    contigs = []
    
    # hotspot settings
    with open(os.path.join(input_dir,name,'seq.fasta'),'r') as f:
        pep_len = len(f.readlines()[1].strip())
    pep_chain = name.split('_')[-1]
    anchor_nums = 2
    if pep_len <= 5:
        anchor_index = [pep_len // 2]
        dist_list = reverse_engineer_dist_list(anchor_index, pep_len)  
    else:
        anchor_index = get_anchor_index(pep_len, anchor_nums)
        dist_list = reverse_engineer_dist_list(anchor_index, pep_len)
    pep_str = ""
    for i in range(len(anchor_index)):
        pep_str += f'{dist_list[i]}-{dist_list[i]}/'
        pep_str += f'{pep_chain}{anchor_index[i]+1}-{anchor_index[i]+1}/'
    pep_str += f'{dist_list[-1]}-{dist_list[-1]}/0'
    contigs.append(pep_str)
    
    for chain,chain_len in chain_dic.items():
        if chain != pep_chain:
            contigs.append(f'{chain}1-{chain_len}/0')
    
    command = [
        "python", PROGEN,
        "--num_designs", f"{num_samples}",
        "--out", os.path.join(output_dir,name,'pgs','sample'),
        "--pdb", os.path.join(input_dir,name,'pocket_merge_renum.pdb'),
        "--T", "25", # default setting
        "--save_best_plddt", # default setting
        "--contigs", *contigs,
    ]
    # print(command)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # logging.info("Command output: %s", result.stdout)
        return name
    except subprocess.CalledProcessError as e:
        logging.error("Command failed with return code: %d", e.returncode)
        logging.error("Command output: %s", e.output)
        logging.error("Command stderr: %s", e.stderr)
        return None

def process_one_item(name='1a1m_C',num_samples=10):
    process_one_item_pg(name,num_samples)
    process_one_item_rf(name,num_samples)
    

if __name__ == "__main__":
    
    logging.basicConfig(filename='/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/logs/process_rf_3.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    names = os.listdir("/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_ebm_sto_1")
    print(len(names))
    for name in tqdm(names):
        logging.info(f"Processing {name}")
        try:
            process_one_item_rf(name,8)
        except:
            continue
    # process_one_item_rf(names[0],8)

    