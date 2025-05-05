from utils import *
from geometry import *

import os
import pandas as pd
import subprocess
import torch
import numpy as np
import shutil
from tqdm import tqdm

from joblib import delayed, Parallel

from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model, Chain


HELPERS = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/Tools/ProteinMPNN/helper_scripts"
RUNNER = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/Tools/ProteinMPNN/protein_mpnn_run.py"

def get_chain_nums(pdb_path,chain_id):
    parser = PDBParser()
    chain = parser.get_structure('X',pdb_path)[0][chain_id]
    residue_nums = [residue.get_id()[1] for residue in chain]
    return residue_nums

def process_mpnn_bb(input_dir,name='1aze_B',chains_to_design="Z",num_samples=1):
    # input_dir = './Data/Models_new/Codesign/bb/pdbs'
    # output_dir = './Data/Models_new/Codesign/bb/seqs'
    try:
        if not os.path.exists(os.path.join(input_dir,name,'seqs')):
            os.makedirs(os.path.join(input_dir,name,'seqs'))
        dirname = os.path.join(input_dir,name,'seqs')
        # defined dirs
        path_for_parsed_chains=os.path.join(dirname,'parsed_pdbs.jsonl')
        path_for_assigned_chains=os.path.join(dirname,'assigned_pdbs.jsonl')
        path_for_fixed_positions=os.path.join(dirname,'fixed_pdbs.jsonl')
        residue_nums = get_chain_nums(os.path.join(input_dir,name,'bb','gt.pdb'),chains_to_design)
        design_only_positions = " ".join(map(str,residue_nums)) #design only these residues; use flag --specify_non_fixed
        # print(path_for_assigned_chains)
        # print(design_only_positions)
        subprocess.run([
            "python", os.path.join(HELPERS,"parse_multiple_chains.py"),
            "--input_path", os.path.join(input_dir,name,'bb'),
            "--output_path", path_for_parsed_chains,
        ])
        subprocess.run([
            "python", os.path.join(HELPERS,"assign_fixed_chains.py"),
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_assigned_chains,
            '--chain_list', chains_to_design,
        ])
        subprocess.run([
            "python", os.path.join(HELPERS,"make_fixed_positions_dict.py"),
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_fixed_positions,
            '--chain_list', chains_to_design,
            '--position_list', design_only_positions,
            '--specify_non_fixed'
        ])
        # run mpnn
        # print('run mpnns')
        subprocess.run([
            "python", RUNNER,
            "--jsonl_path", path_for_parsed_chains,
            "--chain_id_jsonl", path_for_assigned_chains,
            "--fixed_positions_jsonl", path_for_fixed_positions,
            "--out_folder", dirname,
            "--num_seq_per_target", f"{num_samples}",
            "--sampling_temp", "0.1",
            "--seed", "37",
            "--batch_size","1",
            '--device','cuda:1'
        ])
    except:
        return

def process_one_item_mpnn(name='1a1m_C',chains_to_design="A",num_samples=1):
    # input_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3"
    # output_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3"
    input_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/3avf/input"
    output_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/3avf/output"
    if not os.path.exists(os.path.join(output_dir,name,'mpnns')):
        os.makedirs(os.path.join(output_dir,name,'mpnns'))
    # if not os.path.exists(os.path.join(output_dir,name,'pocket_merge_renum.pdb')):
    #     chain_dic = renumber_pdb(os.path.join(input_dir,name,'pocket_merge.pdb'),os.path.join(output_dir,name,'pocket_merge_renum.pdb'))
    dirname = os.path.join(output_dir,name,'mpnns')
    # defined dirs
    path_for_parsed_chains=os.path.join(dirname,'parsed_pdbs.jsonl')
    path_for_assigned_chains=os.path.join(dirname,'assigned_pdbs.jsonl')
    path_for_fixed_positions=os.path.join(dirname,'fixed_pdbs.jsonl')
    with open(os.path.join("/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pepflowww/Data/Baselines_new/Tests",name,'seq.fasta'),'r') as f:
        pep_len = len(f.readlines()[1].strip())
    design_only_positions=" ".join(map(str,list(range(1,pep_len+1)))) #design only these residues; use flag --specify_non_fixed
    # print(design_only_positions)
    # parsed chains
    # print("parsing chains")
    subprocess.run([
        "python", os.path.join(HELPERS,"parse_multiple_chains.py"),
        "--input_path", os.path.join(input_dir,name,'rfs'),#os.path.join('/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Baselines/Fixbb/',name),
        # "--input_path", os.path.join(input_dir,name,'rfs'),
        "--output_path", path_for_parsed_chains,
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"assign_fixed_chains.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_assigned_chains,
        '--chain_list', chains_to_design,
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"make_fixed_positions_dict.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_fixed_positions,
        '--chain_list', chains_to_design,
        '--position_list', design_only_positions,
        '--specify_non_fixed'
    ])
    # run mpnn
    # print('run mpnns')
    subprocess.run([
        "python", RUNNER,
        "--jsonl_path", path_for_parsed_chains,
        "--chain_id_jsonl", path_for_assigned_chains,
        "--fixed_positions_jsonl", path_for_fixed_positions,
        "--out_folder", dirname,
        "--num_seq_per_target", f"{num_samples}",
        "--sampling_temp", "0.1",
        "--seed", "37",
        "--batch_size","1",
        '--device','cuda:1'
    ])


def write_seq_to_pdb(seq_path,pdb_path,out_path,chain_id):
    # first we should fix GGGGG in rfs with mpnn generated seq
    aa_mapping = {"A": "ALA","C": "CYS","D": "ASP","E": "GLU","F": "PHE","G": "GLY","H": "HIS","I": "ILE","K": "LYS","L": "LEU","M": "MET","N": "ASN","P": "PRO","Q": "GLN","R": "ARG","S": "SER","T": "THR","V": "VAL","W": "TRP","Y": "TYR",
                  'X':'UNK'}
    tmps = []
    for record in SeqIO.parse(seq_path, "fasta"):
        tmps.append(str(record.seq))
    seq = tmps[-1]
    
    parser = PDBParser()
    structure = parser.get_structure("X", pdb_path)
    model = structure[0]
    for chain in model:
        if chain.id == chain_id:  # 假设你要更改的是链A
            for i,res in enumerate(chain):
                if i<len(seq):
                    res.resname = aa_mapping[seq[i]]
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)

def extract_chain(pdb_path, chain_id, out_path):
    # 解析 PDB 文件
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_path)
    
    # 提取指定链 ID 的链
    model = structure[0]
    chain = model[chain_id]
    
    # 创建一个新的结构对象，只包含提取的链
    new_structure = Structure.Structure("new_structure")
    new_model = Model.Model(0)  # 创建一个新的模型对象
    new_chain = Chain.Chain(chain_id)  # 创建一个新的链对象
    
    # 复制链中的所有残基到新的链对象
    for residue in chain:
        new_chain.add(residue)
    
    new_model.add(new_chain)
    new_structure.add(new_model)
    
    # 保存新的结构对象为 PDB 文件
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(out_path)
    
if __name__ == '__main__':
    input_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/3avn/input"
    output_dir="/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/3avn/output"
    if not os.path.exists(os.path.join(output_dir,'mpnns')):
        os.makedirs(os.path.join(output_dir,'mpnns'))
    # if not os.path.exists(os.path.join(output_dir,name,'pocket_merge_renum.pdb')):
    #     chain_dic = renumber_pdb(os.path.join(input_dir,name,'pocket_merge.pdb'),os.path.join(output_dir,name,'pocket_merge_renum.pdb'))
    dirname = os.path.join(output_dir,'mpnns')
    # defined dirs
    path_for_parsed_chains=os.path.join(dirname,'parsed_pdbs.jsonl')
    path_for_assigned_chains=os.path.join(dirname,'assigned_pdbs.jsonl')
    path_for_fixed_positions=os.path.join(dirname,'fixed_pdbs.jsonl')
    pep_len = 10
    design_only_positions=" ".join(map(str,list(range(1,pep_len+1)))) #design only these residues; use flag --specify_non_fixed
    # print(design_only_positions)
    # parsed chains
    # print("parsing chains")
    subprocess.run([
        "python", os.path.join(HELPERS,"parse_multiple_chains.py"),
        "--input_path", os.path.join(input_dir),#os.path.join('/datapool/data2/home/jiahan/ResProj/PepDiff/frame-flow/Data/Baselines/Fixbb/',name),
        # "--input_path", os.path.join(input_dir,name,'rfs'),
        "--output_path", path_for_parsed_chains,
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"assign_fixed_chains.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_assigned_chains,
        '--chain_list', 'Z',
    ])
    subprocess.run([
        "python", os.path.join(HELPERS,"make_fixed_positions_dict.py"),
        "--input_path", path_for_parsed_chains,
        "--output_path", path_for_fixed_positions,
        '--chain_list', 'Z',
        '--position_list', design_only_positions,
        '--specify_non_fixed'
    ])
    # run mpnn
    # print('run mpnns')
    subprocess.run([
        "python", RUNNER,
        "--jsonl_path", path_for_parsed_chains,
        "--chain_id_jsonl", path_for_assigned_chains,
        "--fixed_positions_jsonl", path_for_fixed_positions,
        "--out_folder", dirname,
        "--num_seq_per_target", f"{16}",
        "--sampling_temp", "0.1",
        "--seed", "37",
        "--batch_size","1",
        '--device','cuda:1'
    ])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # extract_chain("/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_ebm_det_2/1aze_B/pdb_full/gen_0.pdb",
    #               'Z',
    #               '/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_ebm_det_2/1aze_B/pdb_full/gen.pdb')
    
    
    # dir = '/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/'
    # for sample in os.listdir(dir):
    #     names = os.listdir(os.path.join(dir,sample))
    #     names = [name for name in names if os.path.isdir(os.path.join(dir,sample,name))]
    #     for name in tqdm(names):
    #         os.makedirs(os.path.join(dir,sample,name,'pdb_single'),exist_ok=True)
    #         for i in range(8):
    #             extract_chain(os.path.join(dir,sample,name,'pdb_full',f'gen_{i}.pdb'),
    #                         'Z',
    #                         os.path.join(dir,sample,name,'pdb_single',f'gen_{i}.pdb'))
    
    
    # # process_one_item_mpnn('1a1m_C')
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/result"
    # for sample in os.listdir(root_dir):
    #     if 'new' not in sample and os.path.isdir(os.path.join(root_dir,sample)):
    #         for name in tqdm(os.listdir(os.path.join(root_dir,sample))):
    #             if os.path.isdir(os.path.join(root_dir,sample,name)):
    #                 process_mpnn_bb(os.path.join(root_dir,sample),name)
    
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3" # rf
    # for name in tqdm(os.listdir(root_dir)):
    #     try:
    #         if os.path.isdir(os.path.join(root_dir,name)):
    #             process_one_item_mpnn(name)
    #     except:
    #         continue
    
    
    # root_dir = '/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_0_single_ebm_sto_2'
    # names = os.listdir(root_dir)
    # names = [name for name in names if '.csv' not in names]
    # # print(names)
    # Parallel(n_jobs=8)(delayed(process_mpnn_bb)(root_dir,name) for name in tqdm(names))
    # process_mpnn_bb('/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/result/10_100_single_ebm_det_2')
    # write_seq_to_pdb('./Data/Baselines_new/Codesign/1a1m_C/mpnns/1a1m_C_0.fasta','./Data/Baselines_new/Codesign/1a1m_C/rfs/1a1m_C.pdb','./Data/Baselines_new/Codesign/1a1m_C/mpnns/1a1m_C_0.pdb','A')
    
    # # dir = '/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/result/'
    # sample = '10_0_single_ebm_sto_2'
    # # # for sample in os.listdir(dir):
    # # #     if "new" not in sample:
    # root_dir = f"/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/{sample}"
    # ls = []
    # names = os.listdir(root_dir)
    # for name in tqdm(names):
    #     try:
    #         os.makedirs(os.path.join(root_dir,name,'pdb_full'),exist_ok=True)
    #         for i in range(8):
    #             write_seq_to_pdb(os.path.join(root_dir,name,'seqs','seqs',f'gen_{i}.fasta'),
    #                             os.path.join(root_dir,name,'bb',f'gen_{i}.pdb'),
    #                             os.path.join(root_dir,name,'pdb_full',f'gen_{i}.pdb'),'Z')
    #     except:
    #         ls.append(name)
    #         print(name)
    # with open(os.path.join(root_dir,'failed.txt'),'w') as f:
    #     f.write('\n'.join(ls))
            
# /datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/pdb/rf_2/1aze_B/mpnns/seqs/sample_0.fasta

# '/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/pdb/rf_2/1r1s_B/mpnns/seqs/sample_0.fasta'


    # input_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3"
    # # with open("/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/amisc/rf_1_failed.txt",'w') as f:
    # for name in tqdm(os.listdir(input_dir)):
    #     os.makedirs(os.path.join(input_dir,name,'pdb_full'),exist_ok=True)
    #     # print(name)
    #     for i in range(8):
    #         if os.path.exists(os.path.join(input_dir, name, 'mpnns','seqs', f'sample_{i}.fa')) and not os.path.exists(os.path.join(input_dir,name,'pdb_full',f'sample_{i}.pdb')):
    #             os.rename(os.path.join(input_dir, name, 'mpnns','seqs', f'sample_{i}.fa'),os.path.join(input_dir, name, 'mpnns','seqs', f'sample_{i}.fasta'))
    #             write_seq_to_pdb(os.path.join(input_dir, name, 'mpnns','seqs', f'sample_{i}.fasta'),os.path.join(input_dir,name,'rfs',f'sample_{i}.pdb'),os.path.join(input_dir,name,'pdb_full',f'sample_{i}.pdb'),'A')
    #                 # f.write(name+'\n')
            