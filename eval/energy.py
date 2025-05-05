import pyrosetta
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

import os
import pandas as pd
import subprocess
import numpy as np
import shutil
from tqdm import tqdm
import pickle

from joblib import delayed, Parallel
# from utils import *

from Bio.PDB import PDBParser, PDBIO, Select, is_aa

import logging

# input_dir=".Tests"
# output_dir="./Pack"

def get_chain_dic(input_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)
    chain_dic = {}
    for model in structure:
        for chain in model:
            chain_dic[chain.id] = len([res for res in chain if is_aa(res) and res.has_id('CA')])

    return chain_dic

def get_rosetta_score_base(pdb_path,chain_id='Z',interface='A_B'):
    try:
        init()
        pose = pyrosetta.pose_from_pdb(pdb_path)
        # chains = list(get_chain_dic(pdb_path).keys())
        # chains.remove(chain_id)
        # interface = f'{chain_id}_{"".join(chains)}'
        # interface='AB_Z'
        fast_relax = FastRelax() # cant be pickled
        scorefxn = get_fa_scorefxn()
        fast_relax.set_scorefxn(scorefxn)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        stabs,binds = [],[]
        for i in range(2):
            fast_relax.apply(pose)
            stab = scorefxn(pose)
            mover.apply(pose)
            bind = pose.scores['dG_separated']
            stabs.append(stab)
            binds.append(bind)
        return {'name':pdb_path,'stab':np.array(stabs).mean(),'bind':np.array(binds).mean()}
    except:
        return {'name':pdb_path,'stab':999.0,'bind':999.0}


def get_rosetta_score(pdb_path,chain='A'):
    try:
        init()
        pose = pyrosetta.pose_from_pdb(pdb_path)
        chains = list(get_chain_dic(os.path.join(input_dir,name,'pocket_merge_renum.pdb')).keys())
        chains.remove(chain)
        interface = f'{chain}_{"".join(chains)}'
        # interface='A_B'
        fast_relax = FastRelax() # cant be pickled
        scorefxn = get_fa_scorefxn()
        fast_relax.set_scorefxn(scorefxn)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        fast_relax.apply(pose)
        energy = scorefxn(pose)
        mover.apply(pose)
        dg = pose.scores['dG_separated']
        return [pdb_path,energy,dg]
    except:
        return [pdb_path,999.0,999.0]

def pack_sc(name='1a1m_C',num_samples=10):
    try:
        if os.path.exists(os.path.join(output_dir,name,'rosetta')):
            shutil.rmtree(os.path.join(output_dir,name,'rosetta'))
        os.makedirs(os.path.join(output_dir,name,'rosetta'),exist_ok=True)
        init()
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())  # Only repack, don't change amino acid types
        packer = PackRotamersMover()
        packer.task_factory(tf)
        for i in range(num_samples):
            pose = pose_from_pdb(os.path.join(input_dir,name,f'pocket_merge_renum_bb.pdb'))
            packer.apply(pose)
            pose.dump_pdb(os.path.join(output_dir,name,'rosetta',f'packed_{i}.pdb'))
    except:
        return None
    

if __name__ == '__main__':
    # input_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/5ih2_rep"
    # output_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/5ih2_M_draw"
    # init()
    # tf = TaskFactory()
    # tf.push_back(RestrictToRepacking())  # Only repack, don't change amino acid types
    # packer = PackRotamersMover()
    # packer.task_factory(tf)
    # for i in range(8):
    #     pose = pose_from_pdb(os.path.join(input_dir,f'gen_{i}.pdb'))
    #     packer.apply(pose)
    #     pose.dump_pdb(os.path.join(output_dir,f'gen_{i}.pdb'))
    
    
    
    
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/5ih2_M"
    # pack_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp/5ih2_draw"
    # init()
    # tf = TaskFactory()
    # tf.push_back(RestrictToRepacking())  # Only repack, don't change amino acid types
    # packer = PackRotamersMover()
    # packer.task_factory(tf)
    # for pdb in os.listdir(root_dir):
    #     pose = pose_from_pdb(os.path.join(root_dir,pdb))
    #     packer.apply(pose)
    #     pose.dump_pdb(os.path.join(pack_dir,pdb))
    
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/new_10_0_single_gt_sto_3/3avn_G/"
    # output_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/tmp"
    # samples = ["gt.pdb","gen_1.pdb","gen_2.pdb","gen_0.pdb"]
    # init()
    # tf = TaskFactory()
    # tf.push_back(RestrictToRepacking())  # Only repack, don't change amino acid types
    # packer = PackRotamersMover()
    # packer.task_factory(tf)
    # for pdb in samples:
    #     pose = pose_from_pdb(os.path.join(root_dir,pdb))
    #     packer.apply(pose)
    #     pose.dump_pdb(os.path.join(output_dir,pdb))
        
    # print(get_chain_dic("/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/pdb/10_10_single_ebm_sto_1/1bm2_L/gen_0.pdb"))
    # print(get_rosetta_score_base("/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/pdb/10_10_single_ebm_sto_1/1bm2_L/gen_0.pdb","Z"))
    
    # # 配置日志记录, pf
    # logging.basicConfig(filename='/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/logs/rosetta_pf_1.log', level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/pf_1/pdbs"
    # names = os.listdir(root_dir)
    # results = []
    # for name in tqdm(names):
    #     for file in os.listdir(os.path.join(root_dir, name)):
    #         if file.endswith(".pdb"):
    #             score = get_rosetta_score_base(os.path.join(root_dir, name, file), chain_id=name.split('_')[-1])
    #             results.append(score)
    #             logging.info(f'Processed file: {file} in directory: {name}, score: {score["stab"]}/{score["bind"]}')
    
    # with open('/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/amisc/pf_1.pkl', 'wb') as f:
    #     pickle.dump(results, f)
        
        
    # # 配置日志记录, rf
    # logging.basicConfig(filename='/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/logs/rosetta_rf_3.log', level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')

    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/pdb/rf_3"
    # names = os.listdir(root_dir)
    # results = []
    # for name in tqdm(names):
    #     for file in os.listdir(os.path.join(root_dir, name, 'pdb_full')):
    #         if file.endswith(".pdb"):
    #             score = get_rosetta_score_base(os.path.join(root_dir, name, 'pdb_full', file), chain_id='A', interface='A_B')
    #             results.append(score)
    #             logging.info(f'Processed file: {file} in directory: {name}, score: {score["stab"]}/{score["bind"]}')
    
    # with open('/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/amisc/rf_3.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    # with open("/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/logs/energys/now.txt",'r') as f:
    #     processed = f.readlines()
    #     processed = [x.strip() for x in processed]
    
    # processed = []
    
    # # 配置日志记录, rf
    # dir = "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/result"
    # sample = "10_0_single_ebm_sto_2"
    # log_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/logs/energys"
    # logging.basicConfig(filename=os.path.join(log_dir,f"gt.log"), level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # root_dir = os.path.join(dir,sample)
    # names = os.listdir(root_dir)
    
    # log_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/logs/rosetta_rf.log"
    # logging.basicConfig(filename=log_dir, level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pep-design-ori/pdb/rf_2"
    # names = os.listdir(root_dir)


    # results = []
    # for name in tqdm(names):
    #     if name not in processed and os.path.isdir(os.path.join(root_dir, name, 'pdb_full')):
    #         logging.info(f'Processing directory: {name}')
    #         chains = list(get_chain_dic(os.path.join(root_dir, name, 'pdb_full','gen_0.pdb')).keys())
    #         chains.remove('Z')
    #         interface = f'{"Z"}_{"".join(chains)}'
    #         # for file in os.listdir(os.path.join(root_dir, name, 'pdb_full')):
    #         #     if file.endswith(".pdb"):
    #         file = 'gt.pdb'
    #         score = get_rosetta_score_base(os.path.join(root_dir, name, 'bb', file), interface=interface)
    #         results.append(score)
    #         logging.info(f'Processed file: {file} in directory: {name}, score: {score["stab"]}/{score["bind"]}')
        # # gt
        # score = get_rosetta_score_base(os.path.join(root_dir, name, 'bb', 'gt.pdb'), chain_id=name.split('_')[-1])
        # results.append(score)
        # logging.info(f'Processed file: {gt.pdb} in directory: {name}, score: {score["stab"]}/{score["bind"]}')
    
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/10_100_single_ebm_det_2"
    # root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/result/new_10_0_single_gt_sto_2"
    # log_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/logs"
    # logging.basicConfig(filename=os.path.join(log_dir,f"draw.log"), level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # names = os.listdir(root_dir)
    
    # print('5ih2_M' in names)
    # results = []
    # for name in tqdm(names):

    #     if name!='5ih2_M':
    #         continue
    log_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/logs"
    logging.basicConfig(filename=os.path.join(log_dir,f"rf_pglad.log"), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    root_dir = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/PepGLAD/test/codesign/pdb"
    results = []
    names = os.listdir(root_dir)
    for name in names:    
        for pdb in os.listdir(os.path.join(root_dir, name)):
            if pdb.endswith(".pdb"):
                chains = list(get_chain_dic(os.path.join(root_dir, name, pdb)).keys())
                if len(chains) != 2:
                    continue
                interface = f'{chains[0]}_{chains[0]}'
                score = get_rosetta_score_base(os.path.join(root_dir, name, pdb), interface=interface)
                results.append(score)
                logging.info(f'Processed file: {pdb} in directory: {name}, score: {score["stab"]}/{score["bind"]}')
    with open(f'/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori/amisc/rf_pglad.pkl', 'wb') as f:
        pickle.dump(results, f)