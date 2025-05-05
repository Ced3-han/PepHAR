import sys, os
sys.path.append(f"/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori")

path_to_remove = '/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pepar'
sys.path = [x for x in sys.path if path_to_remove not in x]
print(sys.path)

# os.environ['WANDB_CONSOLE'] = 'off'


from evaluate.sample import AnchorBasedSampler
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from tqdm.autonotebook import tqdm
import torch
import os
import argparse

from datasets.protein_peptide_dataset import ProteinPeptideDataset
from evaluate.tools import to_protein, to_peptide, ToStandard
from utils.misc import load_config
from models import get_model
from evaluate.sample_new import AutoregressiveSampler_new
from evaluate.writer import save_pdb_rec_pep
from datasets.transforms.truncate_protein_transform import TruncateProteinV3
from datasets.transforms.to_prediction_transform import ToAAPredicitonTaskV4

from evaluate.tools import data_to_batch, generate_next_coord, generate_prev_coord
from utils.misc import load_config
import torch
from models import get_model
from Bio.SVDSuperimposer import SVDSuperimposer
import torch.nn.functional as F
from utils.train import recursive_to

from evaluate.plotting import plot_protein
import pandas as pd
from pathlib import Path

# python scripts/sample.py --gpu 0 --anchor_strategy "ebm" --extend_strategy "sto" --anchor_nums 2

project_root = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pep-design-ori"
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=8)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--density_config_path', type=str, default=f'{project_root}/logs/density_v4_x5o2_2024_09_08__11_25_36/density_v4_x5o2.yml')
parser.add_argument('--density_param_path', type=str, default=f'{project_root}/logs/density_v4_x5o2_2024_09_08__11_25_36/checkpoints/1400.pt')
parser.add_argument('--prediction_config_path', type=str, default=f'{project_root}/logs/prediction_d2_x2o1_2024_09_08__11_21_33/prediction_d2_x2o1.yml')
parser.add_argument('--prediction_param_path', type=str, default=f'{project_root}/logs/prediction_d2_x2o1_2024_09_08__11_21_33/checkpoints/2400.pt')

parser.add_argument('--anchor_steps', type=int, default=10, help='锚点步骤数')
parser.add_argument('--finetune_steps', type=int, default=0, help='微调步骤数')
parser.add_argument('--dist_strategy', type=str, choices=['single', 'multi'], default='single', help='距离策略: single 或 multi')
parser.add_argument('--anchor_strategy', type=str, choices=['rand', 'gt', 'ebm'], default='gt', help='锚点策略: rand, gt 或 ebm')
parser.add_argument('--extend_strategy', type=str, choices=['sto', 'det'], default='sto', help='扩展策略: sto 或 det')
parser.add_argument('--anchor_nums', type=int, default=1, help='锚点数量')

args = parser.parse_args()
# args.device = 'cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu'
args.device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
args.max_pep_length_density = 32

data_root = "/datapool/data2/home/ruihan/data/jiahan/ResProjj/PepDiff/pepar/data/test_data.pkl"
transform = transforms.Compose([TruncateProteinV3()])
dataset = ProteinPeptideDataset(data_root,train=False,transform=transform, return_name=True)
dataset_no = ProteinPeptideDataset(data_root,train=False,transform=None, return_name=True)


pdb_path = f'result/new_{args.anchor_steps}_{args.finetune_steps}_{args.dist_strategy}_{args.anchor_strategy}_{args.extend_strategy}_{args.anchor_nums}'
csv_name = f'{project_root}/{pdb_path}/test.csv'
Path(f'{project_root}/{pdb_path}').mkdir(parents=True, exist_ok=True)

table = {'name':[],'rmsd':[],'recovery':[],'rec_length':[],'pep_length':[],'num':[],'valid':[]}

for i in tqdm(range(len(dataset))):
    for j in range(args.n_samples):

        # try: 
            name, data = dataset[i]
            name, data_full = dataset_no[i]
            # if name != '5ih2_M':
            #     continue

            ## Save GT
            if j == 0:
                save_pdb_rec_pep(data_full, data_full, f'{project_root}/{pdb_path}/{name}/gt.pdb')

            ## Sample new 
            sampler = AnchorBasedSampler(args)

            # Multiple Anchors, default 0,100
            peptide, metrics = sampler.sample(data, anchor_steps=args.anchor_steps, finetune_steps=args.finetune_steps, 
                                              dist_strategy=args.dist_strategy, anchor_strategy=args.anchor_strategy, 
                                              extend_strategy=args.extend_strategy, verbose=False, anchor_nums=args.anchor_nums)

            # Save Data
            table['name'].append(name)
            table['num'].append(j)
            table['rmsd'].append(metrics['rmsd'])
            table['recovery'].append(metrics['recovery'])
            table['rec_length'].append((data['rec_aa'] != 20).sum().item())
            table['pep_length'].append((data['pep_aa'] != 20).sum().item())
            table['valid'].append(metrics['valid'])
            # table['ppl'].append(0.0)
            # table['div'].append(0.0)
            pd.DataFrame(table).to_csv(csv_name, index=None)

            ## Save New
            save_pdb_rec_pep(data_full, peptide, f'{project_root}/{pdb_path}/{name}/gen_{j}.pdb')
