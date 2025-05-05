from collections import defaultdict
import datetime
import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
import torch.cuda.amp as amp
import torch.distributed as distrib
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import wandb

from utils.vc import get_version, has_changes
from utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from datasets.padding_collate import PaddingCollateV3
from utils.train import ValidationLossTape, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, \
    sum_weighted_losses, log_losses_wandb
from utils.misc import BlackHole
from utils.protein.constants import resindex_to_ressymb
from datasets import get_dataset
from models import get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--postfix', type=str, default=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    
    # Version control
    # branch, version = get_version()
    # version_short = '%s-%s' % (branch, version[:7])
    # if has_changes() and not args.debug:
    #     c = input('Start training anyway? (y/n) ')
    #     if c != 'y':
    #         exit()

    # Load configs
    config, config_name = load_config(args.configs)
    seed_all(config.train.seed)

    # Init wandb
    wandb.init(project='PepDesign', name=f'{config_name}_{args.postfix}', config=config)
    wandb.config.update({'device': args.device})
    # You can also load different config files with the command line argument --configs.

    # Logging
    # Logging
    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix='%s' % config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    logger = get_logger('train', log_dir)
    writer = BlackHole()
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    # tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.configs))):
        shutil.copyfile(args.configs, os.path.join(log_dir, os.path.basename(args.configs)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    train_dataset = get_dataset(config.data.train)
    val_dataset = get_dataset(config.data.val)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollateV3(), num_workers=args.num_workers)
    # train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=args.num_workers)
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollateV3(), num_workers=args.num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    wandb.watch(model)  # set the model grad to wandb
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])


    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward pass
        loss_dict, info_dict = model.get_loss(batch)
        # print(accs.shape)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        log_losses_wandb(loss_dict, it, 'train', logger, writer, others={
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            **{f'{k}': wandb.Histogram(v.detach().cpu()) if v.dim() > 0 else v.item() for k, v in info_dict.items()},
            # 'time_forward': (time_forward_end - time_start) / 1000,
            # 'time_backward': (time_backward_end - time_forward_end) / 1000,
        })


    def validate(it):
        loss_tape = ValidationLossTape()
        info_tape = defaultdict(list)
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                loss_dict, info_dict = model.get_loss(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss
                loss_tape.update(loss_dict, 1)
                for k, v in info_dict.items():
                    info_tape[k].append(v.detach().cpu().reshape(-1) if v.dim() > 0 else v.item())
        
        avg_loss = loss_tape.log(it, logger, writer, 'val', 
            others={f'{k}': wandb.Histogram(torch.cat(v)) if isinstance(v[-1], torch.Tensor) else np.mean(v) for k, v in info_tape.items()}
        )
        
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss


    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')