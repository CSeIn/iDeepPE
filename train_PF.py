# -*- coding: utf-8 -*-

import argparse

import os
import torch
import numpy as np
import random

from data.data import *
from solver_PF import Solver
from models.conformer_cmgan import TSCNet_PF
from collections import OrderedDict

import wandb

parser = argparse.ArgumentParser()

def param(x):
    try:
        if x.startswith('('):
            y = tuple(float(s) for s in x.strip('()').split(','))
        elif x.startswith('['):
            y = list(int(s) for s in x.strip('[]').split(','))
        elif x.startswith('"'):
            y = x.strip('"')
        elif x.startswith("'"):
            y = x.strip("'")
        elif '.' in x:
            y = float(x)
        else:
            y = int(x)
        return y
    except:
        raise argparse.ArgumentTypeError()
        

# General config
# Dataset
parser.add_argument('--tr-json', default="data/json/oracle_MVDR_tr.json",
                    help='path to oracleMVDR outpt tr.json')
parser.add_argument('--cv-json', default="data/json/oracle_MVDR_dt.json",
                    help='path to oracleMVDR outpt dt.json')
parser.add_argument('--randomSNR', type=int, default=False,
                    help='Whether mixing random SNR noisy waves')
parser.add_argument('--num-spk', type=int, nargs='+', default=[1],
                    help='Number of speakers. (ex. --num-spk 2 3 4)')
parser.add_argument('--sample-rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4.0, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--drop', default=0, type=float,
                    help='Drop files shorter than this input (seconds).')
parser.add_argument('--aug', type=int, default=0,
                    help='Number of data to augmented for each speaker number. 0 for no augmentation')
parser.add_argument('--seq', type=int, default=0,
                    help='If True, sequential segmentation will be performed.')
parser.add_argument('--fake-target', type=int, default=0,
                    help='Fill valid target with fake target.')
parser.add_argument('--on-memory', type=int, default=0,
                    help='Load dataset on memory')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Shuffle datasets')
parser.add_argument('--batch-size', type=int, default=5,
                    help='Batch size')
parser.add_argument('--batch-type', type=str, default='uniform-exclusive',
                    choices=["random", "uniform", "exclusive", "uniform-exclusive"])
parser.add_argument('--num-workers', type=int, default=1,
                    help='Number of wokers for batch')
# Network architecture
parser.add_argument('--gainweights',type=param, nargs='+', default=[1.2, 1.0, 0.1], # #xi, SPP, phis
                    help='g')
# Training config
parser.add_argument('--use-cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=1000, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--early-stop', default=1000, type=int,
                    help='Early stop training when no improvement for (input) epochs')
parser.add_argument('--max-norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# optimizer
parser.add_argument('--optimizer', default='AdamW', type=str,
                    choices=['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam',
                             'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD'],
                    help='Optimizer See pytorch docs.')
parser.add_argument('--opt-params', type=param, nargs='+',
                    default=[5.e-4, (0.9,0.999), 1.e-8, 0., 0.],
                    help='Parameters for optimizer\n'
                    'Adadelta:      lr, rho, eps, weight_decay\n'
                    'Adagrad:       lr, lr_decay, weight_decay, initial_accumulator_value, eps\n'
                    'Adam:          lr, betas, eps, weight_decay, amsgrad\n'
                    'AdamW:         lr, betas, eps, weight_decay, amsgrad\n'
                    'SparseAdam:    lr, betas, eps\n'
                    'Adamax:        lr, betas, eps, weight_decay\n'
                    'ASGD:          lr, lambd, alpha, t0, weight_decay\n'
                    'LBFGS:         lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size, line_search_fn\n'
                    'RMSprop:       lr, alpha, eps, weight_decay, momentum, entered\n'
                    'Rprop:         lr, etas, step_sizes\n'
                    'SGD:           lr, momentum, dampening, weight_decay, nesterov')
# scheduler
parser.add_argument('--scheduler', type=str, default='StepLR',
                    choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
                             'ReduceLROnPlateau', 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'CosineAnnealingWarmUpRestarts'],
                    help='lr_scheduler. See pytorch docs.')
parser.add_argument('--scheduler-params', type=param, nargs='+',
                    default=[400, 0.5],
                    help='Parameters for scheduler\n'
                    'StepLR:                step_size, gamma\n'
                    'MultiStepLR:           milestones, gamma\n'
                    'ExponentialLR:         gamma\n'
                    'CosineAnnealingLR:     T_max, eta_min\n'
                    'ReduceLROnPlateau:     mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps\n'
                    'OneCycleLR:            max_lr, total_steps, epochs, step_per_epoch,\n'
                    '                       pct_start, anneal_strategy, cycle_momentum,\n'
                    '                       base_momentum, max_momentum, div_factor, final_div_factor\n'
                    'CosineAnnealingWarmRestarts    T_0, T_mult, eta_min'
                    'CosineAnnealingWarmUpRestarts   T_0, T_mult, eta_max, T_up, gamma')
# save and load model
parser.add_argument('--save-folder', default='out/PF_',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='', nargs='?',
                    help='Continue from checkpoint model')
parser.add_argument('--seed', default=-1,
                    help='Random seed', type=int)
parser.add_argument('--tseed', default=-1,
                    help='Torch random seed', type=int)
parser.add_argument('--nseed', default=-1,
                    help='Numpy random seed', type=int)
# logging
parser.add_argument('--num-print', default=5, type=int,
                    help='Number of printing per epoch')

def main(args):
    os.makedirs(args.save_folder, exist_ok=True)

    # Generate random seed
    seed = args.seed
    if seed < 0:
        try:
            with open('/dev/random', 'rb') as f:
                seed = int.from_bytes(f.read(4), byteorder='little')
        except:
            seed = int.from_bytes(os.urandom(4), byteorder='little')

    nseed = args.nseed
    if nseed < 0:
        try:
            with open('/dev/random', 'rb') as f:
                nseed = int.from_bytes(f.read(4), byteorder='little')
        except:
            nseed = int.from_bytes(os.urandom(4), byteorder='little')

    tseed = args.tseed
    if tseed < 0:
        try:
            with open('/dev/random', 'rb') as f:
                tseed = int.from_bytes(f.read(4), byteorder='little')
        except:
            tseed = int.from_bytes(os.urandom(4), byteorder='little')
    print('random seed: {0:>10d}\nnumpy seed:  {1:>10d}\ntorch seed:  {2:>10d}'.format(
                                                                    seed, nseed, tseed))
    random.seed(seed)
    np.random.seed(nseed)
    torch.manual_seed(tseed)
    
    
    # model

    model = TSCNet_PF(num_channel=48, num_features=257)
    # torch.backends.cudnn.benchmark = True 
    
    print('Model size: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    if args.use_cuda:
        print("cuda on!", flush=True)
        model = torch.nn.DataParallel(model)
        model.cuda()
        print("cuda done!", flush=True)

    # data
    print("Loading training data set...", flush=True)
    tr_dataset = AudioDataset(args.tr_json, num_spk=args.num_spk,
                                   sample_rate=args.sample_rate, segment=args.segment,
                                   drop=args.drop, seq=args.seq,
                                   on_memory=args.on_memory)
    print("Loading validation data set...", flush=True)
    cv_dataset = AudioDataset(args.cv_json, num_spk=args.num_spk,
                                   sample_rate=args.sample_rate, segment=args.segment,
                                   drop=args.drop, seq=args.seq,
                                   on_memory=args.on_memory)
    
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                            num_workers=args.num_workers, collate_fn=Audio_collate_fn, pin_memory=True)
    cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, collate_fn=Audio_collate_fn, pin_memory=True)

    # optimizer
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), *args.opt_params)
    # optimizer=torch.optim.Adam(lr=0.001, clipvalue=1.0)
    # solver
    print_freq = max(int(len(tr_dataset)/args.batch_size/args.num_print), 1)
    solver = Solver(tr_loader, cv_loader, model, optimizer, print_freq, args)
    solver.train()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    wandb.init(project='project_name') 
    wandb.run.name = 'PF_experiment' # Set a name for the run
    main(args)
