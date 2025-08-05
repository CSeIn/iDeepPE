#!/usr/bin/env python

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

from data.data import EvalDataset, EvalDataLoader
from models.modules.conv_stft import ConvSTFT, ConviSTFT
import scipy
import pysepm
import scipy.io
from util.oracle_test_mse import *



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


parser = argparse.ArgumentParser('oracle MVDR for preparing the data for train_PF')
parser.add_argument('--tt-json', type=str, default="data/json/tr1spk-6ch.json",
                    help='path to tt.json')
parser.add_argument('--num-spk', type=int, nargs='+', default=[1],
                    help='Number of speakers. (ex. --num-spk 2 3 4)')
parser.add_argument('--out-dir', type=str, default='oracle_MVDR_out',
                    help='Directory putting separated wav files')
parser.add_argument('--use-cuda', type=int, default=1,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample-rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--mode', type=str, default='tr',
                    help='tr? dt? et?')
parser.add_argument('--save', type=int, default=1,
                    help='Save separated wave file')


def evaluate(args):
    # STFT / ISTFT
    win_len = 512
    win_inc = 256
    fft_len = 512
    win_type = 'hann'# 'hanning'
    fix = True
    win_len = win_len
    win_inc = win_inc
    fft_len = fft_len
    win_type = win_type

    stft = ConvSTFT(win_len, win_inc, fft_len, win_type,
                    feature_type='real', fix=fix)
    istft = ConviSTFT(win_len, win_inc, fft_len, win_type,
                      feature_type='real', fix=fix)
    stft.cuda()
    istft.cuda()

    # Load data
    eval_dataset = EvalDataset(
        args.tt_json, args.num_spk, sample_rate=args.sample_rate)
    eval_loader = EvalDataLoader(eval_dataset, batch_size=1)
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(
        args.out_dir,  args.mode), exist_ok=True)


    with torch.no_grad():
        t = tqdm(total=len(eval_dataset), mininterval=0.5)
        for i, data in enumerate(eval_loader):
            mix_batch, src_batch, noise_batch, file_names = data
            pad_mask = torch.ones((1, 1, mix_batch.size(-1)))

            if args.use_cuda:
                mix_batch = mix_batch.cuda()
                src_batch = src_batch.cuda()
                noise_batch = noise_batch.cuda()
                pad_mask = pad_mask.cuda()
                num_mic = torch.zeros(1).type(mix_batch.type()).cuda()

            B, C, T = mix_batch.size()  
            batch_input,  batch_target1, batch_target2, target_phiS_dB = create_input_target(mix_batch, src_batch, noise_batch)  
            
            B, K2, L = batch_target1.size()  
            batch_target2 = batch_target2.transpose(1,2).contiguous()
            batch_target2 = batch_target2.view(B,L,257,10).contiguous()
            spp1_oracle = batch_target1[:,:,:]
            sin_RTF_oracle =  batch_target2[:,:,:,0:5]
            cos_RTF_oracle =  batch_target2[:,:,:,5:10]
            RTF_oracle = imap_RTF(sin_RTF_oracle, cos_RTF_oracle)
            
            estimated_zz, estimated_zs, estimated_zn = mvdr_oracle2(mix_batch, src_batch, noise_batch, spp1_oracle, RTF_oracle)
            estimated_zz= torch.sum(estimated_zz,dim=0)
            estimated_zs= torch.sum(estimated_zs,dim=0)
            estimated_zn= torch.sum(estimated_zn,dim=0)
            
            if args.save:
                file_name_zz = os.path.join(args.out_dir, args.mode, file_names[0])
                file_name_zs = os.path.join(args.out_dir, args.mode, file_names[0])
                file_name_zn = os.path.join(args.out_dir, args.mode, file_names[0])
                
                torchaudio.save(file_name_zz+'.wav',       estimated_zz.cpu(), args.sample_rate)
                torchaudio.save(file_name_zs+'.Clean.wav', estimated_zs.cpu(), args.sample_rate)
                torchaudio.save(file_name_zn+'.Noise.wav', estimated_zn.cpu(), args.sample_rate)

            t.update()

    return 0


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    z = evaluate(args)

