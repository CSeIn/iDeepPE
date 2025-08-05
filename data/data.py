# -*- coding: utf-8 -*-
""" Data loader for general speech separation
Also working on single-channel, multi-channel and varying number of speakers tasks

Co-author : GIST SAPL (hsK, JWOH)
Created on Jun 30, 2021 
"""

import json
import os

import numpy as np
import random
import math
import torch
import torchaudio
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from scipy import io

def audioread(path, sr):
    x, fs = torchaudio.load(path)
    # Resample the differently sampled audio file fs -> sr
    if fs != sr:
        resample = torchaudio.transforms.Resample(fs, sr)
        x = resample(x)
    return x, sr


class AudioDataset(data.Dataset):

    def __init__(self, json_files, num_spk=[1], sample_rate=16000, segment=4.0,
                 drop=0, seq=False, on_memory=True):
        """
        Args:
            json_dir:       Path to xx.json
            num_spk:        List of number of speakers
            sample_rate:    Sampling rate for audio files
            segment:        Duration of audio segment (s), when set to -1, use full audio
            drop:           Drop files shorter than drop (s)
            seq:            Sequential segmentation. If false, use one random segment for each file. Will not work if aug is True.
            on_memory:      Load data on memory
        """
        super(AudioDataset, self).__init__()

        self.audio_dict = {}
        self.seg_len = int(segment * sample_rate)
        self.sample_rate = sample_rate
        self.num_spk = num_spk


        self.seq = seq
        self.on_memory = on_memory
 
        
        min_len = int(drop * sample_rate)
        if seq: assert drop < segment
        
        drop_utt = 0
        drop_len = 0


        for nspk in num_spk:
            self.audio_dict[nspk] = []

        # for json_file in json_files:
        with open(json_files, 'r') as f:
            infos = json.load(f)

            nspk, self.num_ch = infos[0]
            # if not nspk in self.num_spk: 
            #     continue
            for paths in tqdm(infos[1:], mininterval=0.5):
                audio = []
                mixture, fs = audioread(paths[0], sample_rate)
                data_len = mixture.size(-1)
                
                if segment > 0:
                    if min_len > data_len:  # skip files shorter than drop length
                        drop_utt += 1
                        drop_len += data_len
                        continue
                
                if seq: #and self.seg_len > 0:
                    num_seg, rest_len = divmod(data_len, self.seg_len)
                    if on_memory:
                        # audio_dict item: [audio, pad_len]
                        # [C, T]
                        audio = torch.cat([audioread(path, sample_rate)[0] for path in paths],0)
                        if rest_len >= min_len:     # pad at front of the signal
                            temp_audio = audio[..., :rest_len]
                            temp_audio = F.pad(temp_audio, (self.seg_len-rest_len, 0))
                            self.audio_dict[nspk].append([temp_audio, self.seg_len-rest_len])
                        for i in range(num_seg):
                            self.audio_dict[nspk].append([audio[..., rest_len+self.seg_len*i:rest_len+self.seg_len*(i+1)], 0])
                    else:
                        # [paths, (start, end)]
                        # if start < 0, padding will be applied
                        if rest_len >= min_len:
                            self.audio_dict[nspk].append([paths, (rest_len-self.seg_len, rest_len)])
                        for i in range(num_seg):
                            self.audio_dict[nspk].append([paths, (rest_len+self.seg_len*i, rest_len+self.seg_len*(i+1))])
                # Full-length or random-segmentation audio init.
                else:
                    if on_memory:
                        audio = torch.cat([audioread(path, sample_rate)[0] for path in paths],0)
                        self.audio_dict[nspk].append([audio, 0])
                    else:
                        self.audio_dict[nspk].append([paths, 0])
                # num_seg, rest_len = divmod(data_len, self.seg_len)
                # if rest_len >= min_len:
                #     self.audio_dict[nspk].append([paths, (rest_len-self.seg_len, rest_len)])
                # for i in range(num_seg):
                #     self.audio_dict[nspk].append([paths, (rest_len+self.seg_len*i, rest_len+self.seg_len*(i+1))])
        
        if drop:
            print("Drop {} utts({:.2f} h) which is shorter than {} samples".format(
                    drop_utt, drop_len/sample_rate/3600, min_len))



        self.length = {}
        for nspk in self.num_spk:
            self.length[nspk] = len(self.audio_dict[nspk])


    def __getitem__(self, index):
        ind = index
        for nspk in self.num_spk:
            if ind < self.length[nspk]:
                break
            else:
                ind -= self.length[nspk]

        pad = False
        pad_range = [0, 0]

                
        
        if self.on_memory:
            audio, pad_len = self.audio_dict[nspk][ind]
            if pad_len:
                pad=True
                pad_range=[pad_len, 0]
        else:
            audio_paths, position = self.audio_dict[nspk][ind]
            audio = torch.cat([audioread(audio_path, self.sample_rate)[0] for audio_path in audio_paths[0:self.num_ch*3]])
            # sideinfo_path = audio_paths[self.num_ch*3]
            # print(sideinfo_path)
            # mat_file = io.loadmat(sideinfo_path)
            # sideinfo = mat_file['sideinfo']
            # VAD_time = sideinfo['VAD_time'][0][0][0]
            # TDOA_time = sideinfo['TDOA_time'][0][0]
            # ft1= torch.FloatTensor(VAD_time)
            # ft2= torch.FloatTensor(TDOA_time)
            # ft11=ft1.view([1,-1])
            # ft21=ft2.view([6,-1])
            # sideinfos=torch.cat([ft11,ft21], dim=0)
            # sideinfos= pad_range
            if self.seq:    # minus: padding, plus: position index
                start, end = position
                audio = audio[..., :end]
                # sideinfos = sideinfos[..., :end]
                if start < 0:
                    pad=True
                    pad_range=[-start, 0]
                    audio = F.pad(audio, (-start, 0), 'constant', 0)
                    # sideinfos = F.pad(sideinfos, (-start, 0), 'constant', 0)
                else:
                    audio = audio[..., start:]
                    # sideinfos = sideinfos[..., start:]
        # Random Segmentation
        if self.seg_len > 0:
            utt_len = audio.size(-1)
            if utt_len >= self.seg_len:  # segmentation
                start = np.random.randint(utt_len - self.seg_len + 1)
                end = start + self.seg_len
                audio = audio[..., start:end]
            else:   # pad signal
                pad_len = self.seg_len - utt_len
                pad_range = [pad_len, 0]
                audio = F.pad(audio, (pad_len, 0), 'constant', 0)

        # audio is a 2D tensor with shape
        # [
        #   mix-ch1
        #   mix-ch2
        #     ...
        #   src1-ch1
        #   src1-ch2
        #     ...
        #   src2-ch1
        #     ...
        # ]


        # spk_mask = torch.ones(nspk)

        # mix [M, T], src [M*S, T], spk_mask [S]
        # pad_range: list of [start, end]
        # return audio[:self.num_ch], audio[self.num_ch:], pad_range, spk_mask
        return audio[:self.num_ch], audio[self.num_ch*1:self.num_ch*2], audio[self.num_ch*2:self.num_ch*3], pad_range
        # return audio[:3], audio[3:6], audio[6:9], pad_range, sideinfos

    def __len__(self):
        return sum(self.length.values())


# ================================= For dataloader ===============================
def Audio_collate_fn(batch):
    mix_batch = []; src_batch = []; noise_batch = []; pad_masks = []; 

    num_len = len(set(map(lambda x: torch.Tensor.size(x[0], 1), batch)))
    # if num_len > 1:
    max_seg_len = max(batch, key=lambda x: x[0].size(1))[0].size(1)

    for bch in batch:
        mix, src, noise, pad_range = bch
        M, T = mix.size()

        padded_zeros = torch.zeros([M,max_seg_len-T])
        mix = torch.cat((mix,padded_zeros),1)
        src = torch.cat((src,padded_zeros),1)
        noise = torch.cat((noise,padded_zeros),1)
        # src = src.reshape(-1, M, T).transpose(0,1)     # M, S, T
        # src = src.reshape(M, T)     # M, T
        if num_len > 1:
            pad_mask = mix.new_ones(1, max_seg_len)
        else:
            pad_mask = mix.new_ones(1, mix.size(-1))

        # if num_len > 1:
        #     utt_len = mix.size(-1)
        #     if utt_len < max_seg_len:
        #         pad_size = max_seg_len - utt_len
        #         pad_range[0] = pad_size + pad_range[0]

        #         mix = F.pad(mix, (pad_size, 0), 'constant', 0)
        #         src = F.pad(src, (pad_size, 0), 'constant', 0)
        #         noise = F.pad(noise, (pad_size, 0), 'constant', 0)
        pad_mask[..., T+1:] = 0.

        # if pad_mask.sum()>0.5*T :
        mix_batch.append(mix)
        src_batch.append(src)
        noise_batch.append(noise)
        pad_masks.append(pad_mask)
        # sideinfo_batch.append(sideinfos)
        # spk_masks.append(spk_mask)

    # B: Batch / M: # of MICs / S: Source / T: Time
    mix_batch = torch.stack(mix_batch)  # [B, M, T]
    src_batch = torch.stack(src_batch)  # [B, M, T]
    noise_batch = torch.stack(noise_batch)  # [B, M, T]
    pad_masks = torch.stack(pad_masks)  # [B, 1, T]
    # sideinfo_batch = torch.stack(sideinfo_batch)  # [B, 1, T]
    # spk_masks = torch.stack(spk_masks)  # [B, S]

    return mix_batch, src_batch, noise_batch, pad_masks #sideinfo_batch



# Eval data part
class EvalDataset(data.Dataset):
    def __init__(self, tt_json, num_spk=[1], sample_rate=16000, on_memory=False):
        """
        Args:
            mix_path: directory including mixture wav files or path to .json file
        """
        super(EvalDataset, self).__init__()

        self.num_spk = num_spk
        self.audio_dict = {}
        self.on_memory = on_memory
        self.sample_rate = sample_rate
        
        for nspk in num_spk:
            self.audio_dict[nspk] = []

        #for json_file in tt_json:
        with open(tt_json, 'r') as f: infos = json.load(f)
        nspk, self.num_ch = infos[0]
        # if not nspk in self.num_spk: continue

        for paths in tqdm(infos[1:], mininterval=0.5):
            if self.num_ch == 1:
                file_name = os.path.splitext(os.path.basename(paths[0]))[0]
            else:
                file_name = os.path.splitext(os.path.basename(paths[0]))[0] #need to set reference mic number index-1, CHiME4 : 5-1 = 4
            
            if on_memory:
                audio = torch.cat([audioread(path, sample_rate)[0] for path in paths], 0)
                self.audio_dict[nspk].append([audio, file_name])
            else:
                self.audio_dict[nspk].append([paths, file_name])

        self.length = {}
        for nspk in self.num_spk:
            self.length[nspk] = len(self.audio_dict[nspk])


    def __getitem__(self, index):
        ind = index
        for i, (nspk, l) in enumerate(self.length.items()):
            if ind < l:
                break
            else:
                ind -= l

        audio_path, file_name = self.audio_dict[nspk][ind]
        # audio_paths, position = self.audio_dict[nspk][ind]
        if not self.on_memory:
            # audio = torch.cat([audioread(audio_path, self.sample_rate)[0]] for audio_path in audio)
            audio = torch.cat([audioread(audio_path, self.sample_rate)[0] for audio_path in audio_path[0:self.num_ch*3]])
            # sideinfo_path = audio_path[self.num_ch*3]
            # mat_file = io.loadmat(sideinfo_path)
            # sideinfo = mat_file['sideinfo']
            # VAD_time = sideinfo['VAD_time'][0][0][0]
            # TDOA_time = sideinfo['TDOA_time'][0][0]
            # ft1= torch.FloatTensor(VAD_time)
            # ft2= torch.FloatTensor(TDOA_time)
            # ft11=ft1.view([1,-1])
            # ft21=ft2.view([6,-1])
            # sideinfos=torch.cat([ft11,ft21], dim=0)
            
        return audio[:self.num_ch], audio[self.num_ch*1:self.num_ch*2], audio[self.num_ch*2:self.num_ch*3], file_name#, sideinfos

    def __len__(self):
        return sum(self.length.values())


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    assert len(batch) == 1

    bch = batch[0]

    mix, src, noise, file_name= bch
    M,T = mix.size()
    # src = src.reshape(-1, M, T).transpose(0,1)
    # src = src.reshape(M, T)
    
    mix_batch = torch.stack([mix])
    src_batch = torch.stack([src])
    noise_batch = torch.stack([noise])
    # spk_masks = torch.stack([spk_mask])
    file_names = [file_name]
    # sideinfo_batch = torch.stack([sideinfo])


    return mix_batch, src_batch, noise_batch, file_names
