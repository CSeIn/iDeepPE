# -*- coding: utf-8 -*-

import librosa
import argparse
import json
import os
from glob import glob
from scipy import io

parser = argparse.ArgumentParser(description="Data preprocessing",
                                epilog="Before implementing this code, make sure that you extracted clean, noise")

# LibriSpeech based loader is not working yet
parser.add_argument('--in-dir',  type=str,  #required=True,
                    default="oracle_MVDR_out",
                    help='input Directory path including tr, dt, et')
parser.add_argument('--out-dir', type=str,  #required=True,
                    default="data/json/",
                    help='Directory path to put output .json files')
parser.add_argument('--num-sp', type=int, default=1,
                    help='Number of speakers')
parser.add_argument('--num-ch', type=int, default=1,
                    help='Number of channels')
parser.add_argument('--tr', type=int, default=1,
                    help='Make tr.json')
parser.add_argument('--dt', type=int, default=1,
                    help='Make dt.json')
# parser.add_argument('--et', type=int, default=1,
#                     help='Make et.json')

def audioread(path, sr=None):
    x, fs = librosa.load(path, 16000)
    return x, sr

def preprocess(args):
    in_dir = os.path.abspath(args.in_dir)

    # in_dir_side = os.path.abspath(args.in_dir_side)
    # Make json files for following datasets
    data_types = []
    if args.tr:
        data_types.append('tr')
    if args.dt:
        data_types.append('dt')
    # if args.et:
    #     data_types.append('et')

    for data_type in data_types:
        file_infos = []
        file_infos.append([1, args.num_ch])
    
        audio_paths_clean = sorted(glob(in_dir + '/' + data_type + '/*.Clean.wav', recursive=True))
    
        for wav_file in audio_paths_clean:
            wav_paths = []
            path_parts = wav_file.split(os.path.sep)
            uttid, _ = os.path.splitext(path_parts[-1])
            
    
            relative_path  = in_dir + '/' + data_type + '/' + uttid[0:-6] + '.wav'
            wav_paths.append(relative_path)
    
            clean_path  = in_dir + '/' + data_type + '/' + uttid[0:-6] + '.Clean.wav'
            wav_paths.append(clean_path)
    
            noise_path  = in_dir + '/' + data_type + '/' + uttid[0:-6] + '.Noise.wav'
            wav_paths.append(noise_path)
                
            file_infos.append(wav_paths)   
     
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        with open(os.path.join(args.out_dir, 'oracle_MVDR_' + data_type + '.json'), 'w') as f:
            json.dump(file_infos, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    preprocess(args)