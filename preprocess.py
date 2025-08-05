# -*- coding: utf-8 -*-
import librosa
import argparse
import json
import os
from glob import glob
from scipy import io

parser = argparse.ArgumentParser(description="Data preprocessing",
                                epilog="If args.num_ch == 1, data is split by speakers."
                                        " Otherwise, data is split by sample")

# LibriSpeech based loader is not working yet
parser.add_argument('--in-dir',  type=str,  #required=True,
                    default="H:/CHiME3/16kHz/isolated", #data1/ms/DB/CHiME3/data/audio/16kHz/isolated",
                    help='input Directory path including tr, dt, et')
parser.add_argument('--in-dir-ext',  type=str,  #required=True,
                    default="H:/CHiME3/16kHz/isolated_ext", #data1/ms/DB/CHiME3/data/audio/16kHz/isolated_ext",
                    help='input_ext Directory path including tr, dt, et')
parser.add_argument('--out-dir', type=str,  #required=True,
                    default="data/json/",
                    help='Directory path to put output .json files')
parser.add_argument('--num-sp', type=int, default=1,
                    help='Number of speakers, 1 for speech enhancement')
parser.add_argument('--num-ch', type=int, default=6,
                    help='Number of microphones')
parser.add_argument('--tr', type=int, default=1,
                    help='Make tr.json')
parser.add_argument('--dt', type=int, default=1,
                    help='Make dt.json')
parser.add_argument('--et', type=int, default=1,
                    help='Make et.json')




def audioread(path, sr=None):
    x, fs = librosa.load(path, 16000)
    return x, sr

def preprocess(args):
    in_dir = os.path.abspath(args.in_dir)
    in_dir_ext = os.path.abspath(args.in_dir_ext)
    # Make json files for following datasets
    data_types = []
    if args.tr:
        data_types.append('tr')
    if args.dt:
        data_types.append('dt')
    if args.et:
        data_types.append('et')

    # Single speaker
    ref_mic_idx = 5
    mic_order = [ref_mic_idx]+ list(range(1,ref_mic_idx))+ list(range(ref_mic_idx+1,args.num_ch+1))
    for data_type in data_types:
        file_infos = []
        file_infos.append([1, args.num_ch])
        audio_paths_noisy = sorted(glob(in_dir + '/' + data_type +'05_*_simu/*.CH5.wav', recursive=True))
        for wav_file in audio_paths_noisy:
            wav_paths = []
            path_parts = wav_file.split(os.path.sep)
            uttid, _ = os.path.splitext(path_parts[-1])
            
            for idx in mic_order:
                relative_path  = in_dir + '/' + path_parts[-2] + '/' + uttid[0:-1] + '{}'.format(idx) + '.wav'
                wav_paths.append(relative_path)

            for idx in mic_order:
                clean_path  = in_dir_ext + '/' + path_parts[-2] + '/' + uttid[0:-1] + '{}'.format(idx) + '.Clean.wav'
                wav_paths.append(clean_path)

            for idx in mic_order:
                noise_path  = in_dir_ext + '/' + path_parts[-2] + '/' + uttid[0:-1] + '{}'.format(idx) + '.Noise.wav'
                wav_paths.append(noise_path)
                
            file_infos.append(wav_paths)   
 
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        with open(os.path.join(args.out_dir, data_type + '{}spk-{}ch.json'.format(args.num_sp, args.num_ch)), 'w') as f:
            json.dump(file_infos, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    preprocess(args)