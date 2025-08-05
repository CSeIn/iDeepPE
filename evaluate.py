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
from util.criterion_ms import PESQ, eSTOI, SI_SNR, STOI, PESQNB2
from models.conformer_cmgan import TSCNet, TSCNet_PF
from models.modules.conv_stft import ConvSTFT, ConviSTFT
import scipy
import pysepm
from util.map_function import map_selector
import scipy.io
from util.gain import mmse_lsa
from util.oracle_test_mse import *
import pandas as pd


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


parser = argparse.ArgumentParser('Separate speech')
parser.add_argument('--modelBF-path', type=str, default="out/BF_model/temp_best.pth.tar",
                    help='Path to model_BF file created by training')
parser.add_argument('--modelPF-path', type=str, default="out/PF_model/temp_best.pth.tar",
                    help='Path to model_PF file created by training')
parser.add_argument('--tt-json', type=str, default="data/json/et1spk-6ch.json",
                    help='path to tt.json')
parser.add_argument('--num-spk', type=int, nargs='+', default=[1],
                    help='Number of speakers. (ex. --num-spk 2 3 4)')
parser.add_argument('--out-dir', type=str, default='out/waves/iDeepPE',
                    help='Directory putting separated wav files')
parser.add_argument('--use-cuda', type=int, default=1,
                    help='Whether use GPU to separate speech')
parser.add_argument('--sample-rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--fake-target', type=int, default=0,
                    help='Fill valid target with fake target.')
parser.add_argument('--out-file', type=str,  
                    default="out/waves/result_iDeepPE.txt",
                    help='Output result file path')
parser.add_argument('--save', type=int, default=1,
                    help='Save enhanced wave file')


def evaluate(args):
    # Load model

    model = TSCNet(num_channel=48, num_features=257)

    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()        

    model_PF = TSCNet_PF(num_channel=48, num_features=257)

    if args.use_cuda:
        model_PF = torch.nn.DataParallel(model_PF)
        model_PF.cuda()        
        
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

    #BF speech PSD (multichannel) 
    mphis_map = map_selector('NormalCDF', None)
    mphis_prior = scipy.io.loadmat('util/phiS_prior_multichannel.mat')
    mu_mphis = mphis_prior['mu_phiS']
    std_mphis = mphis_prior['std_phiS']  
    mphis_map.mu = torch.tensor(mu_mphis).cuda()
    mphis_map.sigma = torch.tensor(std_mphis).cuda()
    
    #PF
    xi_map = map_selector('NormalCDF', None)
    xi_prior = scipy.io.loadmat('util/xi_prior_PF.mat')
    mu_xi = xi_prior['mu_xi']
    std_xi = xi_prior['std_xi']
    xi_map.mu = torch.tensor(mu_xi).cuda()
    xi_map.sigma = torch.tensor(std_xi).cuda()
    phis_map = map_selector('NormalCDF', None)
    phis_prior = scipy.io.loadmat('util/phiS_prior_PF.mat')
    mu_phis = phis_prior['mu_phiS']
    std_phis = phis_prior['std_phiS']  
    phis_map.mu = torch.tensor(mu_phis).cuda()
    phis_map.sigma = torch.tensor(std_phis).cuda() 
    
    model_info = torch.load(args.modelBF_path)
    try:
        model.load_state_dict(model_info['model_state_dict'])
    except KeyError:
        state_dict = OrderedDict()
        for k, v in model_info['model_state_dict'].items():
            name = k.replace("module.", "")    # remove 'module.'
            state_dict[name] = v
        model.load_state_dict(state_dict)
        
    model_info = torch.load(args.modelPF_path)
    try:
        model_PF.load_state_dict(model_info['model_state_dict'])
    except KeyError:
        state_dict = OrderedDict()
        for k, v in model_info['model_state_dict'].items():
            name = k.replace("module.", "")    # remove 'module.'
            state_dict[name] = v
        model_PF.load_state_dict(state_dict)
    # print(model)
    model.eval()
    model_PF.eval()
    # Load data
    eval_dataset = EvalDataset(
        args.tt_json, args.num_spk, sample_rate=args.sample_rate)
    eval_loader = EvalDataLoader(eval_dataset, batch_size=1)
    os.makedirs( args.out_dir, exist_ok=True)

    results = []
    perf_si_snr = {}
    perf_pesq = {}
    perf_pesqnb = {}
    perf_estoi = {}
    perf_csig = {}
    perf_cbak = {}
    perf_covl = {}

    for n in args.num_spk:
        perf_si_snr[n] = []
        perf_pesq[n] = []
        perf_pesqnb[n] = []
        perf_estoi[n] = []
        perf_csig[n] = []
        perf_cbak[n] = []
        perf_covl[n] = []

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
            batch_input, batch_target1_cmgan, batch_target2_cmgan, target_phiS_dB_cmgan = create_input_target_cmgan(mix_batch, src_batch, noise_batch) 
            
            spp1_oracle = batch_target1_cmgan.permute(0,2,1).contiguous()
            target_phiS_dB = target_phiS_dB_cmgan.permute(0,2,1).contiguous() 
            batch_target2 = batch_target2_cmgan.permute(0,2,3,1).contiguous() #[B,10,T,F]->[B,T,F,10]
            sin_RTF_oracle =  batch_target2[:,:,:,0:5]
            cos_RTF_oracle =  batch_target2[:,:,:,5:10]
            RTF_oracles = imap_RTF2(sin_RTF_oracle, cos_RTF_oracle)
            target_mphiS = torch.pow(10, target_phiS_dB/10) #[B,F,T]      

            ## BF model out (cmgan) ##
            dnn_output1, dnn_output2,  dnn_output3= model(batch_input) #input: [1,11,T,F] output1:[1,T,F], output2:[1,10,T,F] 
            # apriori SPP
            dnn_output1=dnn_output1.squeeze(dim=1)
            spp1_est= dnn_output1.permute(0,2,1).contiguous() #[B,F,T]
            # RTF
            B, K, L, F2 = dnn_output2.size() 
            dnn_output2 = dnn_output2.permute(0,2,3,1).contiguous() #[B,10,T,F]->[B,T,F,10]
            sin_RTF_est =  dnn_output2[:,:,:,0:5]
            cos_RTF_est =  dnn_output2[:,:,:,5:10]
            RTF_est = imap_RTF2(sin_RTF_est, cos_RTF_est) 
            # mphiS
            dnn_output3= dnn_output3.squeeze(dim=1)
            mphis_mapped = dnn_output3.permute(0,2,1).contiguous() #[B,F,T]
            estimated_mphis_dB = mphis_map.inverse(mphis_mapped.permute(0,2,1)).permute(0,2,1)          
            
            ## MVDR ##  
            mix_comp, RTF_est = eval_mvdr_part1(mix_batch, RTF_est)  
            estimated_zz, noise_cov, Y_pow = eval_mvdr_part2(mix_comp, spp1_est, RTF_est, 0) #noncausal MCRA(BMCRA)
            input_batch, Z_STFT_mag, Z_STFT_pha = eval_mvdr_part3(estimated_zz, 1)
            
            #model BF phis estimate
            pow_Y_dB = torch.abs(10*torch.log10(Y_pow[:,:,:,0,0]+1e-12))
            estimated_mphis_dB = torch.clamp(estimated_mphis_dB, max=pow_Y_dB) 
            estimated_mphis = torch.pow(10, estimated_mphis_dB/10) 
            #null- beamformed signal
            ref_mic_Y = mix_batch[:,0,:]
            Rmic_Y_STFT_mag, Rmic_Y_STFT_pha = stft(ref_mic_Y)
            Rmic_pow_Y = torch.pow(Rmic_Y_STFT_mag,2)
            pow_Z = torch.pow(Z_STFT_mag,2)
            G_mvdr = torch.clamp(torch.div(pow_Z,Rmic_pow_Y+1e-12), max =1) 

            ## PF model out(cmgan) ##
            input_batch = input_batch.permute(0,2,1).contiguous() #[B,F,T]->[B,T,F]
            dnn_mvdr_output, dnn_mvdr_output2, dnn_mvdr_output3 = model_PF(input_batch) #output:[1,T,F]
            estimated_xi_mapped = dnn_mvdr_output.permute(0,2,1).contiguous() #[B,T,F]->[B,F,T]
            spp_post  = dnn_mvdr_output2.permute(0,2,1).contiguous()
            estimated_phis_mapped = dnn_mvdr_output3.permute(0,2,1).contiguous()
            ###

            ### MMSE estimator ###
            ws_ps = 0.5
            wo_ps = 0
            w_xi = 0.01
            w_eta = 0.99
            w_phis = 0.045

            #SPP integration 
            spp_post_s = ws_ps*spp1_est + (1-ws_ps)*spp_post
            spp_post_o = wo_ps*spp1_est + (1-wo_ps)*spp_post
            pow_Z_dB = torch.abs(10*torch.log10(pow_Z+1e-12))

            estimated_xi_dB = xi_map.inverse(estimated_xi_mapped.permute(0,2,1)).permute(0,2,1)
            estimated_phis_dB = phis_map.inverse(estimated_phis_mapped.permute(0,2,1)).permute(0,2,1)

            estimated_phis_dB = torch.clamp(estimated_phis_dB, max=pow_Z_dB) 
            estimated_xi_PF = torch.pow(10, estimated_xi_dB/10) 
            estimated_phis = torch.pow(10, estimated_phis_dB/10)  
            mmse_phin =  torch.mul(torch.div(1, 1+estimated_xi_PF),pow_Z)  

            #SNR integration 
            phiS_integrated = torch.mul(w_xi, estimated_mphis) + torch.mul(1-w_xi, estimated_phis)
            xi_BF = torch.div(phiS_integrated,mmse_phin+1e-12) 
            estimated_xi_s = xi_BF
            estimated_phis_s = estimated_phis 
            phiS_H1 = torch.mul(torch.pow(torch.div(estimated_xi_s, 1+estimated_xi_s),2),pow_Z) + torch.mul(torch.div(1, 1+estimated_xi_s), estimated_phis_s)            
            phiS = torch.mul(spp_post_s, phiS_H1)

            #NSR integration 
            noise_ref_BF = torch.real(noise_cov[:,:,:,0,0]) 
            estimated_mphio = torch.mul(G_mvdr,noise_ref_BF) 
            phiO_integrated = torch.mul(w_eta, estimated_mphio) + torch.mul(1-w_eta, mmse_phin) 
            xi_mcra = torch.div(estimated_phis,phiO_integrated+1e-12) 
            estimated_xi_o = xi_mcra
            estimated_phio = mmse_phin
            phiN_H1 = torch.mul(torch.pow(torch.div(1, 1+estimated_xi_o),2),pow_Z) + torch.mul(torch.div(estimated_xi_o, 1+estimated_xi_o), estimated_phio) 
            phiN = torch.mul(1-spp_post_o, pow_Z) + torch.mul(spp_post_o, phiN_H1)

            # Phis integration: refined phis 
            sm_a2 = w_phis*mphis_mapped 
            phiS = torch.mul(sm_a2, estimated_mphis) + torch.mul(1-sm_a2, phiS)
            
            xi_post = torch.div(phiS, phiN+1e-12) #a priori
            gamma_post =torch.div(pow_Z, phiN+1e-12) #a posteriori
            G_mmselsa = mmse_lsa(xi_post,gamma_post) 
            G_mmselsa = torch.clamp(G_mmselsa,min= 10**(-32/20))

            # estimate_mag =  Z_STFT_mag #MVDR
            estimate_mag = torch.mul(G_mmselsa, Z_STFT_mag) #MVDR +PF
            estimate_source= istft(estimate_mag, Z_STFT_pha)
            
            num_spk = 1
            src_resize = src_batch[:, 0, 0:estimate_source.shape[2]]
            mix_resize = mix_batch[:, 0, 0:estimate_source.shape[2]]
            pad_resize = pad_mask[:, :, 0:estimate_source.shape[2]]
            
            ref_cpu = src_resize.squeeze().cpu().numpy()
            mix_cpu = mix_resize.squeeze().cpu().numpy()
            est_cpu = estimate_source.squeeze().cpu().numpy()


            utt_si_snr = SI_SNR(src_resize, estimate_source, pad_resize)
            utt_pesq = PESQ(ref_cpu,est_cpu,16000)
            utt_pesqnb = PESQNB2(ref_cpu,est_cpu,16000)
            utt_estoi = STOI(ref_cpu,est_cpu,16000)
            csig, cbak, covl = pysepm.composite(ref_cpu, est_cpu, 16000)
            
            results.append({
                'File Name': file_names[0],
                'utt_pesq': utt_pesq,
                'utt_pesqnb': utt_pesqnb,
                'utt_estoi': utt_estoi,
                'utt_si_snr': utt_si_snr.squeeze().cpu().numpy().item(),
                'csig': csig,
                'cbak': cbak,
                'covl': covl,
         
            })

            perf_si_snr[num_spk].append(utt_si_snr.squeeze().cpu().numpy())
            perf_pesq[num_spk].append(utt_pesq)
            perf_pesqnb[num_spk].append(utt_pesqnb)
            perf_estoi[num_spk].append(utt_estoi)
            perf_csig[num_spk].append(csig)
            perf_cbak[num_spk].append(cbak)
            perf_covl[num_spk].append(covl)            

            if args.save:      
                scs = estimate_source[0].cpu()
                mix = mix_batch[:,0,:].cpu()    
                src = src_batch[:,0,:].cpu()
                
                file_name = os.path.join(args.out_dir, file_names[0])
                torchaudio.save(file_name + 'est.wav', scs, args.sample_rate, encoding="PCM_S", bits_per_sample=16)    

            t.update()

    df = pd.DataFrame(results)
    excel_file_path = os.path.join(args.out_dir, 'results.xlsx')
    df.to_excel(excel_file_path, index=False)

    si_snr_cpu = perf_si_snr[1]
    pesq_cpu = perf_pesq[1]
    pesqnb_cpu = perf_pesqnb[1]
    estoi_cpu = perf_estoi[1]
    csig_cpu = perf_csig[1]
    cbak_cpu = perf_cbak[1]
    covl_cpu = perf_covl[1]

    return si_snr_cpu, pesq_cpu, estoi_cpu, pesqnb_cpu, csig_cpu, cbak_cpu, covl_cpu


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    si_snr_cpu, pesq_cpu, estoi_cpu, pesqnb_cpu, csig_cpu, cbak_cpu, covl_cpu= evaluate(
        args)
    mean_si_snr = np.mean(si_snr_cpu)
    mean_pesq = np.mean(pesq_cpu)
    mean_estoi = np.mean(estoi_cpu)
    mean_pesqnb = np.mean(pesqnb_cpu)
    mean_csig = np.mean(csig_cpu)
    mean_cbak = np.mean(cbak_cpu)
    mean_covl = np.mean(covl_cpu)

    with open(args.out_file, 'w') as f:

        f.write('average PESQ: {:.10f}'.format(mean_pesq))
        f.write('\n{}\n'.format('-'*50))        
        f.write('average PESQNB: {:.10f}'.format( mean_pesqnb ))
        f.write('\n{}\n'.format('-'*50))        
        f.write('average eSTOI: {:.10f}'.format(mean_estoi))
        f.write('\n{}\n'.format('-'*50))
        f.write('average SI-SNR: {:.10f}'.format(mean_si_snr))
        f.write('\n{}\n'.format('-'*50))
        f.write('average csig: {:.10f}'.format(mean_csig))
        f.write('\n{}\n'.format('-'*50))
        f.write('average cbak: {:.10f}'.format(mean_cbak))
        f.write('\n{}\n'.format('-'*50))
        f.write('average covl: {:.10f}'.format(mean_covl))
        f.write('\n{}\n'.format('-'*50))

    with open(args.out_file, 'r') as f:
        text = f.read()
    print('-'*50)
    print(text)
