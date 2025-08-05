# -*- coding: utf-8 -*-

import os
import time

import torch
import torch.nn as nn
import numpy as np
import random
from models.modules.conv_stft import ConvSTFT, ConviSTFT
from util.criterion_ms import si_sdr, PESQ, eSTOI, SI_SNR
from util.map_function import map_selector
import scipy.io
from util.oracle_test_mse import *
from models.modules.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
import wandb

class Solver(object):

    def __init__(self, tr_loader, cv_loader, model, optimizer, print_freq, args):

        self.tr_loader = tr_loader
        self.cv_loader = cv_loader
        self.model = model
        self.optimizer = optimizer
        self.batch_size = args.batch_size

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        self.scheduler_name = args.scheduler
       

        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        # logging
        self.print_freq = print_freq
        # loss
        self.tr_loss = []
        self.cv_loss = []
        
        # STFT / ISTFT
        win_len=512
        win_inc=256
        fft_len=512
        win_type='hann'# 'hanning' 
        fix = True
        self.feat_dim = fft_len // 2 +1 
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len 
        self.win_type = win_type 
        
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='real', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='real', fix=fix)

        self.stft.cuda()
        self.istft.cuda()
        self.xi_map = map_selector('NormalCDF', None)
        xi_prior = scipy.io.loadmat('util/xi_prior_PF.mat')
        mu_xi = xi_prior['mu_xi']
        std_xi = xi_prior['std_xi']
        self.xi_map.mu = torch.tensor(mu_xi).cuda()
        self.xi_map.sigma = torch.tensor(std_xi).cuda()
        self.phis_map = map_selector('NormalCDF', None)
        phis_prior = scipy.io.loadmat('util/phiS_prior_PF.mat')
        mu_phis = phis_prior['mu_phiS']
        std_phis = phis_prior['std_phiS']  
        self.phis_map.mu = torch.tensor(mu_phis).cuda()
        self.phis_map.sigma = torch.tensor(std_phis).cuda()
        self.randomSNR = args.randomSNR
        self.snr_levels = list([0.0, 5.0, 10.0, 15.0]) 
        # self.snr_levels = list([-2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]) 
        self.gainweights = args.gainweights
        
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from)
            self.model.load_state_dict(cont['model_state_dict'])
            try:
                self.start_epoch = cont['epoch']+1
            except:
                self.start_epoch = 1
                print("Error on loading `last epoch`")

            try:
                self.optimizer.load_state_dict(cont['optimizer_state'])
            except:
                print("Error on loading optimizer")

            if self.scheduler_name == 'ReduceLROnPlateau':
                self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(optimizer, *args.scheduler_params, verbose=True)
            elif self.scheduler_name == 'No':
                breakpoint()
                try:
                    self.scheduler = CosineAnnealingWarmUpRestarts(optimizer, *args.scheduler_params, last_epoch=cont['epoch']-1)   
                except:
                    self.scheduler = CosineAnnealingWarmUpRestarts(optimizer, *args.scheduler_params)
            elif self.scheduler_name == 'CosineAnnealingWarmUpRestarts':
                self.scheduler = CosineAnnealingWarmUpRestarts(optimizer, *args.scheduler_params, last_epoch=cont['epoch']-1)   
            else:
                self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(optimizer, *args.scheduler_params,
                                                                                        last_epoch=cont['epoch']-1, verbose=True)
                breakpoint()
            try:
                self.scheduler.load_state_dict(cont['scheduler_state'])
            except:
                print('Error on loading scheduler')

            try:
                random.setstate(cont['random_state'])
                np.random.set_state(cont['nrandom_state'])
                torch.set_rng_state(cont['trandom_state'])
            except:
                print("Error on loading some random state")
        else:
            print(self.scheduler_name)
            if self.scheduler_name == 'CosineAnnealingWarmUpRestarts':
                self.scheduler = CosineAnnealingWarmUpRestarts(optimizer, *args.scheduler_params)
            elif self.scheduler_name == 'No':
                self.scheduler = CosineAnnealingWarmUpRestarts(optimizer, *args.scheduler_params)
            else:
                self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(optimizer, *args.scheduler_params, verbose=True)
            self.start_epoch = 1
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_val_loss = float("inf")
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs+1):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            print("Training...")
            print(time.ctime(start))

            tr_loss, tr_loss_1, tr_loss_2, tr_loss_3 = self._run_one_epoch(epoch)

            print('=' * 85)
            print('Train Summary | End of Epoch {0:5d} | Time {1:.2f}s'.format(
                      epoch, time.time() - start))

       
            print('-'*70)
            print('  1 spkr  | Average Loss: {0: 8.5f} '.format(
                    tr_loss))
              
            print('=' * 85)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                cv_loss, cv_loss_1, cv_loss_2, cv_loss_3 = self._run_one_epoch(epoch, cross_valid=True)

            val_loss = []
            print('='*85)
            print('Valid Summary | End of Epoch {0:5d} | Time {1:.2f}s'.format(
                      epoch, time.time() - start))
            print('-'*70)

       
           
            val_loss.append(cv_loss)    
            print('  1 spkr  | Average Loss: {0: 8.5f} '.format(
                cv_loss))

            print('-'*70)

            val_loss = sum(val_loss)


            if self.scheduler_name == "ReduceLROnPlateau":
                self.scheduler.step(val_loss)
            elif self.scheduler_name == 'No':
                print('scheduler=No')
                pass
            else:
                self.scheduler.step()


            rand_state = random.getstate()
            nrand_state = np.random.get_state()
            trand_state = torch.get_rng_state()
            # log
            wandb.log({"Training loss": tr_loss})
            wandb.log({"Validation loss": val_loss})

            wandb.log({"Training loss1": tr_loss_1})
            wandb.log({"Training loss2": tr_loss_2})
            wandb.log({"Training loss3": tr_loss_3})
            wandb.log({"Validation loss1": cv_loss_1})
            wandb.log({"Validation loss2": cv_loss_2})
            wandb.log({"Validation loss3": cv_loss_3})

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'random_state': rand_state,
                    'nrandom_state': nrand_state,
                    'trandom_state': trand_state}, file_path)
                print('Saving checkpoint model to %s' % file_path)


            if val_loss >= self.best_val_loss:
                self.val_no_impv += 1
          
                if self.val_no_impv >= self.early_stop and self.early_stop:
                    print("No imporvement for {} epochs, early stopping.".format(self.early_stop))
                    break
            else:
                self.val_no_impv = 0

            # Save the best model
            self.cv_loss.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(
                    self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'random_state': rand_state,
                    'nrandom_state': nrand_state,
                    'trandom_state': trand_state}, best_file_path)
                print("Find better validated model, saving to %s" % best_file_path)

            print('=' * 85)

  


    def _run_one_epoch(self, epoch, cross_valid=False):
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        # torch.autograd.set_detect_anomaly(True)
        BCE_loss = nn.BCELoss()
        MSE_loss = nn.MSELoss()

        total_loss = 0
        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
   
      
        start = time.time()
        for i, data in enumerate(data_loader):
            mix_batch, src_batch, noise_batch, pad_mask = data #, 

            if self.randomSNR and not cross_valid:
                B, Mt, Tt = mix_batch.size()  
                batch_wavlength = torch.sum(pad_mask,dim=2).int().tolist()
                noise_batch_new = torch.zeros([B,Mt,Tt])
                for b in range(B):
                    flag = True
                    while flag:
                        radnomb = random.choice(list(range(B)))
                        if batch_wavlength[radnomb] < batch_wavlength[b]: 
                            flag = True
                        else: 
                            flag = False
                    randint = np.random.randint(0, 1+batch_wavlength[radnomb][0]- batch_wavlength[b][0])  
                    noise_batch_new[b,:,:batch_wavlength[b][0]] = noise_batch[radnomb,:,randint:randint+batch_wavlength[b][0]]
        			
                
                src_batch_sum = torch.unsqueeze(torch.sum(src_batch,1),1)
                noise_batch_new_sum = torch.unsqueeze(torch.sum(noise_batch_new,1),1)
                norm_src= torch.mean(torch.pow(src_batch_sum,2),dim=2)
                norm_noise = torch.mean(torch.pow(noise_batch_new_sum,2),dim=2)
                snr_batch = torch.unsqueeze(torch.tensor(random.choices(self.snr_levels, k=B)), 1)
                gaindB = torch.pow(10, 0.1*snr_batch)
                mixgain_batch  =torch.sqrt(torch.div(torch.div(norm_src,norm_noise), gaindB)) ; 
                noise_batch = torch.mul(noise_batch_new, torch.unsqueeze(mixgain_batch,2).repeat(1,Mt,Tt))
                mix_batch = src_batch + noise_batch

            
            if self.use_cuda:
                
                mix_batch = mix_batch.cuda()    # [B, C, T]
                src_batch = src_batch.cuda()    # [B, C, T]
                noise_batch = noise_batch.cuda()    # [B, C, T]
                pad_mask = pad_mask.cuda()      # [B, 1, T]
    

            mix_STFT_mag, mix_STFT_pha = self.stft(mix_batch)
            src_STFT_mag, src_STFT_pha = self.stft(src_batch)
            noise_STFT_mag, noise_STFT_pha = self.stft(noise_batch)   
            
            B, F, L = mix_STFT_mag.size()  
            pad_frames = torch.ceil((torch.sum(pad_mask[:,0,:],1))/256)
            pad_mag = torch.zeros([B,F,L]).cuda()
   
            for batch_idx in range(B):
                pad_mag[batch_idx, : , :pad_frames[batch_idx].int() ]= 1     
            
            pad_mag_bool = pad_mag.bool()   
            pow_Y = torch.pow(mix_STFT_mag,2)
            pow_S = torch.pow(src_STFT_mag,2)
            pow_N = torch.pow(noise_STFT_mag,2)     
            target_xi = torch.div(pow_S,pow_N+1e-12)
            target_xi_dB = 10*torch.log10(target_xi+1e-12)     
            target_xi_dB_mapped = self.xi_map.map(target_xi_dB.permute(0,2,1)).permute(0,2,1) #[B,F,T]

            max_S, _ = torch.max(pow_S,2)
            max_S2 = max_S.repeat(L,1,1).permute(1,2,0)
            target_IBM = torch.greater(torch.div(pow_S, pow_N+1e-12), 10**(-12/10)).type(torch.float32)  #[B,F,T]

            target_phiS_dB = 10*torch.log10(pow_S+1e-12)
            target_phiS_dB_mapped = self.phis_map.map(target_phiS_dB.permute(0,2,1)).permute(0,2,1) #[B,F,T]

            mix_STFT_mag = mix_STFT_mag.permute(0,2,1).contiguous()  #[B,F,T]->[B,T,F]
            dnn_output, dnn_output2, dnn_output3= self.model(mix_STFT_mag) 

            xi_mapped =  dnn_output.permute(0,2,1).contiguous()  #[B,T,F]->[B,F,T]
            loss1 = self.gainweights[0] * BCE_loss(torch.masked_select(xi_mapped,pad_mag_bool), torch.masked_select(target_xi_dB_mapped.detach(),pad_mag_bool))  
     
            ibm_mapped  = dnn_output2.permute(0,2,1).contiguous()           
            loss2 = self.gainweights[2] * BCE_loss(torch.masked_select(ibm_mapped,pad_mag_bool), torch.masked_select(target_IBM.detach(),pad_mag_bool))

            phis_mapped = dnn_output3.permute(0,2,1).contiguous()             
            loss3 = self.gainweights[1] * BCE_loss(torch.masked_select(phis_mapped,pad_mag_bool), torch.masked_select(target_phiS_dB_mapped.detach(),pad_mag_bool))
            
            loss = loss1 + loss2 + loss3 

            total_loss += loss.item()
            loss_1 += loss1.item()
            loss_2 += loss2.item()
            loss_3 += loss3.item()  

            
            if not cross_valid:

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_norm > 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                else:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
                self.optimizer.step()
  
            num_sample = (i+1) * B    
            if i % 1 == 1000:

                print('Epoch {0:3d} | Iter {1:5d} | {2:7.1f} ms/sample'.format(
                          epoch, i + 1, 1000 * (time.time() - start) / ((i + 1)*B)))
               
                print('-'*70)
                print('  Average Loss: {}, '.format(
                        total_loss/num_sample))                
     

                print('-'*85, flush=True)
            
     
        total_loss /= num_sample
        loss_1 /= num_sample
        loss_2 /= num_sample
        loss_3 /= num_sample
        del loss
        torch.cuda.empty_cache()

        return total_loss, loss_1, loss_2, loss_3
