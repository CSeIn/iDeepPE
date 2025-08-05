from pesq import pesq
import numpy as np
from pystoi import stoi
import torch
import math

import torch.nn as nn
from util.oracle_test_mse import *

def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    inf
    >>> si_sdr(reference, reference * 2)
    inf
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.3704606032577304
    >>> si_sdr(reference, reference * 2 + 1)
    6.3704606032577304
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    # assert reference.dtype == np.float64, reference.dtype
    # assert estimation.dtype == np.float64, estimation.dtype

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)

def zero_mean(source, estimate_source, pad_mask):
    # mask padding position along T
    est_source = estimate_source * pad_mask

    B, C, T = est_source.size()

    # Zero-mean norm
    source_lengths = torch.sum(pad_mask, dim=2).squeeze().int()
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(est_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = est_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= pad_mask
    zero_mean_estimate *= pad_mask

    return zero_mean_target, zero_mean_estimate


def calc_pair_wise_si_snr(source, estimate_source, pad_mask):
    EPS = 1e-8
    zero_mean_target, zero_mean_estimate = zero_mean(source, estimate_source, pad_mask)

    # SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C(est), C(tar)]

    return pair_wise_si_snr

def SI_SNR(reference, estimated, pad_mask = None):
    """

    Parameters
    ----------
    reference : tensor [(B, )C, T]
        Reference signal
    estimated : tensor [(B, )C, T]
        Estimated signal
    pad_mask  : tensor [(B, )1, T]
        Indicate padded range

    Returns
    -------
    SI_SNR without PIT

    """
    EPS = 1e-12#8
    ref = reference.clone() if reference.dim() == 3 else reference.unsqueeze(1)
    est = estimated.clone() if estimated.dim() == 3 else estimated.unsqueeze(1)

    B, C, T = ref.size()
    if pad_mask == None:
        pad_mask = ref.new_ones([B, 1, T])

    est *= pad_mask
    length = pad_mask.sum(-1, keepdim=True)
    ref -= ref.sum(-1, keepdim=True) / length
    est -= est.sum(-1, keepdim=True) / length

    ref *= pad_mask
    est *= pad_mask

    alpha = (est*ref).sum(-1, keepdim=True) / ((ref**2).sum(-1, keepdim=True) + EPS)
    s = alpha*ref

    return 10*torch.log10((s**2).sum(-1) / (((s-est)**2).sum(-1) + EPS) + EPS)

def SI_SNR_pit_2ch(src, noise, bf_signal, pad_mask = None):
    loss_src_can1 = -SI_SNR(src, bf_signal[:,:,0,:], pad_mask)
    loss_src_can1 = torch.mean(loss_src_can1,1)
    loss_noise_can1 = -SI_SNR(noise, bf_signal[:,:,1,:], pad_mask)
    loss_noise_can1 = torch.mean(loss_noise_can1,1) 
    loss_sum_can1 = loss_src_can1 + loss_noise_can1
    
    loss_src_can2 = -SI_SNR(src, bf_signal[:,:,1,:], pad_mask)
    loss_src_can2 = torch.mean(loss_src_can2,1)
    loss_noise_can2 = -SI_SNR(noise, bf_signal[:,:,0,:], pad_mask)
    loss_noise_can2 = torch.mean(loss_noise_can2,1) 
    loss_sum_can2 = loss_src_can2 + loss_noise_can2   
    
    loss_selected = torch.min(loss_sum_can1, loss_sum_can2)
    return loss_selected
    
def SI_SNR_2ch(src, noise, bf_signal, pad_mask = None):
    loss_src = -SI_SNR(src, bf_signal[:,:,0,:], pad_mask)
    loss_src = torch.mean(loss_src,1)
    loss_noise = -SI_SNR(noise, bf_signal[:,:,1,:], pad_mask)
    loss_noise = torch.mean(loss_noise,1) 
    loss_sum = loss_src + loss_noise
    
    return loss_sum

MAE_loss = nn.L1Loss()
def mc_RIMAG(ref, enh, pad):
    ref_comp = mc_stft(ref)
    B, C, F, L = ref_comp.size() 

    ref_comp = ref_comp.view(B*C,F,L).contiguous()
    ref_comp_real = torch.real(ref_comp)
    ref_comp_imag = torch.imag(ref_comp)
    ref_comp_mag  = torch.sqrt(torch.pow(ref_comp_real,2)+torch.pow(ref_comp_imag,2))     
    
    enh_comp = mc_stft(enh)
    enh_comp = enh_comp.view(B*C,F,L).contiguous()
    enh_comp_real = torch.real(enh_comp)
    enh_comp_imag = torch.imag(enh_comp)
    enh_comp_mag  = torch.sqrt(torch.pow(enh_comp_real,2)+torch.pow(enh_comp_imag,2))  
    
    loss_real = MAE_loss(torch.masked_select(enh_comp_real,pad), torch.masked_select(ref_comp_real,pad))    
    loss_imag = MAE_loss(torch.masked_select(enh_comp_imag,pad), torch.masked_select(ref_comp_imag,pad))    
    loss_mag  = MAE_loss(torch.masked_select(enh_comp_mag,pad), torch.masked_select(ref_comp_mag,pad)) 

    loss_sum = loss_real + loss_imag + loss_mag
    return loss_sum

def RIMAG(ref, enh, pad, stftc):
    ref_comp = stftc(ref)
    # _, _, F, L = ref_comp.size() 
    # ref_comp = ref_comp.view(B*C,F,L).contiguous()
    ref_comp_real = torch.real(ref_comp)
    ref_comp_imag = torch.imag(ref_comp)
    ref_comp_mag  = torch.sqrt(torch.pow(ref_comp_real,2)+torch.pow(ref_comp_imag,2))     
    
    enh_comp = stftc(enh)
    # enh_comp = enh_comp.view(B*C,F,L).contiguous()
    enh_comp_real = torch.real(enh_comp)
    enh_comp_imag = torch.imag(enh_comp)
    enh_comp_mag  = torch.sqrt(torch.pow(enh_comp_real,2)+torch.pow(enh_comp_imag,2))  
    
    loss_real = MAE_loss(torch.masked_select(enh_comp_real,pad), torch.masked_select(ref_comp_real,pad))    
    loss_imag = MAE_loss(torch.masked_select(enh_comp_imag,pad), torch.masked_select(ref_comp_imag,pad))    
    loss_mag  = MAE_loss(torch.masked_select(enh_comp_mag,pad), torch.masked_select(ref_comp_mag,pad)) 

    loss_sum = loss_real + loss_imag + loss_mag
    return loss_sum

def RIMAG_in_STFT(ref_real, ref_imag, enh_real, enh_imag, pad):
    B,C,F,L = ref_real.size()
    pad = pad.view(B,C,F,L).contiguous()
    ref_mag  = torch.sqrt(torch.pow(ref_real,2)+torch.pow(ref_imag,2))   
    enh_mag  = torch.sqrt(torch.pow(enh_real,2)+torch.pow(enh_imag,2)) 
    
    loss_real = MAE_loss(torch.masked_select(enh_real,pad),  torch.masked_select(ref_real,pad))    
    loss_imag = MAE_loss(torch.masked_select(enh_imag,pad),  torch.masked_select(ref_imag,pad))    
    loss_mag  = MAE_loss(torch.masked_select(enh_mag,pad),  torch.masked_select(ref_mag,pad)) 

    loss_sum = loss_real + loss_imag + loss_mag
    return loss_sum   

def RIMAG_in_STFT_1ch(ref_real, ref_imag, enh_real, enh_imag, pad):
    B,F,L = ref_real.size()
    # pad = pad.view(B,F,L).contiguous()
    ref_mag  = torch.sqrt(torch.pow(ref_real,2)+torch.pow(ref_imag,2))   
    enh_mag  = torch.sqrt(torch.pow(enh_real,2)+torch.pow(enh_imag,2)) 
    
    loss_real = MAE_loss(torch.masked_select(enh_real,pad),  torch.masked_select(ref_real,pad))    
    loss_imag = MAE_loss(torch.masked_select(enh_imag,pad),  torch.masked_select(ref_imag,pad))    
    loss_mag  = MAE_loss(torch.masked_select(enh_mag ,pad),  torch.masked_select(ref_mag ,pad)) 

    loss_sum = loss_real + loss_imag + loss_mag
    return loss_sum   

def PESQ(reference, estimation, rate):
    return pesq(rate, reference, estimation, 'wb')

def PESQNB(reference, estimation, rate):
    return pesq(rate, reference, estimation, 'nb')

def PESQNB2(reference, estimation, rate):
    return (math.log(4.0/(pesq(rate, reference, estimation, 'nb') - 0.999) - 1.0) - 4.6607) / (-1.4945)

def eSTOI(reference, estimation, rate):
    return stoi(reference, estimation, rate, extended=True)

def STOI(reference, estimation, rate):
    return stoi(reference, estimation, rate, extended=False)