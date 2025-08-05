
import torch
import torch.nn.functional as F
from models.modules.conv_stft import ConvSTFT, ConviSTFT

# STFT / ISTFT
win_len = 512
win_inc = 256
fft_len = 512
win_type = 'hann'#'hanning'
fix = True
win_len = win_len
win_inc = win_inc
fft_len = fft_len
win_type = win_type

stft = ConvSTFT(win_len, win_inc, fft_len, win_type,
                feature_type='real', fix=fix)
istft = ConviSTFT(win_len, win_inc, fft_len, win_type,
                  feature_type='real', fix=fix)
stftc = ConvSTFT(win_len, win_inc, fft_len, win_type,
                feature_type='complex', fix=fix)
istftc = ConviSTFT(win_len, win_inc, fft_len, win_type,
                  feature_type='complex', fix=fix)
stft = stft.cuda()
istft = istft.cuda()
stftc = stftc.cuda()
istftc = istftc.cuda()
def mc_stft(batch):
    B, C, T = batch.size()   
    comp_list =[]
    # pha_list =[]
    for c in range(C):
        comp = stftc(batch[:,c,:])
        comp_list.append(torch.unsqueeze(comp,1))

    comp_out = torch.cat(comp_list, dim=1)    
    return comp_out

def mc_istft(mag, phase):
    B, C, F, T = mag.size()  # Assuming the input shape is (B, C, F, T, 2) for complex numbers
    time_list = []

    for b in range(B):
        # Perform inverse STFT on the complex input
        time_signal = istft(mag[b, :, :, :],phase[b, :, :, :])
        time_list.append(torch.unsqueeze(time_signal, 0))

    time_out = torch.cat(time_list, dim=0)
    return time_out

def STFTc(batch):
    B, C, T = batch.size()   

    comp = stftc(batch)
 
    return comp

def Hermitian(c, dim):
    if dim == 3:
        cH=torch.conj(torch.unsqueeze(c,dim))
    if dim == 4:
        cH=torch.conj(c.permute(0,1,2,4,3))
    return cH

def compute_covariance(comp):
    comp2 = comp.permute(0, 2, 3, 1).contiguous()
    comp3 = torch.unsqueeze(comp2,4) 
    comp4 = Hermitian(comp2, 3)
    comp5 = torch.matmul(comp3,comp4)  
    return comp5


def compute_covariance_spp(comp, spp):
    B, C, F, L = comp.size() 
    
    comp2 = comp.permute(0, 2, 3, 1).contiguous()
    comp3 = torch.unsqueeze(comp2,4) 
    comp4 = Hermitian(comp2, 3)
    comp5 = torch.matmul(comp3,comp4)  
    comp6 = torch.zeros(B,F,L,C,C).type(torch.cfloat).cuda()
    # av =0.90 + spp*(1-0.90)
    av =0.6 + spp*(1-0.6)
    # av = spp
    av2 = torch.unsqueeze(torch.unsqueeze(av,3).repeat(1,1,1,C),4).repeat(1,1,1,1,C)
    for l in range(L):
        if l == 0:
            comp6[:,:,l,:,:] = comp5[:,:,l,:,:]
        else:   
            comp6[:,:,l,:,:] = av2[:,:,l,:,:] * comp6[:,:,l-1,:,:] + (1-av2[:,:,l,:,:]) * comp5[:,:,l,:,:]
    return comp6

def compute_covariance_spp_bi(comp, spp):
    B, C, F, L = comp.size() 
    
    comp2 = comp.permute(0, 2, 3, 1).contiguous()
    comp3 = torch.unsqueeze(comp2,4) 
    comp4 = Hermitian(comp2, 3)
    comp5 = torch.matmul(comp3,comp4)  
    comp6 = torch.zeros(B,F,L,C,C).type(torch.cfloat).cuda()
    comp7 = torch.zeros(B,F,L,C,C).type(torch.cfloat).cuda()
    av =0.90 + spp*(1-0.90)
    # av =0.6 + spp*(1-0.6)
    # av = spp
    av2 = torch.unsqueeze(torch.unsqueeze(av,3).repeat(1,1,1,C),4).repeat(1,1,1,1,C)
    for l in range(L):
        if l == 0:
            comp6[:,:,l,:,:] = comp5[:,:,l,:,:]
            comp7[:,:,L-l-1,:,:] = comp5[:,:,L-l-1,:,:]
        else:   
            comp6[:,:,l,:,:] = av2[:,:,l,:,:] * comp6[:,:,l-1,:,:] + (1-av2[:,:,l,:,:]) * comp5[:,:,l,:,:]
            # comp6[:,:,l,:,:] = 0.9 * comp6[:,:,l-1,:,:] + 0.1 * (av2[:,:,l,:,:]) * comp5[:,:,l,:,:]
            comp7[:,:,L-l-1,:,:] = av2[:,:,L-l-1,:,:] * comp7[:,:,L-l,:,:] + (1-av2[:,:,L-l-1,:,:]) * comp5[:,:,L-l-1,:,:]
            # comp7[:,:,L-l-1,:,:] = 0.9 * comp7[:,:,L-l,:,:] + 0.1 * (av2[:,:,L-l-1,:,:]) * comp5[:,:,L-l-1,:,:]
    comp8 = (comp6+comp7)/2
    return comp8, comp5

       
def compute_noisecov(noise_cov):
    B, F, L, C, C = noise_cov.size() 
    eyem = torch.eye(C).type(torch.cfloat).cuda() 
    # cov_eye = torch.real(torch.mul(noise_cov,eyem))
    cov_eye = torch.mul(noise_cov,eyem)
    cov_eye_inv = torch.linalg.inv(cov_eye)
    return cov_eye, cov_eye_inv

def MVDR(mix_comp, noise_cov_inv, RTF):
    B, C, F, L = mix_comp.size() 
    conj_RTF = Hermitian(RTF, 3)
    RTFo = torch.unsqueeze(RTF,4)
    phi_o = 1/(torch.abs(torch.real(torch.matmul(torch.matmul(conj_RTF, noise_cov_inv), RTFo)))+1e-12)
    F = torch.matmul(torch.matmul(noise_cov_inv, RTFo), phi_o.type(torch.cfloat))
    
    mix_comp2 = torch.unsqueeze(mix_comp.permute(0, 2, 3, 1).contiguous(),4)   
    Z = torch.matmul(Hermitian(F,4), mix_comp2)
    Z = Z[:,:,:,0,0]
    return Z, phi_o
   
def MVDR_separate(mix_comp, noise_cov_inv, RTF):
    B, C, F, L = mix_comp.size() 
    conj_RTF = Hermitian(RTF, 3)
    RTFo = torch.unsqueeze(RTF,4)
    phi_o = 1/torch.abs(torch.real(torch.matmul(torch.matmul(conj_RTF, noise_cov_inv), RTFo)))
    F = torch.matmul(torch.matmul(noise_cov_inv, RTFo), phi_o.type(torch.cfloat))
    
    mix_comp2 = torch.unsqueeze(mix_comp.permute(0, 2, 3, 1).contiguous(),4)   
    # Z = torch.matmul(Hermitian(F,4), mix_comp2)
    Z = torch.squeeze(torch.mul(torch.conj(F), mix_comp2),4).permute(0,3,1,2).contiguous()
    return Z

def MVDR_separate_test(mix_comp, noise_cov_inv, RTF, src_comp, noise_comp):
    B, C, F, L = mix_comp.size() 
    conj_RTF = Hermitian(RTF, 3)
    RTFo = torch.unsqueeze(RTF,4)
    phi_o = 1/torch.abs(torch.real(torch.matmul(torch.matmul(conj_RTF, noise_cov_inv), RTFo)))
    F = torch.matmul(torch.matmul(noise_cov_inv, RTFo), phi_o.type(torch.cfloat))
    
    F2 =F.permute(0, 1, 2, 4, 3).contiguous()
    mix_comp2 = torch.unsqueeze(mix_comp.permute(0, 2, 3, 1).contiguous(),4)   
    Z = torch.squeeze(torch.matmul(torch.conj(F2), mix_comp2),4).permute(0,3,1,2).contiguous()
    
    # src_comp2 = torch.unsqueeze(src_comp.permute(0, 2, 3, 1).contiguous(),4) 
    # Zs = torch.squeeze(torch.matmul(torch.conj(F2), src_comp2),4).permute(0,3,1,2).contiguous()
    src_comp_ref = src_comp[:,0,:,:]
    conj_F2_ref =torch.conj(torch.squeeze(F2[:,:,:,:,0], dim =-1))
    Zs = torch.mul(src_comp_ref, conj_F2_ref)
    # breakpoint()
    noise_comp2 = torch.unsqueeze(noise_comp.permute(0, 2, 3, 1).contiguous(),4) 
    Zn = torch.squeeze(torch.matmul(torch.conj(F2), noise_comp2),4).permute(0,3,1,2).contiguous()
    # breakpoint()
    return Z, Zs, Zn
    
def IRM_single(mix_batch, src_batch, noise_batch):
    mix_STFT_mag, mix_STFT_pha = stft(mix_batch)
    src_STFT_mag, src_STFT_pha = stft(src_batch)
    noise_STFT_mag, noise_STFT_pha = stft(noise_batch)
    
    pow_Y = torch.pow(mix_STFT_mag, 2)
    pow_S = torch.pow(src_STFT_mag, 2)
    pow_N = torch.pow(noise_STFT_mag, 2)
    
    Wienerlike = torch.divide(pow_S, pow_S + pow_N + 1e-12) 
    estimate_mag = torch.mul(Wienerlike, mix_STFT_mag )
    estimate_source= istft(estimate_mag, mix_STFT_pha)
    
    return estimate_source

def mvdr_oracle2(mix_batch, src_batch, noise_batch, spp, RTF_est):
    
    mix_comp = mc_stft(mix_batch)
    src_comp = mc_stft(src_batch)
    noise_comp = mc_stft(noise_batch)
    B, C, F, L = mix_comp.size() 
    mix_cov = compute_covariance(mix_comp)
    src_cov = compute_covariance(src_comp)
    
    # src_ref_mag = torch.abs(src_comp[:,0,:,:])
    # src_ref_phase = torch.angle(src_comp[:,0,:,:])
    # pow_S = torch.pow(src_ref_mag,2)
    # noise_ref_mag = torch.abs(noise_comp[:,0,:,:])
    # noise_ref_phase = torch.angle(noise_comp[:,0,:,:])
    # pow_N = torch.pow(noise_ref_mag,2)
    # mix_ref_mag = torch.abs(mix_comp[:,0,:,:])
    # mix_ref_phase = torch.angle(mix_comp[:,0,:,:])
    # pow_Y = torch.pow(mix_ref_mag,2)
    # IBM = torch.greater(torch.div(pow_S, pow_N+1e-12), 10**(-15/10)).type(torch.float32) 
    
    temp_ones = torch.ones(B,L,F,1).cuda()
    RTF_est = torch.cat([temp_ones,RTF_est],3)
    RTF_est = RTF_est.permute(0,2,1,3).contiguous()
    
    # spp = spp.permute(0,2,1).contiguous()
    # noise_cov = compute_covariance_spp(mix_comp, spp)
    # noise_cov = compute_covariance_TI_spp(mix_comp, spp)
    # noise_cov = compute_covariance_sm_bi(noise_comp)
    noise_cov, Y_pow = compute_covariance_spp_bi(mix_comp, torch.pow(spp,2))
    
    cov_eye_inv = torch.linalg.inv(noise_cov+1e-5*torch.eye(6).cuda())
    Z = MVDR_separate(mix_comp, cov_eye_inv, RTF_est)
    Zs = MVDR_separate(src_comp, cov_eye_inv, RTF_est)
    Zn = MVDR_separate(noise_comp, cov_eye_inv, RTF_est)
    # Zs, _ = MVDR(src_comp, cov_eye_inv, RTF_oracle)
     
    Z = torch.squeeze(Z,0)
    Z_mag = torch.abs(Z )
    Z_phase = torch.angle(Z )  
    estimated_zz= istft(Z_mag, Z_phase)
    
    Zs = torch.squeeze(Zs,0)
    Zs_mag = torch.abs(Zs )
    Zs_phase = torch.angle(Zs )  
    estimated_zs= istft(Zs_mag, Zs_phase)

    Zn = torch.squeeze(Zn,0)
    Zn_mag = torch.abs(Zn )
    Zn_phase = torch.angle(Zn )  
    estimated_zn= istft(Zn_mag, Zn_phase)
    
    return estimated_zz, estimated_zs, estimated_zn

def eval_mvdr_part1(mix_batch, RTF_est):
    
    mix_comp = mc_stft(mix_batch)
    B, C, F, L = mix_comp.size() 
    # mix_cov = compute_covariance(mix_comp)
       
    temp_ones = torch.ones(B,L,F,1).cuda()
    RTF_est = torch.cat([temp_ones,RTF_est],3)
    RTF_est = RTF_est.permute(0,2,1,3).contiguous()
    
    return mix_comp, RTF_est
    
def eval_mvdr_part2(mix_comp, spp, RTF_est, causal):
    
    if causal:
        noise_cov = compute_covariance_spp(mix_comp, spp)
    else:
        noise_cov, Y_pow = compute_covariance_spp_bi(mix_comp, spp)

    cov_eye_inv = torch.linalg.inv(noise_cov+1e-5*torch.eye(6).cuda())
    Z = MVDR_separate(mix_comp, cov_eye_inv, RTF_est)
    
    Z = torch.squeeze(Z,0)
    Z_mag = torch.abs(Z )
    Z_phase = torch.angle(Z )  
 
    B, C, F, L = mix_comp.size() 
    if B==1:
        estimated_zz= istft(Z_mag, Z_phase)
    else:
        estimated_zz= mc_istft(Z_mag, Z_phase)
 
    return estimated_zz, noise_cov, Y_pow

def eval_mvdr_part3(estimated_zz, target_type):    
    
    estimated_zz = estimated_zz.permute(1,0,2).contiguous()
    Z_comp = mc_stft(estimated_zz)
    B, C, F, L = Z_comp.size() 
    Z_batch_sum = torch.unsqueeze(torch.sum(estimated_zz,1),1)
    Z_STFT_mag, Z_STFT_pha = stft(Z_batch_sum)           
    B, F, L = Z_STFT_mag.size()  
    # pow_Z = torch.pow(Z_STFT_mag,2)
    if target_type == 0:
        Z_comp_sum = torch.sum(Z_comp,1)
        Z_comp_sum_real = torch.real(Z_comp_sum)
        Z_comp_sum_imag = torch.imag(Z_comp_sum)
        input_batch = torch.cat([Z_comp_sum_real,Z_comp_sum_imag], dim=1) 
    elif target_type  == 1:
        input_batch = Z_STFT_mag
    elif target_type == 2:
        Z_comp_mag = torch.abs(Z_comp)
        input_batch = Z_comp_mag.view(B,C*F,L).contiguous()
    elif target_type == 3:
        Z_comp_real = torch.real(Z_comp).view(B,C*F,L).contiguous()
        Z_comp_imag = torch.imag(Z_comp).view(B,C*F,L).contiguous()
        input_batch = torch.cat([Z_comp_real,Z_comp_imag], dim=1) 
    elif target_type == 4:   
        Z_comp1 = Z_comp[:,0,:,:]
        Z_comp2 = Z_comp1 + Z_comp[:,1,:,:]
        Z_comp3 = Z_comp2 + Z_comp[:,2,:,:]
        Z_comp4 = Z_comp3 + Z_comp[:,3,:,:]
        Z_comp5 = Z_comp4 + Z_comp[:,4,:,:]
        Z_comp6 = Z_comp5 + Z_comp[:,5,:,:]
        input_batch = torch.abs(torch.cat([Z_comp1,Z_comp2,Z_comp3,Z_comp4,Z_comp5,Z_comp6], dim=1) )

    return input_batch, Z_STFT_mag, Z_STFT_pha

def map_RTF(RTF):
    #RTF : complex   
    sin_RTF = torch.imag(RTF) #-1<sin<1
    cos_RTF = torch.real(RTF) #-1<cos<1
    sin_RTF = (sin_RTF+1)/2 #0<sin<1
    cos_RTF = (cos_RTF+1)/2 #0<cos<1
    return sin_RTF, cos_RTF

def imap_RTF(sin_RTF, cos_RTF):  
    sin_RTF = sin_RTF*2 - 1 #0<sin<1
    cos_RTF = cos_RTF*2 - 1 #0<cos<1
    RTF = sin_RTF*1j + cos_RTF
    return RTF

def imap_RTF2(sin_RTF, cos_RTF):  
    sin_RTF = sin_RTF*2 - 1 #0<sin<1
    cos_RTF = cos_RTF*2 - 1 #0<cos<1
    atan2_RTF = torch.atan2(sin_RTF, cos_RTF)
    RTF = torch.exp(1j*atan2_RTF)
    return RTF


def create_input_target_cmgan(mix_batch, src_batch, noise_batch):
    
    mix_comp = mc_stft(mix_batch)
    B, C, F, L = mix_comp.size() 
    mix_cov = compute_covariance(mix_comp)
    RTF_est = torch.div(mix_cov[:,:,:,:,0],torch.unsqueeze(mix_cov[:,:,:,0,0],3).repeat(1,1,1,C)+1e-12)
    RTF_est = torch.div(RTF_est+1e-12,torch.abs(RTF_est)+1e-12) 
    sin_RTF_est, cos_RTF_est = map_RTF(RTF_est[:,:,:,1:])
    mix_ref_mag = torch.abs(mix_comp[:,0,:,:]).unsqueeze(3)
    batch_input = torch.cat([mix_ref_mag,sin_RTF_est,cos_RTF_est],3)

    batch_input = batch_input.permute(0,3,2,1).contiguous() 
    
    src_comp = mc_stft(src_batch)
    noise_comp = mc_stft(noise_batch)
    src_cov = compute_covariance(src_comp)
    RTF_oracle = torch.div(src_cov[:,:,:,:,0],torch.unsqueeze(src_cov[:,:,:,0,0],3).repeat(1,1,1,C)+1e-12)
    RTF_oracle = torch.div(RTF_oracle+1e-12,torch.abs(RTF_oracle)+1e-12) 
    sin_RTF_oracle, cos_RTF_oracle = map_RTF(RTF_oracle[:,:,:,1:])
    src_ref_mag = torch.abs(src_comp[:,0,:,:]).unsqueeze(3)
    pow_S = torch.pow(src_ref_mag,2)
    noise_ref_mag = torch.abs(noise_comp[:,0,:,:]).unsqueeze(3)
    pow_N = torch.pow(noise_ref_mag,2)
    mix_ref_mag = torch.abs(mix_comp[:,0,:,:]).unsqueeze(3)
                    
    IBM = torch.greater(torch.div(pow_S, pow_N+1e-12), 10**(-12/10)).type(torch.float32) 
    batch_target1 = torch.squeeze(IBM,3)
    batch_target1= batch_target1.permute(0,2,1).contiguous() 
    
    
    batch_target = torch.cat([sin_RTF_oracle,cos_RTF_oracle],3)
    batch_target2= batch_target.permute(0,3,2,1).contiguous() 
    pow_S = torch.squeeze(pow_S,3)
    target_phiS_dB = 10*torch.log10(pow_S+1e-12)
    target_phiS_dB= target_phiS_dB.permute(0,2,1).contiguous() 

    return batch_input,  batch_target1, batch_target2, target_phiS_dB 