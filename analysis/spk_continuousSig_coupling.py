#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:04:18 2023

@author: shni2598
"""



import frequency_analysis as fqa
import numpy as np
import mydata
#%%
def wvtphase(spk_mat, sig, analy_dura, onoff_stats_sens, onoff_stats_asso, return_ppc=False):
    
    dt_1 = 10
    dt = 0.1
    init_disgard = 200
    
    n_mua_neuron = spk_mat.shape[0]
    phase_f_on = [[] for ii in range(n_mua_neuron)]
    phase_f_off = [[] for ii in range(n_mua_neuron)]
    
    freq_range = [15,200]
    sampling_period = 0.001
    maxscale = int(np.ceil(np.log2((1/sampling_period)/freq_range[0])*10))
    minscale = int(np.floor(np.log2((1/sampling_period)/freq_range[1])*10))
    # scale = 2**(np.arange(minscale, maxscale + 1, 3)/10)
    scale = 2**(np.linspace(minscale, maxscale, 20).astype(int)/10)
    wavelet_name = 'cmor15-1'
        
        
    for tri_i, dura in enumerate(analy_dura):
        
        start_time = dura[0] + init_disgard + 5
        end_time = dura[1] - 4
        t_ext = 500
        
        sig_i = sig[start_time-t_ext:end_time+t_ext]
        
        coef, freq = fqa.mycwt(sig_i, wavelet_name, sampling_period, scale = scale,  method = 'fft', L1_norm = True)
        
        coef = coef[:,t_ext:-t_ext]
    
        onoff_bool_bothOn = np.logical_and(onoff_stats_sens.onoff_bool[tri_i], onoff_stats_asso.onoff_bool[tri_i])
        onoff_bool_off1 = np.logical_not(onoff_stats_sens.onoff_bool[tri_i])
        onoff_bool_off2 = np.logical_not(onoff_stats_asso.onoff_bool[tri_i])
        onoff_bool_bothOff = np.logical_and(onoff_bool_off1, onoff_bool_off2)
    
        spk_mat_tri = spk_mat[:, (dura[0] + init_disgard + 5)*dt_1 : (dura[1] - 5)*dt_1]
        
        #%
        for n_id in range(0, n_mua_neuron):
            spk_t = spk_mat_tri.indices[spk_mat_tri.indptr[n_id]:spk_mat_tri.indptr[n_id+1]]
            
            usemua = 0
            if usemua:
                spk_t = np.round(spk_t * dt - 0.5).astype(int)        
            else:
                spk_t = np.round(spk_t * dt).astype(int)
            
            #%
            
            spk_t_bothOn = spk_t[onoff_bool_bothOn[spk_t]]
            
            
            spk_t_bothOff = spk_t[onoff_bool_bothOff[spk_t]]
            
            #%
            
            phase_f_on_i = np.zeros([len(coef), spk_t_bothOn.shape[0]])
            phase_f_off_i = np.zeros([len(coef), spk_t_bothOff.shape[0]])
            
            for ii, coef_i in enumerate(coef): 
                phase_f_on_i[ii] = np.angle(coef_i)[spk_t_bothOn]
                phase_f_off_i[ii] = np.angle(coef_i)[spk_t_bothOff]
            
            phase_f_on[n_id].append(phase_f_on_i)
            phase_f_off[n_id].append(phase_f_off_i)
    
    #%
    indptr_on = np.zeros(n_mua_neuron + 1)
    indptr_off = np.zeros(n_mua_neuron + 1)
    len_on = 0;
    len_off = 0;
    for n_id in range(0, n_mua_neuron):
        phase_f_on[n_id] = np.hstack(phase_f_on[n_id]).astype('float32')
        phase_f_off[n_id] = np.hstack(phase_f_off[n_id]).astype('float32')       
        len_on += phase_f_on[n_id].shape[1]
        len_off += phase_f_off[n_id].shape[1]
        indptr_on[n_id + 1] = len_on
        indptr_off[n_id + 1] = len_off
                
    plv_f_bothOn = np.zeros([n_mua_neuron, phase_f_on[0].shape[0]])
    plv_f_bothOff = np.zeros([n_mua_neuron, phase_f_on[0].shape[0]])
    
    for n_id in range(0, n_mua_neuron):
    
        plv_f_bothOn[n_id] = get_plv(phase_f_on[n_id])
        plv_f_bothOff[n_id] = get_plv(phase_f_off[n_id])
        
    if return_ppc:    
        
        ppc_f_bothOn = np.zeros([n_mua_neuron,phase_f_on[0].shape[0]])
        ppc_f_bothOff = np.zeros([n_mua_neuron,phase_f_on[0].shape[0]])
        
        for n_id in range(0, n_mua_neuron):
        
            ppc_f_bothOn[n_id] = get_ppc(phase_f_on[n_id])
            ppc_f_bothOff[n_id] = get_ppc(phase_f_off[n_id])

    phase_f_on = np.hstack(phase_f_on)
    phase_f_off = np.hstack(phase_f_off)
    
    if return_ppc:
        data = mydata.mydata()
        data.phase_f_on = phase_f_on
        data.phase_f_off = phase_f_off
        data.indptr_on = indptr_on
        data.indptr_off = indptr_off        
        data.plv_f_bothOn = plv_f_bothOn
        data.plv_f_bothOff = plv_f_bothOff
        data.ppc_f_bothOn = ppc_f_bothOn
        data.ppc_f_bothOff = ppc_f_bothOff
        data.freq = freq
        return data
    else:
        data = mydata.mydata()
        data.phase_f_on = phase_f_on
        data.phase_f_off = phase_f_off
        data.indptr_on = indptr_on
        data.indptr_off = indptr_off        
        data.plv_f_bothOn = plv_f_bothOn
        data.plv_f_bothOff = plv_f_bothOff
        data.freq = freq
        return data
    

def get_ppc(d):
    
    d_len = d.shape[1]
    sum_cos = np.zeros(d.shape[0])
    for ii in range(0, d_len-1):
        sum_cos += np.cos(d[:,ii:ii+1] - d[:,ii+1:]).sum(1)
    
    return sum_cos/((d_len-1)*d_len/2)

def get_plv(d):
    return np.abs((np.exp(d*1j)).mean(1))
    


