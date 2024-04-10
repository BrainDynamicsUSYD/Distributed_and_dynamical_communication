#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:22:12 2022

@author: Shencong Ni
"""

'''
Calculate the phase locking value (PLV) between the MUA in area 1 and area 2.
Run the onoff_detection.py first before running this script.
'''

import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
from scipy.stats import sem
#import load_data_dict
import mydata
import numpy as np

import firing_rate_analysis as fra
import frequency_analysis as fqa

import connection as cn
import coordination as cd

import sys
#import os
import matplotlib.pyplot as plt
#import shutil

#from scipy import optimize

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']
#%%
import phase_sync

#%%
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num

savefile_name = 'phaseSyncmua_ctr_%d.file'%loop_num #'phaseSyncmua_spon%d.file'%loop_num
mua_loca = [0, 0];   #[0, 0]  [-32, -32]


analyStim = 0 # 1 for the condition with inputs, 0 for the spontaneous activity
surrogate = 1
sameSampleSize = 1


data_dir = 'raw_data/'
datapath = '' + data_dir
dataAnaly_dir = 'raw_data/' # 'raw_data/cor/'
dataAnaly_path = '' + dataAnaly_dir

data_onoff_name = 'data_anly_onoff_testthres_win10_min10_smt1_mtd1_ctr_'

#%%

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

sync_data = mydata.mydata()
#%%
repeat = 1

mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
passband = np.arange(30, 121, 10)
passband = np.array([passband - 5, passband + 5]).T

if analyStim:
    n_StimAmp = data.a1.param.stim1.n_StimAmp
    n_perStimAmp = data.a1.param.stim1.n_perStimAmp
    stim_amp = data.a1.param.stim1.stim_amp[0] #[400]
    
    #%%
    '''no attention/uncued'''
    
    cohe_ON_noatt_reliz = np.zeros([repeat, len(passband)]) # coherence during On states (not presented in the paper)
    phsLockInd_ON_noatt_reliz  = np.zeros([repeat, len(passband)]) # PLV during On states
    cohe_OFF_noatt_reliz =  np.zeros([repeat, len(passband)]) # coherence during Off states (not presented in the paper)
    phsLockInd_OFF_noatt_reliz = np.zeros([repeat, len(passband)]) # PLV during Off states
    
    if surrogate:
        cohe_ON_p_noatt_reliz = np.zeros([repeat, len(passband), 200]) # coherence during On states for the surrogate data
        phsLockInd_ON_p_noatt_reliz = np.zeros([repeat, len(passband), 200]) # PLV during On states for the surrogate data
        cohe_OFF_p_noatt_reliz = np.zeros([repeat, len(passband), 200]) # coherence during Off states for the surrogate data
        phsLockInd_OFF_p_noatt_reliz = np.zeros([repeat, len(passband), 200]) # PLV during Off states for the surrogate data
    #%
                    
    print(loop_num)
    data.load(datapath+'data%d.file'%loop_num)
    data_anly.load(dataAnaly_path+'%s%d.file'%(data_onoff_name, loop_num)) # 
    

    simu_time_tot = data.param.simutime
    if hasattr(data.a1.ge, 'spk_matrix'):
        data.a1.ge.get_spk_it()
        data.a2.ge.get_spk_it()
        data.a1.ge.spk_matrix = data.a1.ge.spk_matrix.tocsc()
        data.a2.ge.spk_matrix = data.a2.ge.spk_matrix.tocsc()
    
    elif hasattr(data.a1.ge, 't_ind'):
        data.a1.ge.get_sparse_spk_matrix_csrindptr([data.a1.param.Ne, simu_time_tot*10], mat_type='csc')
        data.a2.ge.get_sparse_spk_matrix_csrindptr([data.a2.param.Ne, simu_time_tot*10], mat_type='csc')
        data.a1.ge.get_spk_it()
        data.a2.ge.get_spk_it()
    
    else:    
        data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10], 'csc')
        data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10], 'csc')
    
    spk_mat_MUA_1 = data.a1.ge.spk_matrix[mua_neuron].copy()
    spk_mat_MUA_2 = data.a2.ge.spk_matrix[mua_neuron].copy()
    
    analy_dura = data.a1.param.stim1.stim_on[:n_perStimAmp].copy()
    ignTrans = data_anly.onoff_sens.ignore_respTransient; print(ignTrans)

    if surrogate:
        cohe_ON_noatt_reliz[0] ,phsLockInd_ON_noatt_reliz[0], _, cohe_OFF_noatt_reliz[0], phsLockInd_OFF_noatt_reliz[0], _,\
        cohe_ON_p_noatt_reliz[0], phsLockInd_ON_p_noatt_reliz[0], _, cohe_OFF_p_noatt_reliz[0], phsLockInd_OFF_p_noatt_reliz[0], _  = phase_sync.get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, analy_dura, \
                           data_anly.onoff_sens.stim_noatt[0].onoff_bool, data_anly.onoff_asso.stim_noatt[0].onoff_bool, \
                           passband=passband, ignTrans=ignTrans, \
                           MUA_window = 1, sample_interval = 1, sameSampleSize = sameSampleSize, surrogate = surrogate)
    else:
        cohe_ON_noatt_reliz[0] ,phsLockInd_ON_noatt_reliz[0], _, cohe_OFF_noatt_reliz[0], phsLockInd_OFF_noatt_reliz[0], _  = phase_sync.get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, analy_dura, \
                           data_anly.onoff_sens.stim_noatt[0].onoff_bool, data_anly.onoff_asso.stim_noatt[0].onoff_bool, \
                           passband=passband, ignTrans=ignTrans, \
                           MUA_window = 1, sample_interval = 1, sameSampleSize = sameSampleSize, surrogate = surrogate)
        
    sync_data.cohe_ON_noatt_reliz = cohe_ON_noatt_reliz
    sync_data.phsLockInd_ON_noatt_reliz = phsLockInd_ON_noatt_reliz
    sync_data.cohe_OFF_noatt_reliz = cohe_OFF_noatt_reliz
    sync_data.phsLockInd_OFF_noatt_reliz = phsLockInd_OFF_noatt_reliz
    if surrogate:
        sync_data.cohe_ON_p_noatt_reliz = cohe_ON_p_noatt_reliz
        sync_data.phsLockInd_ON_p_noatt_reliz = phsLockInd_ON_p_noatt_reliz
        sync_data.cohe_OFF_p_noatt_reliz = cohe_OFF_p_noatt_reliz
        sync_data.phsLockInd_OFF_p_noatt_reliz = phsLockInd_OFF_p_noatt_reliz
    #%%
    '''attention/cued'''
    cohe_ON_att_reliz  = np.zeros([repeat, len(passband)]) #[None for _ in range(repeat)]
    phsLockInd_ON_att_reliz  =  np.zeros([repeat, len(passband)]) #[None for _ in range(repeat)]
    cohe_OFF_att_reliz =  np.zeros([repeat, len(passband)]) #[None for _ in range(repeat)]
    phsLockInd_OFF_att_reliz = np.zeros([repeat, len(passband)]) #[None for _ in range(repeat)]
    if surrogate:
        cohe_ON_p_att_reliz = np.zeros([repeat, len(passband), 200]) # [None for _ in range(repeat)]
        phsLockInd_ON_p_att_reliz = np.zeros([repeat, len(passband), 200]) #[None for _ in range(repeat)] 
        cohe_OFF_p_att_reliz = np.zeros([repeat, len(passband), 200])  #[None for _ in range(repeat)]
        phsLockInd_OFF_p_att_reliz = np.zeros([repeat, len(passband), 200])  #[None for _ in range(repeat)]
                    
    
    analy_dura = data.a1.param.stim1.stim_on[n_perStimAmp:].copy()
    ignTrans = data_anly.onoff_sens.ignore_respTransient; print(ignTrans)
    #%
    if surrogate:
        cohe_ON_att_reliz[0] ,phsLockInd_ON_att_reliz[0], _, cohe_OFF_att_reliz[0], phsLockInd_OFF_att_reliz[0], _, \
        cohe_ON_p_att_reliz[0], phsLockInd_ON_p_att_reliz[0], _, cohe_OFF_p_att_reliz[0], phsLockInd_OFF_p_att_reliz[0], _  = phase_sync.get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, analy_dura, \
                           data_anly.onoff_sens.stim_att[0].onoff_bool, data_anly.onoff_asso.stim_att[0].onoff_bool, \
                           passband=passband, ignTrans=ignTrans, \
                           MUA_window = 2, sample_interval = 1, sameSampleSize = sameSampleSize, surrogate = surrogate)
    else:
        cohe_ON_att_reliz[0] ,phsLockInd_ON_att_reliz[0], _, cohe_OFF_att_reliz[0], phsLockInd_OFF_att_reliz[0], _  = phase_sync.get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, analy_dura, \
                           data_anly.onoff_sens.stim_att[0].onoff_bool, data_anly.onoff_asso.stim_att[0].onoff_bool, \
                           passband=passband, ignTrans=ignTrans, \
                           MUA_window = 2, sample_interval = 1, sameSampleSize = sameSampleSize, surrogate = surrogate)
        
    sync_data.cohe_ON_att_reliz = cohe_ON_att_reliz
    sync_data.phsLockInd_ON_att_reliz = phsLockInd_ON_att_reliz
    sync_data.cohe_OFF_att_reliz = cohe_OFF_att_reliz
    sync_data.phsLockInd_OFF_att_reliz = phsLockInd_OFF_att_reliz
    if surrogate:
        sync_data.cohe_ON_p_att_reliz = cohe_ON_p_att_reliz
        sync_data.phsLockInd_ON_p_att_reliz = phsLockInd_ON_p_att_reliz
        sync_data.cohe_OFF_p_att_reliz = cohe_OFF_p_att_reliz
        sync_data.phsLockInd_OFF_p_att_reliz = phsLockInd_OFF_p_att_reliz
        
#%%
    
else:
    '''spontaneous'''
    cohe_ON_spon_reliz  = np.zeros([repeat, len(passband)]) #[None for _ in range(repeat)]
    phsLockInd_ON_spon_reliz  = np.zeros([repeat, len(passband)]) # [None for _ in range(repeat)]
    cohe_OFF_spon_reliz = np.zeros([repeat, len(passband)]) # [None for _ in range(repeat)]
    phsLockInd_OFF_spon_reliz = np.zeros([repeat, len(passband)]) # [None for _ in range(repeat)]
    if surrogate:
        cohe_ON_p_spon_reliz =  np.zeros([repeat, len(passband), 200]) #[None for _ in range(repeat)]
        phsLockInd_ON_p_spon_reliz = np.zeros([repeat, len(passband), 200]) #[None for _ in range(repeat)] 
        cohe_OFF_p_spon_reliz = np.zeros([repeat, len(passband), 200]) #[None for _ in range(repeat)]
        phsLockInd_OFF_p_spon_reliz = np.zeros([repeat, len(passband), 200]) #[None for _ in range(repeat)]
                    
    data.load(datapath+'data%d.file'%loop_num)
    data_anly.load(dataAnaly_path+'%s%d.file'%(data_onoff_name,loop_num)) # data_anly_onoff_thres_cor; data_anly_onoff_thres
    
    
    simu_time_tot = data.param.simutime
    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10], 'csc')
    data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10], 'csc')
    
    spk_mat_MUA_1 = data.a1.ge.spk_matrix[mua_neuron].copy()
    spk_mat_MUA_2 = data.a2.ge.spk_matrix[mua_neuron].copy()
    
    analy_dura = np.array([[5000, 205000]])
    ignTrans = 0 #data_anly.onoff_sens.ignore_respTransient; print(ignTrans)
    #%
    if surrogate:
        cohe_ON_spon_reliz[0] ,phsLockInd_ON_spon_reliz[0], _, cohe_OFF_spon_reliz[0], phsLockInd_OFF_spon_reliz[0], _, \
        cohe_ON_p_spon_reliz[0], phsLockInd_ON_p_spon_reliz[0], _, cohe_OFF_p_spon_reliz[0], phsLockInd_OFF_p_spon_reliz[0], _  = phase_sync.get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, analy_dura, \
                           data_anly.onoff_sens.spon.onoff_bool, data_anly.onoff_asso.spon.onoff_bool, \
                           passband=passband, ignTrans=ignTrans, \
                           MUA_window = 2, sample_interval = 1, sameSampleSize = sameSampleSize, surrogate = surrogate)
    else:
        cohe_ON_spon_reliz[0] ,phsLockInd_ON_spon_reliz[0], _, cohe_OFF_spon_reliz[0], phsLockInd_OFF_spon_reliz[0], _ = phase_sync.get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, analy_dura, \
                           data_anly.onoff_sens.spon.onoff_bool, data_anly.onoff_asso.spon.onoff_bool, \
                           passband=passband, ignTrans=ignTrans, \
                           MUA_window = 2, sample_interval = 1, sameSampleSize = sameSampleSize, surrogate = surrogate)
    
    sync_data.cohe_ON_spon_reliz = cohe_ON_spon_reliz
    sync_data.phsLockInd_ON_spon_reliz = phsLockInd_ON_spon_reliz
    sync_data.cohe_OFF_spon_reliz = cohe_OFF_spon_reliz
    sync_data.phsLockInd_OFF_spon_reliz = phsLockInd_OFF_spon_reliz
    if surrogate:
        sync_data.cohe_ON_p_spon_reliz = cohe_ON_p_spon_reliz
        sync_data.phsLockInd_ON_p_spon_reliz = phsLockInd_ON_p_spon_reliz
        sync_data.cohe_OFF_p_spon_reliz = cohe_OFF_p_spon_reliz
        sync_data.phsLockInd_OFF_p_spon_reliz = phsLockInd_OFF_p_spon_reliz

sync_data.passband = passband
sync_data.save(sync_data.class2dict(), dataAnaly_path+savefile_name)

#%%
