#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:41:15 2022

@author: Shencong Ni
"""

'''
Extract SUA data for each network realization
SUA will be used in reduced rank regression (RRR) analysis
Run the onoff_detection.py first before running this script.
'''
import mydata
import numpy as np
import firing_rate_analysis as fra
import get_onoff_cpts
import connection as cn
import coordination as cd

import sys

import scipy.io as sio


#%%
def supply_onoffPoints(onset, offset, datalen):
    
    if offset[0] < onset[0]:
        onset = np.hstack([[0], onset])
    if onset[-1] > offset[-1]:
        offset = np.hstack([offset, [datalen]])
        
    return onset, offset
 
def find_closestEventBef(t, t_ref):
    
    return t[t <= t_ref][-1]

def find_closestEventAft(t, t_ref):
    
    return t[t >= t_ref][0]
#%%
def multiMUA_unequalDura(spk_mat, dura, record_neugroup, sample_interval=20, window=20, dt=0.1): #, returnHz=True):
    '''
    spk_mat:
        sparse matrix for time and index of spikes
    dura: 2-D array
        each row specifies the start and end of each analysis duration; ms
        length of each duration can be unequal
    record_neugroup: list
        list consisting of arrays, each array contains the index of a group of neurons for MUA calculation 
    sample_interval: ms
        sample interval between two adjacent MUA calculation points
    window: ms
        length of window to count spike for MUA
    dt: ms
        simulation timestep
    (analysis duration which is shorter than 'window' will be ignored automatically, i.e., return np.array([]) to 'spk_')
    '''
   
    spk = []
    for neu_i, neu_ in enumerate(record_neugroup):

        spk_ = fra.get_spkcount_sparmat_multi_unequalDura(spk_mat[neu_], dura, sum_activity=True, \
                    sample_interval = sample_interval,  window = window, dt = dt)
        if len(spk_) == 0:
            spk.append(spk_)
        else:
            spk_ = np.hstack(spk_)
            spk.append(spk_)
    
    spk = np.vstack(spk)
    
    return spk


#%%
def find_sampleDuraSyncUnsync(sens_on, asso_on, sync = True, unsyncType = 'eitherOff'):
    


    if sync:

        bothON = np.logical_and(sens_on, asso_on)         
        _, _, onset_t_bothON, offset_t_bothON = get_onoff_cpts.get_onoff_cpts(bothON)
        onset_t_bothON, offset_t_bothON = supply_onoffPoints(onset_t_bothON, offset_t_bothON, bothON.shape[0])

        sample = np.hstack([bothON, np.zeros(9,dtype=bool)]) #np.zeros(bothON.shape[0]+9, bool) # supply 10 ms(+9), which is equal to the length of window for counting spikes in ON-OFF detection
        for oft in offset_t_bothON:
            sample[oft:oft+9] = True

        _, _, onset_t_sample, offset_t_sample = get_onoff_cpts.get_onoff_cpts(sample)    

        onset_t_sample, offset_t_sample = supply_onoffPoints(onset_t_sample, offset_t_sample, sample.shape[0])
        
    else:
        if unsyncType == 'eitherOff': # eitherOff 
            ON = np.logical_and(sens_on, asso_on)                                        
        else: # bothOff
            ON = np.logical_or(sens_on, asso_on)        
        
        _, _, onset_t_ON, offset_t_ON = get_onoff_cpts.get_onoff_cpts(ON)
        onset_t_ON, offset_t_ON = supply_onoffPoints(onset_t_ON, offset_t_ON, ON.shape[0])

        sample = np.hstack([ON, np.zeros(9,dtype=bool)]) #np.zeros(bothON.shape[0]+9, bool) # supply 10 ms(+9), which is equal to the length of window for counting spikes in ON-OFF detection
        for oft in offset_t_ON:
            sample[oft:oft+9] = True
        
        _, _, onset_t_sample, offset_t_sample = get_onoff_cpts.get_onoff_cpts(np.logical_not(sample))    

        onset_t_sample, offset_t_sample = supply_onoffPoints(onset_t_sample, offset_t_sample, sample.shape[0])

    return onset_t_sample, offset_t_sample

#%%
'''evoked activity; uncued/cued'''

data_dir = 'raw_data/' # path to raw data
datapath =  data_dir

dataAnaly_dir = 'raw_data/' # path to the On/Off detection results
dataAnaly_path = dataAnaly_dir


e_lattice = cn.coordination.makelattice(64, 64, [0,0])
data = mydata.mydata()
data_anly = mydata.mydata()

sys_argv = int(sys.argv[1])
loop_num = sys_argv #


name_sfx = '' # 
unsyncType = 'bothOff'  #  
mua_range = 5
substract_mean = True
for spk_posi in ['ctr']: # 'cor' ctr
    for sfx in ['ctr']:
        for get_att in [0, 1]:
            for analy_Sync in [0, 1]:
                print('spk_posi:',spk_posi, 'sfx:',sfx, 'get_att:',get_att, 'analy_Sync:',analy_Sync)

                if spk_posi == 'ctr':
                
                    elec = [0, 0]
                    neu_id = cn.findnearbyneuron.findnearbyneuron(e_lattice, elec, mua_range, 64)
                    
                    record_neugroup = []
                    for neu in neu_id:
                        record_neugroup.append(neu)
                else:
                    
                    elec = [-32, -32]
                    neu_id = cn.findnearbyneuron.findnearbyneuron(e_lattice, elec, mua_range, 64)
                    
                    record_neugroup = []
                    for neu in neu_id:
                        record_neugroup.append(neu)

                
                MUA_1_all = []
                MUA_2_all = []

                print(loop_num)
                data.load(datapath+'data%d.file'%loop_num)
                data_anly.load(dataAnaly_path+'data_anly_onoff_testthres_win10_min10_smt1_mtd1_%s_%d.file'%(sfx,loop_num)) # data_anly_onoff_thres_cor; data_anly_onoff_thres
            
                n_perStimAmp = data.a1.param.stim1.n_perStimAmp
                n_StimAmp = data.a1.param.stim1.n_StimAmp
                                
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

                
                if get_att:
                    analy_dura = data.a1.param.stim1.stim_on[n_perStimAmp:].copy()
                else:
                    analy_dura = data.a1.param.stim1.stim_on[:n_perStimAmp].copy()
                ignTrans = data_anly.onoff_sens.ignore_respTransient; print(ignTrans)
                for st_ind, t in enumerate(analy_dura): # analy_dura[0].reshape(1,-1)
                    
                    spk_matrix_MUA_1 = data.a1.ge.spk_matrix[:, (t[0]+ignTrans)*10:t[1]*10].copy()
                    spk_matrix_MUA_2 = data.a2.ge.spk_matrix[:, (t[0]+ignTrans)*10:t[1]*10].copy()
                    
                    if get_att:
                        onset_t_sample, offset_t_sample = find_sampleDuraSyncUnsync(data_anly.onoff_sens.stim_att[0].onoff_bool[st_ind].copy(),\
                                                                            data_anly.onoff_asso.stim_att[0].onoff_bool[st_ind].copy(), sync = analy_Sync, unsyncType = unsyncType)  
                    else:
                        onset_t_sample, offset_t_sample = find_sampleDuraSyncUnsync(data_anly.onoff_sens.stim_noatt[0].onoff_bool[st_ind].copy(),\
                                                                            data_anly.onoff_asso.stim_noatt[0].onoff_bool[st_ind].copy(), sync = analy_Sync, unsyncType = unsyncType)  

  
                    
                    '''onset and offset time points'''
                    sample_t = np.vstack((onset_t_sample, offset_t_sample)).T
                
                
                    MUA_1 = multiMUA_unequalDura(spk_matrix_MUA_1, sample_t, record_neugroup, sample_interval=20, window=20, dt=0.1) #, returnHz=True)
                    MUA_2 = multiMUA_unequalDura(spk_matrix_MUA_2, sample_t, record_neugroup, sample_interval=20, window=20, dt=0.1) #, returnHz=True)
                    
                    MUA_1_all.append(MUA_1)
                    MUA_2_all.append(MUA_2)
                                                        
                MUA_1_all_ = np.hstack(MUA_1_all) 
                if substract_mean:
                    MUA_1_all_ = MUA_1_all_ - MUA_1_all_.mean(1).reshape(-1,1)  
                
                MUA_2_all_ = np.hstack(MUA_2_all)
                if substract_mean:
                    MUA_2_all_ = MUA_2_all_ - MUA_2_all_.mean(1).reshape(-1,1)  
                
                data_anly.MUA_1_all = MUA_1_all_
                data_anly.MUA_2_all = MUA_2_all_

                if analy_Sync: sync_n = 'sync'
                else: 
                    if unsyncType == 'eitherOff':
                        sync_n = 'unsyncEtr'
                    else:
                        sync_n = 'unsync'
                        
                if get_att: att_n = 'att'
                else: att_n = 'noatt'

                pyfile_name = '%s_rg%d_%ssua_%s_local_subM%d_%d.file'%(att_n, mua_range,spk_posi,sync_n,  substract_mean, loop_num)
                
                data_anly.save(data_anly.class2dict(), dataAnaly_path+pyfile_name)



    