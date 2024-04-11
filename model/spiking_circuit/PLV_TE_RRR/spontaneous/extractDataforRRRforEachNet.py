#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 21:08:02 2022

@author: Shencong Ni
"""


import mydata
import numpy as np
#import brian2.numpy_ as np
#from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
#import frequency_analysis as fqa
import get_onoff_cpts
#import fano_mean_match
#import find_change_pts
import connection as cn
import coordination as cd

import sys
#import os
import matplotlib.pyplot as plt
#import shutil

#from scipy import optimize
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
        list consists of arrays, each array contains the index of a group of neurons for MUA calculation 
    sample_interval: ms
        sample interval between two adjacent MUA calculation points
    window: ms
        length of window to count spike for MUA
    dt: ms
        simulation timestep
    (analysis duration which is shorter than 'window' will be ignored automatically, i.e., return np.array([]) to 'spk_')
    '''
    #dt_ = int(round(1/dt))
    # spk_mat = spk_mat.tocsc()
    
    #MUA = np.zeros([len(record_neugroup), len(dura)])
    
    spk = []
    for neu_i, neu_ in enumerate(record_neugroup):
        #for stim_i, stim_dura in enumerate(dura):
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
    
    # bothON = np.logical_and(sens_on, asso_on)         
    # _, _, onset_t_bothON, offset_t_bothON = get_onoff_cpts.get_onoff_cpts(bothON)
    # onset_t_bothON, offset_t_bothON = supply_onoffPoints(onset_t_bothON, offset_t_bothON, bothON.shape[0])
    
    # _, _, onset_t_sensON, offset_t_sensON = get_onoff_cpts.get_onoff_cpts(sens_on)
    # onset_t_sensON, offset_t_sensON = supply_onoffPoints(onset_t_sensON, offset_t_sensON, sens_on.shape[0])
    
    # _, _, onset_t_assoON, offset_t_assoON = get_onoff_cpts.get_onoff_cpts(asso_on)
    # onset_t_assoON, offset_t_assoON = supply_onoffPoints(onset_t_assoON, offset_t_assoON, asso_on.shape[0])
    
    
    # sample_dura = []
    # for on_t, of_t in zip(onset_t_bothON, offset_t_bothON):
    #     if of_t - on_t >= overLap_min: # 20 ms threshold
    #         if np.any(on_t == onset_t_sensON):
    #             begin = find_closestEventBef(onset_t_assoON, on_t)
    #         else:
    #             begin = find_closestEventBef(onset_t_sensON, on_t)
            
    #         if np.any(of_t == offset_t_sensON):
    #             end = find_closestEventAft(offset_t_assoON, of_t)
    #         else:
    #             end = find_closestEventAft(offset_t_sensON, of_t)
            
    #         sample_dura.append([begin, end])
    
    # sample_dura = np.array(sample_dura)    
    # #%
    
    # sample = np.zeros(bothON.shape[0]+10, bool) # supply 10 ms, which is equal to the length of window for counting spikes in ON-OFF detection
    
    # for tt in sample_dura:
    #     sample[tt[0]:tt[1]+10] = True # supply 10 ms which is equal to the length of window for counting spikes in ON-OFF detection
    
    # if sync:
    #     _, _, onset_t_sample, offset_t_sample = get_onoff_cpts.get_onoff_cpts(sample)    
    # else:
    #     _, _, onset_t_sample, offset_t_sample = get_onoff_cpts.get_onoff_cpts(np.logical_not(sample))

    # onset_t_sample, offset_t_sample = supply_onoffPoints(onset_t_sample, offset_t_sample, sample.shape[0])
    # if offset_t_sample[-1] == sample.shape[0]:
    #     offset_t_sample[-1] -= 1
        
    # return onset_t_sample, offset_t_sample


    if sync:
        # on_sync = np.logical_and(sens_on, asso_on)        
        # on_t, off_t, onset_t, offset_t = get_onoff_cpts.get_onoff_cpts(on_sync); 
        # offset_t += 10 # ms; supply the length of window to count MUA during ON-OFF detection

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
'''spontaneous; long duration'''

data_dir = 'raw_data/'
datapath =  data_dir

dataAnaly_dir = 'raw_data/' # 'raw_data/cor/'
dataAnaly_path = dataAnaly_dir

e_lattice = cn.coordination.makelattice(64, 64, [0,0])

data = mydata.mydata()
data_anly = mydata.mydata()

sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num


unsyncType = 'bothOff' #'eitherOff'

name_sfx = ''
substract_mean = True
mua_range = 5
for spk_posi in ['ctr']:  # 'ctr', 'cor'
    for sfx in ['ctr']:
        for analy_Sync in [0, 1]:
            print('spk_posi:',spk_posi, 'sfx:',sfx, 'analy_Sync:',analy_Sync)
    # spk_posi = 'cor'
    # mua_range = 8
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
            #%

            
            MUA_1_all = []
            MUA_2_all = []
            # MUA_1_all_net = []
            # MUA_2_all_net = []
            
            #n_neu_perGroup = np.array([len(record_neugroup[i]) for i in range(len(record_neugroup))])
            
            
            # for loop_num in range(20):
            print(loop_num)
            # #data = mydata.mydata()
            # data.load(datapath+'data%d.file'%loop_num)
            # #data_anly = mydata.mydata()
            # data_anly.load(dataAnaly_path+'data_anly_onoff_testthres_%s%d.file'%(sfx,loop_num)) # data_anly_onoff_thres_cor; data_anly_onoff_thres
            # #data_anly.load(dataAnaly_path+'data_anly_onoff_thres%s%d.file'%(sfx,loop_num)) # data_anly_onoff_thres_cor; data_anly_onoff_thres
            
            data.load(datapath+'data%d.file'%loop_num)
            data_anly.load(dataAnaly_path+'data_anly_onoff_testthres_win10_min10_smt1_mtd1_%s_%d.file'%(sfx,loop_num))
            # n_perStimAmp = data.a1.param.stim1.n_perStimAmp
            # n_StimAmp = data.a1.param.stim1.n_StimAmp
            
            # dt = 1/10000;
            # end = int(20/dt); start = int(5/dt)
            # spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/15/data.a1.param.Ne
            # spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne
            
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
            
            # spon_start = 5000; spon_end = 20000
            # spk_matrix_spon_1 = data.a1.ge.spk_matrix[neu, spon_start*10:spon_end*10].copy()
            # spk_matrix_spon_2 = data.a2.ge.spk_matrix[neu, spon_start*10:spon_end*10].copy()
            analy_dura = np.array([5000, 205000])
            ignTrans = 0 #data_anly.onoff_sens.ignore_respTransient; print(ignTrans)
            
            # analy_dura = np.array([[5000, 205000]])
            # for st_ind, t in enumerate(analy_dura):
                
            #     spk_matrix_MUA_1 = data.a1.ge.spk_matrix[:, (t[0]+ignTrans)*10:t[1]*10].copy()
            #     spk_matrix_MUA_2 = data.a2.ge.spk_matrix[:, (t[0]+ignTrans)*10:t[1]*10].copy()
        
            #     if analy_Sync:            
            #         on_sync = np.logical_and(data_anly.onoff_sens.spon.onoff_bool[0], data_anly.onoff_asso.spon.onoff_bool[0])
            #         print(on_sync.shape)
            #         on_t, off_t, onset_t, offset_t = get_onoff_cpts.get_onoff_cpts(on_sync); 
            #         offset_t += 10 # ms; supply the length of window to count MUA during ON-OFF detection
            #     else:
            #         if unsyncType == 'eitherOff':
            #             ON = np.logical_and(data_anly.onoff_sens.spon.onoff_bool[0], data_anly.onoff_asso.spon.onoff_bool[0])
            #         else:    
            #             ON = np.logical_or(data_anly.onoff_sens.spon.onoff_bool[0], data_anly.onoff_asso.spon.onoff_bool[0])
        
            #         on_t, off_t, onset_t, offset_t = get_onoff_cpts.get_onoff_cpts(ON); #print(offset_t)
            #         extend_t = 10 # ms; supply the length of window to count MUA during ON-OFF detection
            #         for oft in offset_t:
            #             ON[oft:oft+extend_t] = True
            
            #         on_t, off_t, onset_t, offset_t = get_onoff_cpts.get_onoff_cpts(np.logical_not(ON))



            # for st_ind, t in enumerate(analy_dura): # analy_dura[0].reshape(1,-1)
                
            spk_matrix_MUA_1 = data.a1.ge.spk_matrix[:, (analy_dura[0]+ignTrans)*10:analy_dura[1]*10].copy()
            spk_matrix_MUA_2 = data.a2.ge.spk_matrix[:, (analy_dura[0]+ignTrans)*10:analy_dura[1]*10].copy()
            
            onset_t_sample, offset_t_sample = find_sampleDuraSyncUnsync(data_anly.onoff_sens.spon.onoff_bool[0].copy(),\
                                                                data_anly.onoff_asso.spon.onoff_bool[0].copy(), sync = analy_Sync, unsyncType = unsyncType)  
        
            """
            if loop_num == 0:
                
                t_s = 1000; t_e = 4000
                t_plt = np.s_[t_s:t_e]
                t_plt_s = np.s_[t_s*10:t_e*10]    
                t_plt_x = np.arange(t_s, t_e) 
                
                'on_t_, off_t_, onset_t_, offset_t_ = get_onoff_cpts.get_onoff_cpts(ON)'
                
                onset_t_ = onset_t_sample[(onset_t_sample >= t_s) & (onset_t_sample <= t_e)]
                offset_t_ = offset_t_sample[(offset_t_sample >= t_s) & (offset_t_sample <= t_e)]
                
                if offset_t_[0] < onset_t_[0]:
                    offset_t_ = np.delete(offset_t_, 0)
                if onset_t_[-1] > offset_t_[-1]:
                    onset_t_ = np.delete(onset_t_, -1)
                

                
                fig, ax = plt.subplots(2,1, figsize=[14,8])
                #ax[0].plot(t_plt_x, ON[t_plt]*80)
                ni,tt = spk_matrix_MUA_1[neu_id,t_plt_s].nonzero()
                ax[0].plot(tt*0.1 + t_s, ni, '|')
                #ax[0].plot(t_plt_x,  data_anly.onoff_sens.stim_att[0].onoff_bool[st_ind][t_plt]*80)
                ni,tt = spk_matrix_MUA_2[neu_id,t_plt_s].nonzero()
                ax[1].plot(tt*0.1 + t_s, ni, '|')   
                #ax[1].plot(t_plt_x, data_anly.onoff_asso.stim_att[0].onoff_bool[st_ind][t_plt]*80)
                for ont, offt in zip(onset_t_, offset_t_):
                    ax[0].axvline(ont, c=clr[0]); ax[0].axvline(offt, c=clr[1])
                    ax[1].axvline(ont, c=clr[0]); ax[1].axvline(offt, c=clr[1])
            """
                # ##
                # extend_t = 20
                # for ont in onset_t:
                #     if ont - extend_t < 0:
                #         on_sync[0:ont] = True
                #     else:
                #         on_sync[ont-extend_t:ont] = True
                # for oft in offset_t:
                #     on_sync[oft:oft+extend_t] = True
                # ##
                
                # # if analy_unSync:
                # #     on_t, off_t, onset_t, offset_t = get_onoff_cpts.get_onoff_cpts(np.logical_not(on_sync))
                # # else:
                # on_t, off_t, onset_t, offset_t = get_onoff_cpts.get_onoff_cpts(on_sync)
                
                # if st_ind == 1:
                #     fig, ax = plt.subplots(2,1)
                #     ax[0].plot(on_sync[:1000]*80)
                #     ni,tt = spk_matrix_MUA_1[neu_id,:10000].nonzero()
                #     ax[0].plot(tt*0.1, ni, '|')
                #     ax[0].plot(data_anly.onoff_sens.stim_att[0].onoff_bool[st_ind][:1000]*80)
                #     ni,tt = spk_matrix_MUA_2[neu_id,:10000].nonzero()
                #     ax[1].plot(tt*0.1, ni, '|')   
                #     ax[1].plot(data_anly.onoff_asso.stim_att[0].onoff_bool[st_ind][:1000]*80)
            
                # onset_t = data_anly.onoff_sens.spon.onset_t[0].copy()
                # offset_t = data_anly.onoff_sens.spon.offset_t[0].copy()
            """
            if offset_t[0] < onset_t[0]:
                offset_t = np.delete(offset_t, 0)
            if onset_t[-1] > offset_t[-1]:
                onset_t = np.delete(onset_t, -1)
               
            #print(data_anly.onoff_sens.spon.onset_t[0]) 
            #print(data_anly.onoff_sens.spon.offset_t[0])
            # print(onset_t[0]<offset_t[0]) 
            # print(onset_t[-1]<offset_t[-1])
            # print(len(onset_t), len(offset_t))
            
            
            '''onset and offset time points'''
            sync_t = np.vstack((onset_t, offset_t)).T
            #onoff_t_spon2 = np.vstack((data_anly.onoff_asso.spon.onset_t[0], data_anly.onoff_asso.spon.offset_t[0])).T
            """
            '''onset and offset time points'''
            sample_t = np.vstack((onset_t_sample, offset_t_sample)).T
        
            #%
            MUA_1 = multiMUA_unequalDura(spk_matrix_MUA_1, sample_t, record_neugroup, sample_interval=20, window=20, dt=0.1) #, returnHz=True)
            MUA_2 = multiMUA_unequalDura(spk_matrix_MUA_2, sample_t, record_neugroup, sample_interval=20, window=20, dt=0.1) #, returnHz=True)
            
            MUA_1_all.append(MUA_1)
            MUA_2_all.append(MUA_2)
                
                
        
            
            MUA_1_all_ = np.hstack(MUA_1_all)
            if substract_mean:
                MUA_1_all_ = MUA_1_all_ - MUA_1_all_.mean(1).reshape(-1,1)  
            # MUA_1_all_net.append(MUA_1_all_.T)
            
            MUA_2_all_ = np.hstack(MUA_2_all)  
            if substract_mean:
                MUA_2_all_ = MUA_2_all_ - MUA_2_all_.mean(1).reshape(-1,1)  
            # MUA_2_all_net.append(MUA_2_all_.T)
            data_anly.MUA_1_all = MUA_1_all_
            data_anly.MUA_2_all = MUA_2_all_                
                # MUA_1_all = []
                # MUA_2_all = []
            #%
            # MUA_1_all_net = np.stack(MUA_1_all_net, 2)
            # MUA_2_all_net = np.stack(MUA_2_all_net, 2)
                
            #%
            # MUA_1_all_net_ = np.array(MUA_1_all_net, dtype=object)
            # MUA_2_all_net_ = np.array(MUA_2_all_net, dtype=object)
            
            # MUA_1_all_net_ = np.empty(len(MUA_1_all_net), dtype=object)
            # MUA_2_all_net_ = np.empty(len(MUA_2_all_net), dtype=object)
            # for ind in range(len(MUA_1_all_net_)):
            #     MUA_1_all_net_[ind] = MUA_1_all_net[ind]
            # for ind in range(len(MUA_2_all_net_)):
            #     MUA_2_all_net_[ind] = MUA_2_all_net[ind]
            
                #%

            if analy_Sync: sync_n = 'sync'
            else: 
                if unsyncType == 'eitherOff':
                    sync_n = 'unsyncEtr'
                else:
                    sync_n = 'unsync'
                    

            # pyfile_name = 'rg%d_%ssua_%s_local_%s%s_%d.file'%(mua_range,spk_posi,sync_n, sfx, name_sfx, loop_num)
            pyfile_name = 'rg%d_%ssua_%s_local_subM%d_%d.file'%(mua_range,spk_posi,sync_n,  substract_mean, loop_num)
            
            data_anly.save(data_anly.class2dict(), dataAnaly_path+pyfile_name)

#%%

