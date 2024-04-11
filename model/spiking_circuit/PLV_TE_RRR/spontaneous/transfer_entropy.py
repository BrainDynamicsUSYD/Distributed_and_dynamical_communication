#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:41:08 2023

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
import os
import matplotlib.pyplot as plt
import time

#import shutil

#from scipy import optimize
import scipy.io as sio

#%%
from jpype import *

jarLocation = 'PATH_to_/infodynamics.jar'
# Path to the 'infodynamics.jar'. Replace it with your path to 'infodynamics.jar'.
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

#%%
'''spontaneous activity'''

datapath = 'raw_data/'
savedatapath = 'raw_data/te_data/'
if not os.path.exists(savedatapath):
    try: os.makedirs(savedatapath)
    except FileExistsError:
        pass
dataAnaly_dir = 'raw_data/' # 'raw_data/cor/'
dataAnaly_path = '' + dataAnaly_dir


e_lattice = cn.coordination.makelattice(64, 64, [0,0])
data = mydata.mydata()
data_anly = mydata.mydata()

mua_win = 1
auto_embd = 1
embd_mtd = 'RAGWITZ' # AisTe RAGWITZ

max_k_search = '8'
max_tau_search = '1'
src_History_d=8; src_History_tau=1 
trg_History_d=8; trg_History_tau=1 
surr = 0
norm = 0
if norm:
    normalize = 'true'
else:    
    normalize = 'false'

mua_range = 5
sample_interval = 1
if auto_embd:
    savefile_name = 'data_anly_TEdelay_muawin%d_autoembd%d_embdmtd%s_surr%d_maxk%s_maxtau%s_muasampintv%d_norm%d_'%(mua_win,auto_embd,embd_mtd,surr,max_k_search,max_tau_search,sample_interval,norm)

else:
    savefile_name = 'data_anly_TEdelay_muawin%d_autoembd%d_surr%d_k%d_tau%d_muasampintv%d_norm%d_'%(mua_win,auto_embd,surr,src_History_d, src_History_tau, sample_interval,norm)

# savefile_name = 'data_anly_TEdelay_muawin%d_autoembd%d_surr%d_searchingmorek_muarg%d_'%(mua_win,auto_embd,surr,mua_range)
#%%
sys_argv = int(sys.argv[1])
loop_num = sys_argv

#overLap_min = 30
name_sfx = '' # Evt_30ovlp
unsyncType = 'bothOff'  # eitherOff bothOff
# substract_mean = False

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
# def multiMUA_unequalDura(spk_mat, dura, record_neugroup, sample_interval=20, window=20, dt=0.1): #, returnHz=True):
#     '''
#     spk_mat:
#         sparse matrix for time and index of spikes
#     dura: 2-D array
#         each row specifies the start and end of each analysis duration; ms
#         length of each duration can be unequal
#     record_neugroup: list
#         list consists of arrays, each array contains the index of a group of neurons for MUA calculation 
#     sample_interval: ms
#         sample interval between two adjacent MUA calculation points
#     window: ms
#         length of window to count spike for MUA
#     dt: ms
#         simulation timestep
#     (analysis duration which is shorter than 'window' will be ignored automatically, i.e., return np.array([]) to 'spk_')
#     '''
#     #dt_ = int(round(1/dt))
#     # spk_mat = spk_mat.tocsc()
    
#     #MUA = np.zeros([len(record_neugroup), len(dura)])
    
#     spk = []
#     for neu_i, neu_ in enumerate(record_neugroup):
#         #for stim_i, stim_dura in enumerate(dura):
#         spk_ = fra.get_spkcount_sparmat_multi_unequalDura(spk_mat[neu_], dura, sum_activity=True, \
#                     sample_interval = sample_interval,  window = window, dt = dt)
#         if len(spk_) == 0:
#             spk.append(spk_)
#         else:
#             spk_ = np.hstack(spk_)
#             spk.append(spk_)
    
#     spk = np.vstack(spk)
    
#     return spk


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
def get_mua(spk_posi, analy_Sync):
    
    # for spk_posi in ['ctr']: # 'cor'
    #     for sfx in ['ctr']:
    #         for get_att in [1, ]:
    #             for analy_Sync in [1]:
    print('spk_posi:',spk_posi, 'analy_Sync:',analy_Sync) # 'sfx:',sfx,
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

    
    
    MUA_1_all = []
    MUA_2_all = []
    # MUA_1_all_net = []
    # MUA_2_all_net = []
    
    #n_neu_perGroup = np.array([len(record_neugroup[i]) for i in range(len(record_neugroup))])
    
    
    # for loop_num in range(40):
    print(loop_num)
    #data = mydata.mydata()
    data.load(datapath+'data%d.file'%loop_num)
    #data_anly = mydata.mydata()
    data_anly.load(dataAnaly_path+'data_anly_onoff_testthres_win10_min10_smt1_mtd1_%s_%d.file'%(spk_posi,loop_num)) # data_anly_onoff_thres_cor; data_anly_onoff_thres
    #data_anly.load(dataAnaly_path+'data_anly_onoff_thres%s%d.file'%(sfx,loop_num)) # data_anly_onoff_thres_cor; data_anly_onoff_thres
    '''
    n_perStimAmp = data.a1.param.stim1.n_perStimAmp
    n_StimAmp = data.a1.param.stim1.n_StimAmp
    '''
    # dt = 1/10000;
    # end = int(20/dt); start = int(5/dt)
    # spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/15/data.a1.param.Ne
    # spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne
    
    simu_time_tot = data.param.simutime
    # data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10], 'csc')
    # data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10], 'csc')
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
    '''
    if get_att:
        analy_dura = data.a1.param.stim1.stim_on[n_perStimAmp:].copy()
    else:
        analy_dura = data.a1.param.stim1.stim_on[:n_perStimAmp].copy()
    '''
    analy_dura = np.array([[5000, 205000]])
    # ignTrans = data_anly.onoff_sens.ignore_respTransient; print(ignTrans)
    ignTrans = 0
    for st_ind, t in enumerate(analy_dura): # analy_dura[0].reshape(1,-1)
        
        spk_matrix_MUA_1 = data.a1.ge.spk_matrix[:, (t[0]+ignTrans)*10:t[1]*10].copy()
        spk_matrix_MUA_2 = data.a2.ge.spk_matrix[:, (t[0]+ignTrans)*10:t[1]*10].copy()
        '''
        if get_att:
            onset_t_sample, offset_t_sample = find_sampleDuraSyncUnsync(data_anly.onoff_sens.stim_att[0].onoff_bool[st_ind].copy(),\
                                                                data_anly.onoff_asso.stim_att[0].onoff_bool[st_ind].copy(), sync = analy_Sync, unsyncType = unsyncType)  
        else:
            onset_t_sample, offset_t_sample = find_sampleDuraSyncUnsync(data_anly.onoff_sens.stim_noatt[0].onoff_bool[st_ind].copy(),\
                                                                data_anly.onoff_asso.stim_noatt[0].onoff_bool[st_ind].copy(), sync = analy_Sync, unsyncType = unsyncType)  
        '''
        onset_t_sample, offset_t_sample = find_sampleDuraSyncUnsync(data_anly.onoff_sens.spon.onoff_bool[st_ind].copy(),\
                                                            data_anly.onoff_asso.spon.onoff_bool[st_ind].copy(), sync = analy_Sync, unsyncType = unsyncType)  
        
        """
        if st_ind == 0 and loop_num == 0:
            
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
            

            
            fig, ax = plt.subplots(2,2, figsize=[14,8])
            #ax[0].plot(t_plt_x, ON[t_plt]*80)
            ni,tt = spk_matrix_MUA_1[neu_id,t_plt_s].nonzero()
            ax[0,0].plot(tt*0.1 + t_s, ni, '|')
            #ax[0].plot(t_plt_x,  data_anly.onoff_sens.stim_att[0].onoff_bool[st_ind][t_plt]*80)
            ni,tt = spk_matrix_MUA_2[neu_id,t_plt_s].nonzero()
            ax[1,0].plot(tt*0.1 + t_s, ni, '|')   
            #ax[1].plot(t_plt_x, data_anly.onoff_asso.stim_att[0].onoff_bool[st_ind][t_plt]*80)
            
            for ot, ft in zip(onset_t_, offset_t_):
                ax[0,0].fill_between(np.arange(ot, ft), 0, 80, color=[1.,1.,1.], alpha=1, zorder=5)
                ax[1,0].fill_between(np.arange(ot, ft), 0, 80, color=[1.,1.,1.], alpha=1, zorder=5)

            #fig, ax = plt.subplots(2,1)
            #ax[0].plot(t_plt_x, ON[t_plt]*80)
            ni,tt = spk_matrix_MUA_1[neu_id,t_plt_s].nonzero()
            ax[0,1].plot(tt*0.1 + t_s, ni, '|')
            #ax[0].plot(t_plt_x,  data_anly.onoff_sens.stim_att[0].onoff_bool[st_ind][t_plt]*80)
            ni,tt = spk_matrix_MUA_2[neu_id,t_plt_s].nonzero()
            ax[1,1].plot(tt*0.1 + t_s, ni, '|')
            
            if analy_Sync == 0:
                if unsyncType == 'eitherOff': # eitherOff
                    figtitle = 'spk%s_onoff%s_sync%dEtr_att%d_%d.png'%(spk_posi, sfx, analy_Sync, get_att,loop_num)
                else:
                    figtitle = 'spk%s_onoff%s_sync%d_att%d_%d.png'%(spk_posi, sfx, analy_Sync, get_att,loop_num)
            else:
                figtitle = 'spk%s_onoff%s_sync%d_att%d_%d.png'%(spk_posi, sfx, analy_Sync, get_att,loop_num)
                
            fig.suptitle(figtitle)
            #fig.savefig('results/subspace_syncUnsync/divideEvt/%s'%figtitle)
            #plt.close(fig)
        """
   
        
        '''onset and offset time points'''
        sample_t = np.vstack((onset_t_sample, offset_t_sample)).T
        #onoff_t_spon2 = np.vstack((data_anly.onoff_asso.spon.onset_t_sample[0], data_anly.onoff_asso.spon.offset_t_sample[0])).T
    
        # print(neu_id.shape)
        MUA_1 = fra.get_spkcount_sparmat_multi_unequalDura(spk_matrix_MUA_1[neu_id], sample_t, sum_activity=True, \
        sample_interval = sample_interval,  window = mua_win, dt = 0.1)
        print('len(MUA_1):',len(MUA_1), end='; ')
        # print(MUA_1[0].shape)
        
        MUA_2 = fra.get_spkcount_sparmat_multi_unequalDura(spk_matrix_MUA_2[neu_id], sample_t, sum_activity=True, \
        sample_interval = sample_interval,  window = mua_win, dt = 0.1)
                    
        # MUA_1 = multiMUA_unequalDura(spk_matrix_MUA_1, sample_t, record_neugroup, sample_interval=20, window=20, dt=0.1) #, returnHz=True)
        # MUA_2 = multiMUA_unequalDura(spk_matrix_MUA_2, sample_t, record_neugroup, sample_interval=20, window=20, dt=0.1) #, returnHz=True)
        
        MUA_1_all += MUA_1
        MUA_2_all += MUA_2  
    
    return MUA_1_all, MUA_2_all, neu_id         
#%%  

def get_TE_delay(MUA_1_all, MUA_2_all, delay_list, auto_embd, \
                 src_History_d=1, src_History_tau=1, trg_History_d=1, trg_History_tau=1, \
                     surr = False):         
    
    r = mydata.mydata()
    
    numTrials = len(MUA_1_all)
    TE_delay = []
    # delay_list = np.arange(1,21)
    # delay_list = np.arange(1,21)
    
    # auto_embd = True
    embd_delay = {'k':[],'l':[],'k_tau':[],'l_tau':[]}

    if surr:
        surrDist_delay = []
    
    # trg_History_d = 1
    # trg_History_tau = 1
    
    # src_History_d = 1
    # src_History_tau = 1
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov

    tic_all = time.perf_counter()
    
    for delay in delay_list:
        
        # if delay == 1:
        #     trg_History_d = 1
        # else:
        #     trg_History_d = 2
        
        # trg_History_tau = delay - 1
    
        # trg_History_d = 2
        # trg_History_tau = delay - 1
        
        # src_History_d = 1
        # src_History_tau = 1
        
        # delay = 10
        
        # 
        # 
        teCalc = teCalcClass()
        teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
        # teCalc.initialise(trg_History_d, trg_History_tau, src_History_d, src_History_tau, delay) # Use target history length of kHistoryLength (Schreiber k)
        if auto_embd:
            if embd_mtd == 'RAGWITZ':
                teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
                                   teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
            elif embd_mtd == 'AisTe':
                teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
                                   teCalcClass.AUTO_EMBED_METHOD_MAX_CORR_AIS_AND_TE)
            else: raise Exception('invalid embedding method!')
                
            teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX,
        		max_k_search)
        
            teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX,
        		max_tau_search)    
            
            teCalc.setProperty(teCalcClass.DELAY_PROP_NAME, str(delay))
            teCalc.setProperty('NORMALISE', normalize)
                
            teCalc.initialise() # Use target history length of kHistoryLength (Schreiber k)
        else:
            teCalc.setProperty('NORMALISE', normalize)
            teCalc.initialise(trg_History_d, trg_History_tau, src_History_d, src_History_tau, delay) # Use target history length of kHistoryLength (Schreiber k)        
            
            
        teCalc.startAddObservations()
        
        for trial in range(0,numTrials):

            teCalc.addObservations(JArray(JDouble, 1)((MUA_1_all[trial]/len(neu_id)/(mua_win*0.001)).tolist()), 
                                    JArray(JDouble, 1)((MUA_2_all[trial]/len(neu_id)/(mua_win*0.001)).tolist()))
        
        # We've finished adding trials:
        print('numTrials:', numTrials)
        print("Finished adding trials")
        teCalc.finaliseAddObservations()
        #%
        # plt.figure()
        # plt.plot(sourceArray)
        # plt.plot(destArray)
        
        
        #%
        # Compute the TE:
        print('delay:',delay)
        print("Computing TE ...")
        '''
        # TE = np.array(teCalc.computeLocalOfPreviousObservations())
        # localValuesPerTrial = np.array(teCalc.getSeparateNumObservations())
        # TE = TE.reshape(-1, localValuesPerTrial[0])
        
        # result_on_noatt.append(TE.mean(0))
        '''
        tic = time.perf_counter()
        TE = teCalc.computeAverageLocalOfObservations()
        print('time elapsed:',np.round((time.perf_counter() - tic)/60,2), 'min')

        embd_delay['k'].append(float(str(teCalc.getProperty(teCalc.K_PROP_NAME))))
        embd_delay['l'].append(float(str(teCalc.getProperty(teCalc.L_PROP_NAME))))
        embd_delay['k_tau'].append(float(str(teCalc.getProperty(teCalc.K_TAU_PROP_NAME))))
        embd_delay['l_tau'].append(float(str(teCalc.getProperty(teCalc.L_TAU_PROP_NAME))))
    
        TE_delay.append(float(TE))
        
        print(teCalc.getProperty(teCalc.PROP_AUTO_EMBED_METHOD))
        
        if surr:
            surr_result = teCalc.computeSignificance(200)
            surrDist_delay.append(np.array(surr_result.distribution))
        
        del teCalc
        
    print('total time elapsed:',np.round((time.perf_counter() - tic_all)/60,2), 'min')
    
    r.TE_delay = TE_delay
    r.embd_delay = embd_delay
    r.delay_list = delay_list
    if surr:
        r.surrDist_delay = surrDist_delay
    
    return r
#%%
spk_posi = 'ctr'
delay_list = np.arange(1,21)

data_anly_te = mydata.mydata()
'''spon, both on'''
data_anly_te.spon = mydata.mydata()

analy_Sync = 1
data_anly_te.spon.bothON = mydata.mydata()

MUA_1_all, MUA_2_all, neu_id = get_mua(spk_posi, analy_Sync)

r = get_TE_delay(MUA_1_all, MUA_2_all, delay_list, auto_embd, \
                 src_History_d, src_History_tau, trg_History_d, trg_History_tau, surr)

data_anly_te.spon.bothON = r   
#%%
'''spon, both off'''
analy_Sync = 0
data_anly_te.spon.bothOFF = mydata.mydata()

MUA_1_all, MUA_2_all, neu_id = get_mua(spk_posi, analy_Sync)

r = get_TE_delay(MUA_1_all, MUA_2_all, delay_list, auto_embd, \
                 src_History_d, src_History_tau, trg_History_d, trg_History_tau, surr)

data_anly_te.spon.bothOFF = r 
           

#%%

data_anly_te.save(data_anly_te.class2dict(), savedatapath+savefile_name+'%d.file'%loop_num)

#%%
sys.exit()

