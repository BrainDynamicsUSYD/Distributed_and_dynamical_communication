#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 17:52:43 2022

@author: Shencong Ni
"""



import mydata
import numpy as np

import firing_rate_analysis as fra
import get_onoff_cpts
import connection as cn
import coordination as cd

import sys
import os
import matplotlib.pyplot as plt
import time

import scipy.io as sio

#%%
from jpype import *

'''
Calculate the transfer entropy (TE) between the MUA in area 1 and area 2.
Run the onoff_detection.py first before running this script.

'Java Information Dynamics Toolkit' is used for this analysis and should be installed before running this script.
See "https://jlizier.github.io/jidt/" for more information about this toolkit.

'''

jarLocation = '/headnode1/shni2598/infodynamics/infodynamics.jar'
# Path to the 'infodynamics.jar'. Replace it with your path to 'infodynamics.jar'.

startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

#%%
'''evoked activity; uncued/cued'''

datapath = 'raw_data/'
savedatapath = 'raw_data/te_data/'
if not os.path.exists(savedatapath):
    try: os.makedirs(savedatapath)
    except FileExistsError:
        pass
    
dataAnaly_dir = 'raw_data/' # 
dataAnaly_path = '' + dataAnaly_dir


e_lattice = cn.coordination.makelattice(64, 64, [0,0])
data = mydata.mydata()
data_anly = mydata.mydata()

mua_win = 1 # ms
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


sys_argv = int(sys.argv[1])
loop_num = sys_argv

name_sfx = '' # 
unsyncType = 'bothOff'  # 

if os.path.isfile(savedatapath+savefile_name+'%d.file'%loop_num):
    sys.exit('Already done. Exit.')

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
def get_mua(spk_posi, get_att, analy_Sync):
    
    print('spk_posi:',spk_posi,  'get_att:',get_att, 'analy_Sync:',analy_Sync) # 'sfx:',sfx,

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
    data_anly.load(dataAnaly_path+'data_anly_onoff_testthres_win10_min10_smt1_mtd1_%s_%d.file'%(spk_posi,loop_num)) # data_anly_onoff_thres_cor; data_anly_onoff_thres

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
    for st_ind, t in enumerate(analy_dura): # 
        
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
    
        MUA_1 = fra.get_spkcount_sparmat_multi_unequalDura(spk_matrix_MUA_1[neu_id], sample_t, sum_activity=True, \
        sample_interval = sample_interval,  window = mua_win, dt = 0.1)
        print('len(MUA_1):',len(MUA_1), end='; ')
        
        MUA_2 = fra.get_spkcount_sparmat_multi_unequalDura(spk_matrix_MUA_2[neu_id], sample_t, sum_activity=True, \
        sample_interval = sample_interval,  window = mua_win, dt = 0.1)
                            
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
    
    embd_delay = {'k':[],'l':[],'k_tau':[],'l_tau':[]}

    if surr:
        surrDist_delay = []
    
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov

    tic_all = time.perf_counter()
    
    for delay in delay_list:
                
        teCalc = teCalcClass()
        teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
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

            	teCalc.addObservations(JArray(JDouble, 1)((MUA_1_all[trial]/len(neu_id)/(mua_win*0.001)).tolist()), JArray(JDouble, 1)((MUA_2_all[trial]/len(neu_id)/(mua_win*0.001)).tolist()))
        
        # finished adding trials:
        print('numTrials:', numTrials)
        print("Finished adding trials")
        teCalc.finaliseAddObservations()

        # Compute the TE:
        print('delay:',delay)
        print("Computing TE ...")

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
delay_list = np.arange(1,21) # time delay, ms

data_anly_te = mydata.mydata()
'''no att, both on'''
get_att = 0
data_anly_te.noatt = mydata.mydata()

analy_Sync = 1
data_anly_te.noatt.bothON = mydata.mydata()

MUA_1_all, MUA_2_all, neu_id = get_mua(spk_posi, get_att, analy_Sync)

r = get_TE_delay(MUA_1_all, MUA_2_all, delay_list, auto_embd, \
                 src_History_d, src_History_tau, trg_History_d, trg_History_tau, surr)

data_anly_te.noatt.bothON = r   

'''no att, both off'''
analy_Sync = 0
data_anly_te.noatt.bothOFF = mydata.mydata()

MUA_1_all, MUA_2_all, neu_id = get_mua(spk_posi, get_att, analy_Sync)

r = get_TE_delay(MUA_1_all, MUA_2_all, delay_list, auto_embd, \
                 src_History_d, src_History_tau, trg_History_d, trg_History_tau, surr)

data_anly_te.noatt.bothOFF = r 
           
'''att, both on'''
get_att = 1
data_anly_te.att = mydata.mydata()

analy_Sync = 1
data_anly_te.att.bothON = mydata.mydata()

MUA_1_all, MUA_2_all, neu_id = get_mua(spk_posi, get_att, analy_Sync)

r = get_TE_delay(MUA_1_all, MUA_2_all, delay_list, auto_embd, \
                 src_History_d, src_History_tau, trg_History_d, trg_History_tau, surr)

data_anly_te.att.bothON = r  

'''att, both off'''
analy_Sync = 0
data_anly_te.att.bothOFF = mydata.mydata()

MUA_1_all, MUA_2_all, neu_id = get_mua(spk_posi, get_att, analy_Sync)

r = get_TE_delay(MUA_1_all, MUA_2_all, delay_list, auto_embd, \
                 src_History_d, src_History_tau, trg_History_d, trg_History_tau, surr)

data_anly_te.att.bothOFF = r 

#%%

data_anly_te.save(data_anly_te.class2dict(), savedatapath+savefile_name+'%d.file'%loop_num)

#%%
sys.exit()






