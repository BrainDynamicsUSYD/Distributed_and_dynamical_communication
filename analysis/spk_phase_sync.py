#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 00:07:48 2021

@author: shni2598
"""


import frequency_analysis as fqa
import firing_rate_analysis as fra
import numpy as np

import get_onoff_cpts
#%%


def get_spk_phase_sync(spk_mat, spk_mat_phase, dura, onoff_bool_sens, onoff_bool_asso, ignTrans, passband):
 

    #ignTrans = 200
    
    supply_start = 5 # 10: ms, length of window used in ON-OFF states detection
    supply_end = 5
    MUA_window = 1
    # bothON_ = np.logical_and(data_anly.onoff_sens.stim_noatt[0].onoff_bool[0], data_anly.onoff_asso.stim_noatt[0].onoff_bool[0])
    # bothON_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothON_, np.zeros(supply_end,dtype=bool)])
    # bothOFF_ = np.logical_not(np.logical_or(onoff_bool_sens[t_i], onoff_bool_asso[t_i]))
    # bothOFF_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothOFF_, np.zeros(supply_end,dtype=bool)])
    
    neu_n = spk_mat.shape[0]
    spk_phase_on = [[[[] for _ in range(len(dura))] for __ in range(neu_n)] for ___ in range(len(passband))]
    spk_phase_off = [[[[] for _ in range(len(dura))] for __ in range(neu_n)] for ___ in range(len(passband))]
    
    
    for t_i, t in enumerate(dura):
        print('trial: %d'%t_i)
        MUA_forPhase = fra.get_spkcount_sum_sparmat(spk_mat_phase, t[0], t[1],\
                       sample_interval = 0.1,  window = MUA_window, dt = 0.1)/spk_mat_phase.shape[0]/(MUA_window/1000)
    
    
    
    
        bothON_ = np.logical_and(onoff_bool_sens[t_i], onoff_bool_asso[t_i])
        bothON_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothON_, np.zeros(supply_end,dtype=bool)])
        bothOFF_ = np.logical_not(np.logical_or(onoff_bool_sens[t_i], onoff_bool_asso[t_i]))
        bothOFF_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothOFF_, np.zeros(supply_end,dtype=bool)])

        
        on_t, off_t, onset_t_bON, offset_t_bON = get_onoff_cpts.get_onoff_cpts(bothON_)
        
        if offset_t_bON[0] < onset_t_bON[0]:
            offset_t_bON = np.delete(offset_t_bON, 0)
        if onset_t_bON[-1] > offset_t_bON[-1]:
            onset_t_bON = np.delete(onset_t_bON, -1)
        
        onset_t_bON += ignTrans
        offset_t_bON += ignTrans
    
        on_t, off_t, onset_t_bOFF, offset_t_bOFF = get_onoff_cpts.get_onoff_cpts(bothOFF_)
        
        if offset_t_bOFF[0] < onset_t_bOFF[0]:
            offset_t_bOFF = np.delete(offset_t_bOFF, 0)
        if onset_t_bOFF[-1] > offset_t_bOFF[-1]:
            onset_t_bOFF = np.delete(onset_t_bOFF, -1)
        
        onset_t_bOFF += ignTrans
        offset_t_bOFF += ignTrans
        

        # _, MUA_2_filt_hil = fqa.get_filt_hilb_1(MUA_2, passband, Fs = 10000, filterOrder = 8)
        # MUA_2_phase = np.angle(MUA_2_filt_hil)
        # MUA_2_phase = np.concatenate([np.zeros(5), MUA_2_phase, np.zeros(4)])

        for pb_i in range(passband.shape[0]):
        
            _, MUA_forPhase_filt_hil = fqa.get_filt_hilb_1(MUA_forPhase, passband[pb_i], Fs = 10000, filterOrder = 8)
            MUA_phase = np.angle(MUA_forPhase_filt_hil)
            MUA_phase = np.concatenate([np.zeros(5), MUA_phase, np.zeros(4)]); #print(MUA_phase.shape)

            for n_i in range(neu_n):
                for ont, offt in zip(onset_t_bON, offset_t_bON):
                    spk_phase_on[pb_i][n_i][t_i].append(MUA_phase[ont*10 + spk_mat[n_i, (t[0]+ont)*10:(t[0]+offt)*10].nonzero()[1]])
            
                if len(spk_phase_on[pb_i][n_i][t_i]) == 0:
                    spk_phase_on[pb_i][n_i][t_i] = np.array([])
                else:
                    spk_phase_on[pb_i][n_i][t_i] = np.concatenate(spk_phase_on[pb_i][n_i][t_i])

                for ont, offt in zip(onset_t_bOFF, offset_t_bOFF):
                    spk_phase_off[pb_i][n_i][t_i].append(MUA_phase[ont*10 + spk_mat[n_i, (t[0]+ont)*10:(t[0]+offt)*10].nonzero()[1]])
            
                if len(spk_phase_off[pb_i][n_i][t_i]) == 0:
                    spk_phase_off[pb_i][n_i][t_i] = np.array([])
                else:
                    spk_phase_off[pb_i][n_i][t_i] = np.concatenate(spk_phase_off[pb_i][n_i][t_i])

            
    ppc_bothON = np.zeros([passband.shape[0], neu_n])
    ppc_bothOFF = np.zeros([passband.shape[0], neu_n])
    
    for pb_i in range(passband.shape[0]):
        print('pass band:',passband[pb_i])
        for n_i in range(neu_n):
            
            ppc_bothON[pb_i][n_i] = get_ppc(spk_phase_on[pb_i][n_i])
            ppc_bothOFF[pb_i][n_i] = get_ppc(spk_phase_off[pb_i][n_i])
    
    return ppc_bothON, ppc_bothOFF


def get_ppc(spk_phase):
    trial_n = len(spk_phase)
    ppc = 0
    nonzero_t_n = 0
    # n_phase = 0
    # for item in spk_phase:
    #     n_phase += len(item)
    # print('n_phase:%d'%n_phase)
    for ti in range(trial_n):
        for tj in range(trial_n):
            #print(ti, tj, nonzero_t_n)
            if ti == tj:
                continue
            if spk_phase[ti].shape[0] == 0 or spk_phase[tj].shape[0] == 0:
                continue
            nonzero_t_n += 1
            
            ppc += np.cos(spk_phase[ti].reshape(-1,1) - spk_phase[tj]).mean()
    
    ppc /= nonzero_t_n  
    return ppc    
