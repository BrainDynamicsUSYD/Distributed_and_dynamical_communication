#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:43:50 2021

@author: shni2598
"""

import frequency_analysis as fqa
import firing_rate_analysis as fra
import numpy as np

#from get_onoff_cpts import get_onoff_cpts 
#%%

def get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, dura, \
                   onoff_bool_sens, onoff_bool_asso, 
                   passband, ignTrans, \
                   MUA_window = 2, sample_interval = 1, sameSampleSize = True, surrogate = False):
    #%
    MUA = []
    bothON = []
    bothOFF = []
    
    hilb_bothON = [[] for _ in passband]
    hilb_bothOFF = [[] for _ in passband]
    
    
    mua_neuron1 = spk_mat_MUA_1.shape[0]
    mua_neuron2 = spk_mat_MUA_2.shape[0]
    
    for t_i, t in enumerate(dura):
        
        MUA_1 = fra.get_spkcount_sum_sparmat(spk_mat_MUA_1, t[0], t[1],\
                           sample_interval = sample_interval,  window = MUA_window, dt = 0.1)/mua_neuron1/(MUA_window/1000)
        MUA_2 = fra.get_spkcount_sum_sparmat(spk_mat_MUA_2, t[0], t[1],\
                           sample_interval = sample_interval,  window = MUA_window, dt = 0.1)/mua_neuron2/(MUA_window/1000)
        
        MUA.append(np.vstack([MUA_1, MUA_2]))
        
                   
        supply_start = (10 - MUA_window) // 2 # 10: ms, length of window used in ON-OFF states detection
        supply_end = (10 - MUA_window) - supply_start
    
        bothON_ = np.logical_and(onoff_bool_sens[t_i], onoff_bool_asso[t_i])
        bothON_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothON_, np.zeros(supply_end,dtype=bool)])
        bothOFF_ = np.logical_not(np.logical_or(onoff_bool_sens[t_i], onoff_bool_asso[t_i]))
        bothOFF_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothOFF_, np.zeros(supply_end,dtype=bool)])
        if sameSampleSize:
            n_off = bothOFF_.sum()
            n_on = bothON_.sum()
            if n_off > n_on:
                bothOFF_[np.where(bothOFF_)[0][:n_off-n_on]] = False
            else:
                bothON_[np.where(bothON_)[0][:n_on-n_off]] = False
                
                
        bothON.append(bothON_)
        bothOFF.append(bothOFF_)
        
        for pb_i, pb in enumerate(passband):
            MUA1_filt, MUA2_filt, MUA1_hilb, MUA2_hilb = fqa.get_filt_hilb(MUA_1, MUA_2, pb, Fs = 1000, filterOrder = 8)
        
            
            #amp_phaseDiff = MUA1_hilb*np.conj(MUA2_hilb)
            hilb_bothON[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothON[t_i]], MUA2_hilb[ignTrans:][bothON[t_i]]]))
            hilb_bothOFF[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothOFF[t_i]], MUA2_hilb[ignTrans:][bothOFF[t_i]]]))
    
    for pb_i in range(len(passband)):
        hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
        hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
        
    cohe_ON, PLV_ON, cohe_OFF, PLV_OFF, WPLI_ON, WPLI_OFF = \
        np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband))
    
        
    for pb_i in range(len(passband)):

        cohe_ON[pb_i], PLV_ON[pb_i], WPLI_ON[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1], return_WPLI = True)
        cohe_OFF[pb_i], PLV_OFF[pb_i], WPLI_OFF[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1], return_WPLI = True)
    
    if not surrogate:
        return cohe_ON ,PLV_ON, WPLI_ON, cohe_OFF, PLV_OFF, WPLI_OFF
    else:
        cohe_ON_p, PLV_ON_p, WPLI_ON_p, cohe_OFF_p, PLV_OFF_p, WPLI_OFF_p = \
            np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200])
        
        for perm_i in range(200):
            if perm_i%20 == 0: print('surrogate: %.1f%%'%(perm_i/2))
            
            hilb_bothON = [[] for _ in passband]
            hilb_bothOFF = [[] for _ in passband]

            for t_i, t in enumerate(dura):
                
                MUA_1_perm = np.random.permutation(MUA[t_i][0])
                MUA_2_perm = np.random.permutation(MUA[t_i][1])
                for pb_i, pb in enumerate(passband):
                    MUA1_filt, MUA2_filt, MUA1_hilb, MUA2_hilb = \
                        fqa.get_filt_hilb(MUA_1_perm, MUA_2_perm, \
                                          pb, Fs = 1000, filterOrder = 8)
                    hilb_bothON[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothON[t_i]], MUA2_hilb[ignTrans:][bothON[t_i]]]))
                    hilb_bothOFF[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothOFF[t_i]], MUA2_hilb[ignTrans:][bothOFF[t_i]]]))
            
            for pb_i in range(len(passband)):
            
                hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
                hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
            
                cohe_ON_p[pb_i, perm_i], PLV_ON_p[pb_i, perm_i], WPLI_ON_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1], return_WPLI = True)
                cohe_OFF_p[pb_i, perm_i], PLV_OFF_p[pb_i, perm_i], WPLI_OFF_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1], return_WPLI = True)
        
        return cohe_ON ,PLV_ON, WPLI_ON, cohe_OFF, PLV_OFF, WPLI_OFF, \
            cohe_ON_p, PLV_ON_p, WPLI_ON_p, cohe_OFF_p, PLV_OFF_p, WPLI_OFF_p
            
           
#%%            
def get_phase_sync_sig(sig1, sig2, dura, \
                   onoff_bool_sens, onoff_bool_asso, 
                   passband, ignTrans, \
                       sameSampleSize = True, surrogate = False):
    #%
    sig = []
    bothON = []
    bothOFF = []
    
    hilb_bothON = [[] for _ in passband]
    hilb_bothOFF = [[] for _ in passband]
    
    
    
    for t_i, t in enumerate(dura):
        
        sig1_seg = sig1[t[0]:t[1]]
        sig2_seg = sig2[t[0]:t[1]]
        
        sig.append(np.vstack([sig1_seg, sig2_seg]))
        
                   
        supply_start = (10 - 1) // 2 # 10: ms, length of window used in ON-OFF states detection
        supply_end = (10 - 1) - supply_start
    
        bothON_ = np.logical_and(onoff_bool_sens[t_i], onoff_bool_asso[t_i])
        bothON_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothON_, np.zeros(supply_end,dtype=bool)])
        bothOFF_ = np.logical_not(np.logical_or(onoff_bool_sens[t_i], onoff_bool_asso[t_i]))
        bothOFF_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothOFF_, np.zeros(supply_end,dtype=bool)])
        if sameSampleSize:
            n_off = bothOFF_.sum()
            n_on = bothON_.sum()
            if n_off > n_on:
                bothOFF_[np.where(bothOFF_)[0][:n_off-n_on]] = False
            else:
                bothON_[np.where(bothON_)[0][:n_on-n_off]] = False
                
                
        bothON.append(bothON_)
        bothOFF.append(bothOFF_)
        
        for pb_i, pb in enumerate(passband):
            sig1_filt, sig2_filt, sig1_hilb, sig2_hilb = fqa.get_filt_hilb(sig1_seg, sig2_seg, pb, Fs = 1000, filterOrder = 8)
        
            
            #amp_phaseDiff = sig1_hilb*np.conj(sig2_hilb)
            hilb_bothON[pb_i].append(np.vstack([sig1_hilb[ignTrans:][bothON[t_i]], sig2_hilb[ignTrans:][bothON[t_i]]]))
            hilb_bothOFF[pb_i].append(np.vstack([sig1_hilb[ignTrans:][bothOFF[t_i]], sig2_hilb[ignTrans:][bothOFF[t_i]]]))
    
    for pb_i in range(len(passband)):
        hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
        hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
        
    cohe_ON, PLV_ON, cohe_OFF, PLV_OFF, WPLI_ON, WPLI_OFF = \
        np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband))
    
        
    for pb_i in range(len(passband)):

        cohe_ON[pb_i], PLV_ON[pb_i], WPLI_ON[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1], return_WPLI = True)
        cohe_OFF[pb_i], PLV_OFF[pb_i], WPLI_OFF[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1], return_WPLI = True)
    
    if not surrogate:
        return cohe_ON ,PLV_ON, WPLI_ON, cohe_OFF, PLV_OFF, WPLI_OFF
    else:
        cohe_ON_p, PLV_ON_p, WPLI_ON_p, cohe_OFF_p, PLV_OFF_p, WPLI_OFF_p = \
            np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200])
        
        for perm_i in range(200):
            if perm_i%20 == 0: print('surrogate: %.1f%%'%(perm_i/2))
            
            hilb_bothON = [[] for _ in passband]
            hilb_bothOFF = [[] for _ in passband]

            for t_i, t in enumerate(dura):
                
                sig1_perm = np.random.permutation(sig[t_i][0])
                sig2_perm = np.random.permutation(sig[t_i][1])
                for pb_i, pb in enumerate(passband):
                    sig1_filt, sig2_filt, sig1_hilb, sig2_hilb = \
                        fqa.get_filt_hilb(sig1_perm, sig2_perm, \
                                          pb, Fs = 1000, filterOrder = 8)
                    hilb_bothON[pb_i].append(np.vstack([sig1_hilb[ignTrans:][bothON[t_i]], sig2_hilb[ignTrans:][bothON[t_i]]]))
                    hilb_bothOFF[pb_i].append(np.vstack([sig1_hilb[ignTrans:][bothOFF[t_i]], sig2_hilb[ignTrans:][bothOFF[t_i]]]))
            
            for pb_i in range(len(passband)):
            
                hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
                hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
            
                cohe_ON_p[pb_i, perm_i], PLV_ON_p[pb_i, perm_i], WPLI_ON_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1], return_WPLI = True)
                cohe_OFF_p[pb_i, perm_i], PLV_OFF_p[pb_i, perm_i], WPLI_OFF_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1], return_WPLI = True)
        
        return cohe_ON ,PLV_ON, WPLI_ON, cohe_OFF, PLV_OFF, WPLI_OFF, \
            cohe_ON_p, PLV_ON_p, WPLI_ON_p, cohe_OFF_p, PLV_OFF_p, WPLI_OFF_p
         
#%%


def get_phase_sync_sig_whole(sig1, sig2, dura, passband, ignTrans, \
                                 surrogate = False):
    #%
    sig = []
    
    hilb = [[] for _ in passband]
    # hilb_bothOFF = [[] for _ in passband]
    
    
    
    for t_i, t in enumerate(dura):
        
        sig1_seg = sig1[t[0]:t[1]]
        sig2_seg = sig2[t[0]:t[1]]
        
        sig.append(np.vstack([sig1_seg, sig2_seg]))
        
                   
        # supply_start = (10 - 1) // 2 # 10: ms, length of window used in ON-OFF states detection
        # supply_end = (10 - 1) - supply_start
    
        # bothON_ = np.logical_and(onoff_bool_sens[t_i], onoff_bool_asso[t_i])
        # bothON_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothON_, np.zeros(supply_end,dtype=bool)])
        # bothOFF_ = np.logical_not(np.logical_or(onoff_bool_sens[t_i], onoff_bool_asso[t_i]))
        # bothOFF_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothOFF_, np.zeros(supply_end,dtype=bool)])
        # if sameSampleSize:
        #     n_off = bothOFF_.sum()
        #     n_on = bothON_.sum()
        #     if n_off > n_on:
        #         bothOFF_[np.where(bothOFF_)[0][:n_off-n_on]] = False
        #     else:
        #         bothON_[np.where(bothON_)[0][:n_on-n_off]] = False
                
                
        # bothON.append(bothON_)
        # bothOFF.append(bothOFF_)
        
        for pb_i, pb in enumerate(passband):
            sig1_filt, sig2_filt, sig1_hilb, sig2_hilb = fqa.get_filt_hilb(sig1_seg, sig2_seg, pb, Fs = 1000, filterOrder = 8)
                    
            hilb[pb_i].append(np.vstack([sig1_hilb[ignTrans:], sig2_hilb[ignTrans:]]))
    
    for pb_i in range(len(passband)):
        hilb[pb_i] = np.concatenate(hilb[pb_i], 1)
        
    cohe, PLV, WPLI = \
        np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband))    
        
    for pb_i in range(len(passband)):

        cohe[pb_i], PLV[pb_i], WPLI[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb[pb_i][0], hilb[pb_i][1], return_WPLI = True)
    
    if not surrogate:
        return cohe ,PLV, WPLI
    else:
        cohe_p, PLV_p, WPLI_p = \
            np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200])
            
        for perm_i in range(200):
            if perm_i%20 == 0: print('surrogate: %.1f%%'%(perm_i/2))
            
            hilb = [[] for _ in passband]

            for t_i, t in enumerate(dura):
                
                sig1_perm = np.random.permutation(sig[t_i][0])
                sig2_perm = np.random.permutation(sig[t_i][1])
                for pb_i, pb in enumerate(passband):
                    sig1_filt, sig2_filt, sig1_hilb, sig2_hilb = \
                        fqa.get_filt_hilb(sig1_perm, sig2_perm, \
                                          pb, Fs = 1000, filterOrder = 8)
                    hilb[pb_i].append(np.vstack([sig1_hilb[ignTrans:], sig2_hilb[ignTrans:]]))
            
            for pb_i in range(len(passband)):
            
                hilb[pb_i] = np.concatenate(hilb[pb_i], 1)
            
                cohe_p[pb_i, perm_i], PLV_p[pb_i, perm_i], WPLI_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb[pb_i][0], hilb[pb_i][1], return_WPLI = True)
        
        return cohe, PLV, WPLI, cohe_p, PLV_p, WPLI_p


#%%            
def get_phase_sync_sig_surroPhase(sig1, sig2, dura, \
                   onoff_bool_sens, onoff_bool_asso, 
                   passband, ignTrans, \
                       sameSampleSize = True, surrogate = False):
    #%
    sig = []
    bothON = []
    bothOFF = []
    
    hilb_bothON = [[] for _ in passband]
    hilb_bothOFF = [[] for _ in passband]
    
    ext = 500 # ms
    
    for t_i, t in enumerate(dura):
        
        sig1_seg = sig1[t[0]+ignTrans-ext:t[1]+ext]
        sig2_seg = sig2[t[0]+ignTrans-ext:t[1]+ext]
        
        sig.append(np.vstack([sig1_seg, sig2_seg]))
        
                   
        supply_start = (10 - 1) // 2 + 1# 10: ms, length of window used in ON-OFF states detection
        supply_end = (10 - 1) - supply_start
    
        bothON_ = np.logical_and(onoff_bool_sens[t_i], onoff_bool_asso[t_i])
        bothON_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothON_, np.zeros(supply_end,dtype=bool)])
        bothOFF_ = np.logical_not(np.logical_or(onoff_bool_sens[t_i], onoff_bool_asso[t_i]))
        bothOFF_ = np.concatenate([np.zeros(supply_start,dtype=bool), bothOFF_, np.zeros(supply_end,dtype=bool)])
        if sameSampleSize:
            n_off = bothOFF_.sum()
            n_on = bothON_.sum()
            if n_off > n_on:
                bothOFF_[np.where(bothOFF_)[0][:n_off-n_on]] = False
            else:
                bothON_[np.where(bothON_)[0][:n_on-n_off]] = False
                
                
        bothON.append(bothON_)
        bothOFF.append(bothOFF_)
        
        for pb_i, pb in enumerate(passband):
            sig1_filt, sig2_filt, sig1_hilb, sig2_hilb = fqa.get_filt_hilb(sig1_seg, sig2_seg, pb, Fs = 1000, filterOrder = 8)
        
            
            #amp_phaseDiff = sig1_hilb*np.conj(sig2_hilb)
            hilb_bothON[pb_i].append(np.vstack([sig1_hilb[ext:-ext][bothON[t_i]], sig2_hilb[ext:-ext][bothON[t_i]]]))
            hilb_bothOFF[pb_i].append(np.vstack([sig1_hilb[ext:-ext][bothOFF[t_i]], sig2_hilb[ext:-ext][bothOFF[t_i]]]))
    
    for pb_i in range(len(passband)):
        hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
        hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
        
    cohe_ON, PLV_ON, cohe_OFF, PLV_OFF, WPLI_ON, WPLI_OFF = \
        np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband))
    
        
    for pb_i in range(len(passband)):

        cohe_ON[pb_i], PLV_ON[pb_i], WPLI_ON[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1], return_WPLI = True)
        cohe_OFF[pb_i], PLV_OFF[pb_i], WPLI_OFF[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1], return_WPLI = True)
    
    if not surrogate:
        return cohe_ON ,PLV_ON, WPLI_ON, cohe_OFF, PLV_OFF, WPLI_OFF
    else:
        cohe_ON_p, PLV_ON_p, WPLI_ON_p, cohe_OFF_p, PLV_OFF_p, WPLI_OFF_p = \
            np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200])
        
        for pb_i in range(len(passband)):
            print('surrogate band: ', passband[pb_i])
            for perm_i in range(200):
                if perm_i%50 == 0: print('surrogate: %.1f%%'%(perm_i/2))
                
                cohe_ON_p[pb_i, perm_i], PLV_ON_p[pb_i, perm_i], WPLI_ON_p[pb_i, perm_i], _ = \
                    fqa.get_coherence_phaseLockValue(np.random.permutation(hilb_bothON[pb_i][0]), 
                                                     hilb_bothON[pb_i][1], return_WPLI = True)
                
                cohe_OFF_p[pb_i, perm_i], PLV_OFF_p[pb_i, perm_i], WPLI_OFF_p[pb_i, perm_i], _ = \
                    fqa.get_coherence_phaseLockValue(np.random.permutation(hilb_bothOFF[pb_i][0]), 
                                                     hilb_bothOFF[pb_i][1], 
                                                     return_WPLI = True)
                        
            # hilb_bothON = [[] for _ in passband]
            # hilb_bothOFF = [[] for _ in passband]

            # for t_i, t in enumerate(dura):
                
            #     sig1_perm = np.random.permutation(sig[t_i][0])
            #     sig2_perm = np.random.permutation(sig[t_i][1])
            #     for pb_i, pb in enumerate(passband):
            #         sig1_filt, sig2_filt, sig1_hilb, sig2_hilb = \
            #             fqa.get_filt_hilb(sig1_perm, sig2_perm, \
            #                               pb, Fs = 1000, filterOrder = 8)
            #         hilb_bothON[pb_i].append(np.vstack([sig1_hilb[ignTrans:][bothON[t_i]], sig2_hilb[ignTrans:][bothON[t_i]]]))
            #         hilb_bothOFF[pb_i].append(np.vstack([sig1_hilb[ignTrans:][bothOFF[t_i]], sig2_hilb[ignTrans:][bothOFF[t_i]]]))
            
            # for pb_i in range(len(passband)):
            
            #     hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
            #     hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
            
            #     cohe_ON_p[pb_i, perm_i], PLV_ON_p[pb_i, perm_i], WPLI_ON_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1], return_WPLI = True)
            #     cohe_OFF_p[pb_i, perm_i], PLV_OFF_p[pb_i, perm_i], WPLI_OFF_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1], return_WPLI = True)
        
        return cohe_ON ,PLV_ON, WPLI_ON, cohe_OFF, PLV_OFF, WPLI_OFF, \
            cohe_ON_p, PLV_ON_p, WPLI_ON_p, cohe_OFF_p, PLV_OFF_p, WPLI_OFF_p
         
            
         





          
'''
def get_phase_sync(spk_mat_MUA_1, spk_mat_MUA_2, dura, \
                   onoff_bool_sens, onoff_bool_asso, 
                   passband, ignTrans, \
                   MUA_window = 2, sample_interval = 1, sameSampleSize = True, surrogate = False):
    #%
    print(get_onoff_cpts)
    MUA = []
    bothON = []
    bothOFF = []
    
    hilb_bothON = [[] for _ in passband]
    hilb_bothOFF = [[] for _ in passband]
    
    
    mua_neuron1 = spk_mat_MUA_1.shape[0]
    mua_neuron2 = spk_mat_MUA_2.shape[0]
    
    for t_i, t in enumerate(dura):
        
        MUA_1 = fra.get_spkcount_sum_sparmat(spk_mat_MUA_1, t[0], t[1],\
                           sample_interval = sample_interval,  window = MUA_window, dt = 0.1)/mua_neuron1/(MUA_window/1000)
        MUA_2 = fra.get_spkcount_sum_sparmat(spk_mat_MUA_2, t[0], t[1],\
                           sample_interval = sample_interval,  window = MUA_window, dt = 0.1)/mua_neuron2/(MUA_window/1000)
        
        MUA.append(np.vstack([MUA_1, MUA_2]))
        

        supply_onoff = 10 - MUA_window
        onoff_bool_sens_ = np.concatenate([onoff_bool_sens[t_i], np.zeros(supply_onoff, dtype=bool)])
        on_t, off_t, onset_t, offset_t = get_onoff_cpts(onoff_bool_sens_)
        extend_t = 10 # ms; supply the length of window to count MUA during ON-OFF detection
        for oft in offset_t:
            onoff_bool_sens_[oft:oft+extend_t] = True
        
        onoff_bool_asso_ = np.concatenate([onoff_bool_asso[t_i], np.zeros(supply_onoff, dtype=bool)])
        on_t, off_t, onset_t, offset_t = get_onoff_cpts(onoff_bool_asso_)
        extend_t = 10 # ms; supply the length of window to count MUA during ON-OFF detection
        for oft in offset_t:
            onoff_bool_asso_[oft:oft+extend_t] = True
                       
        bothON_ = np.logical_and(onoff_bool_sens_, onoff_bool_asso_)
        bothOFF_ = np.logical_not(np.logical_or(onoff_bool_sens_, onoff_bool_asso_))        
        
        
        
        if sameSampleSize:
            n_off = bothOFF_.sum()
            n_on = bothON_.sum()
            if n_off > n_on:
                bothOFF_[np.where(bothOFF_)[0][:n_off-n_on]] = False
            else:
                bothON_[np.where(bothON_)[0][:n_on-n_off]] = False
                
                
        bothON.append(bothON_)
        bothOFF.append(bothOFF_)
        
        for pb_i, pb in enumerate(passband):
            MUA1_filt, MUA2_filt, MUA1_hilb, MUA2_hilb = fqa.get_filt_hilb(MUA_1, MUA_2, pb, Fs = 1000, filterOrder = 8)
        
            
            #amp_phaseDiff = MUA1_hilb*np.conj(MUA2_hilb)
            hilb_bothON[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothON[t_i]], MUA2_hilb[ignTrans:][bothON[t_i]]]))
            hilb_bothOFF[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothOFF[t_i]], MUA2_hilb[ignTrans:][bothOFF[t_i]]]))
    
    for pb_i in range(len(passband)):
        hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
        hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
    
    cohe_ON, PLV_ON, cohe_OFF, PLV_OFF = \
        np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband)), np.zeros(len(passband))
    
    for pb_i in range(len(passband)):
        cohe_ON[pb_i], PLV_ON[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1])
        cohe_OFF[pb_i], PLV_OFF[pb_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1])
    
    if not surrogate:
        return cohe_ON ,PLV_ON, cohe_OFF, PLV_OFF
    else:
        cohe_ON_p, PLV_ON_p, cohe_OFF_p, PLV_OFF_p = \
            np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200]), np.zeros([len(passband),200])
        for perm_i in range(200):
            if perm_i%20 == 0: print('surrogate: %.1f%%'%(perm_i/2))
            
            hilb_bothON = [[] for _ in passband]
            hilb_bothOFF = [[] for _ in passband]

            for t_i, t in enumerate(dura):
                
                MUA_1_perm = np.random.permutation(MUA[t_i][0])
                MUA_2_perm = np.random.permutation(MUA[t_i][1])
                for pb_i, pb in enumerate(passband):
                    MUA1_filt, MUA2_filt, MUA1_hilb, MUA2_hilb = \
                        fqa.get_filt_hilb(MUA_1_perm, MUA_2_perm, \
                                          pb, Fs = 1000, filterOrder = 8)
                    hilb_bothON[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothON[t_i]], MUA2_hilb[ignTrans:][bothON[t_i]]]))
                    hilb_bothOFF[pb_i].append(np.vstack([MUA1_hilb[ignTrans:][bothOFF[t_i]], MUA2_hilb[ignTrans:][bothOFF[t_i]]]))
            
            for pb_i in range(len(passband)):
            
                hilb_bothON[pb_i] = np.concatenate(hilb_bothON[pb_i], 1)
                hilb_bothOFF[pb_i] = np.concatenate(hilb_bothOFF[pb_i], 1)
            
                cohe_ON_p[pb_i, perm_i], PLV_ON_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothON[pb_i][0], hilb_bothON[pb_i][1])
                cohe_OFF_p[pb_i, perm_i], PLV_OFF_p[pb_i, perm_i], _ = fqa.get_coherence_phaseLockValue(hilb_bothOFF[pb_i][0], hilb_bothOFF[pb_i][1])
        
        return cohe_ON ,PLV_ON, cohe_OFF, PLV_OFF, \
            cohe_ON_p, PLV_ON_p, cohe_OFF_p, PLV_OFF_p
            
'''        













