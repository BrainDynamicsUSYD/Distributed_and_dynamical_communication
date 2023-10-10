#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:54:53 2021

@author: shni2598
"""

import firing_rate_analysis as fra
import numpy as np
#sfrom sklearn.linear_model import LinearRegression
from scipy.stats import sem

#import matplotlib.pyplot as plt

#%%

class fano_no_mean_match:
    
    def __init__(self):
        
        self.spk_sparmat = None # sparse matrix; time and neuron index of spikes
        self.stim_onoff = None # 2-D array; ms; on and off time of each stimulus # data.a1.param.stim.stim_on[20:40].copy() # ms stimulus onset, off       
        self.t_bf = 100 # ms; time before stimulus onset
        self.t_aft = 100 # ms; time after stimulus off
        self.dt = 0.1 # ms; simulation time step
        self.win = 150 # ms sliding window
        self.move_step = 10 # ms sliding window moving step
        self.n_perStimAmp = 50
        # self.bin_count_interval = 1 # number of spikes; interval of bin used to calculate the histogram of mean spike counts 
        # self.repeat = 100 # repeat times in 'mean-matching'
        # self.method = 'regression' # 'mean' or 'regression'
        
    
    def get_fano(self):
        
        dura = self.stim_onoff.copy()
        dura[:,0] -= round(self.t_bf + self.win/2)
        dura[:,1] += round(self.t_aft + self.win/2)
        
        n_block = dura.shape[0]//self.n_perStimAmp
        
        
        for st_i in range(n_block):
            spk = fra.get_spkcount_sparmat_multi(self.spk_sparmat, dura[st_i*self.n_perStimAmp:(st_i+1)*self.n_perStimAmp], sum_activity=False, \
                       sample_interval = self.move_step,  window = self.win, dt = self.dt)
        
            m = np.nanmean(spk, 0)
            var = np.nanvar(spk, 0)
            
            # plt.figure()
            # plt.plot(np.nanmean(m,0))
            
            fano = var/m
            
            fano_m = np.nanmean(fano, 0)
            fano_sem = sem(fano, 0, nan_policy='omit')
            
            if st_i == 0:
                fano_m_sem = np.zeros([n_block, fano_m.shape[0], 2])
            
            fano_m_sem[st_i, :, 0] = fano_m
            fano_m_sem[st_i, :, 1] = fano_sem
        
        return fano_m_sem
    




