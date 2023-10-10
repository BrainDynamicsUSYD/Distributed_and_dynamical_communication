#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 21:48:48 2021

@author: shni2598
"""

import numpy as np
#from scipy.signal.windows import dpss
#%%

def get_sigCoherence_fft(sig_1, sig_2, analy_dura, tap_win, disgard_t=0, win=200):
    
    # used multitaper methods to achieve optimal spectral concentration; 
   
    
    sig_len = analy_dura[0,1] - analy_dura[0,0] # ms
    sample_t = np.arange(disgard_t, sig_len+1, win)
    
    # nw = 2.5 # time half bandwidth: nw/sig_len(in second); 
    # Kmax = round(2*nw - 1)
    # dpss_w = dpss(win, NW=nw, Kmax=Kmax, sym=False, norm=None, return_ratios=False) # use Slepian functions
    
    if len(tap_win.shape) == 1:
        tap_win = tap_win.reshape(1,-1)
    
    xspect = []
    spect1 = []
    spect2 = []
    for dura in analy_dura:
        
        for t in sample_t[:-1]:
            
            xspect_, spect1_, spect2_ = get_spect_xspect(sig_1[dura[0]+t:dura[0]+t+win], sig_2[dura[0]+t:dura[0]+t+win], tap_win)
            xspect.append(xspect_)
            spect1.append(spect1_)
            spect2.append(spect2_)
            
    xspect_m = np.array(xspect).mean(0)        
    spect1_m = np.array(spect1).mean(0)        
    spect2_m = np.array(spect2).mean(0)        
    
    
    cohe = np.abs(xspect_m/np.sqrt(spect1_m*spect2_m))     

    return cohe


#%%        

def get_spect_xspect(sig1, sig2, tap_win):
    
    tap_sig1 = tap_win*sig1
    sig1_fft = np.fft.rfft(tap_sig1)
    
    
    tap_sig2 = tap_win*sig2
    sig2_fft = np.fft.rfft(tap_sig2)
    
    xspect = (sig1_fft*np.conj(sig2_fft)).mean(0)
    spect1 = (np.abs(sig1_fft)**2).mean(0)
    spect2 = (np.abs(sig2_fft)**2).mean(0)
    
    return xspect, spect1,  spect2