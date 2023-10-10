#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:30:28 2021

@author: shni2598
"""


import matplotlib as mpl
mpl.use('Agg')
from scipy.stats import sem
#import load_data_dict
import mydata
import numpy as np
# import brian2.numpy_ as np
# from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
import frequency_analysis as fqa
import coordination

import fano_mean_match
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil

import coordination
import poisson_stimuli as psti

#%%
cos_alpha = (5*7**0.5)/14
sin_alpha = (21**0.5)/14


rotate_matrix = np.array([[cos_alpha, -sin_alpha],
                          [sin_alpha, cos_alpha]])

aa=np.array([1,0]).reshape(-1,1)

aa_r = np.dot(rotate_matrix, aa)
#%%
plt.figure()
plt.plot([0,aa_r[0,0]], [0,aa_r[1,0]])
#%%

orig  = np.array([0,0])

st1 = np.array([orig[0] + 3**0.5/4, orig[1] + 3/4])

st2 = st1 + np.array([-3**0.5/2, 3/2])
st3 = st2 + np.array([-3**0.5/2, -3/2])
st4 = st3 + np.array([-3**0.5/2, -3/2])
st5 = st4 + np.array([3**0.5, 0])
st6 = st5 + np.array([3**0.5, 0])
st7 = st6 + np.array([3**0.5/2, 3/2])

st = np.array([st1, st2,st3,st4,st5,st6,st7])

st_r = np.dot(rotate_matrix, st.T).T

st_r[:,0] *= 64/np.sqrt(21)
st_r[:,1] *= 64/(np.sqrt(21)/2*3**0.5)

st_r[:,0] += ((1/4) * 64)
#%
hw = 32
st_r = ((st_r + hw)%(2*hw) - hw)


#%%
plt.figure()
for i in range(7):
    
    plt.plot([0,st_r[i,0]], [0,st_r[i,1]])

#%%
sti =  psti.input_spkrate(maxrate = [200]*7, sig=[5]*7, position=st_r).reshape(64,64)
plt.figure()
plt.imshow(sti)

#%%
stim_posi_list = [[0,0],[-64/3,64/3],[-64/3,-64/3],[64/3,-64/3],[64/3,64/3]]

coordination.lattice_dist(np.array(stim_posi_list), 64, stim_posi_list[1])

