# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:20:23 2024

@author: Shencong Ni
"""

'''
Combine the SUA data from each network realization obtained by 'extractDataforRRRforEachNet.py'
The Combined data will be used in reduced rank regression (RRR) analysis
Run 'onoff_detection.py' and 'extractDataforRRRforEachNet.py' first before running this script.
'''

import mydata
import numpy as np

import scipy.io as sio

#%%
dataAnaly_dir = 'raw_data/'
dataAnaly_path = dataAnaly_dir

repeat = 1 # number of total random network realizations

data_anly = mydata.mydata()

name_sfx = '' # Evt_30ovlp
unsyncType = 'bothOff'  # eitherOff bothOff
mua_range = 5
substract_mean = True
for spk_posi in ['ctr']: # 'cor'
    for sfx in ['ctr']:
        for analy_Sync in [0, 1]:
            print('spk_posi:',spk_posi, 'sfx:',sfx, 'analy_Sync:',analy_Sync)

            if analy_Sync: sync_n = 'sync'
            else: 
                if unsyncType == 'eitherOff':
                    sync_n = 'unsyncEtr'
                else:
                    sync_n = 'unsync'
                    



            MUA_1_all_net_ = np.empty(repeat, dtype=object)
            MUA_2_all_net_ = np.empty(repeat, dtype=object)

            
            for loop_num in range(0, repeat):
                # pyfile_name = 'rg%d_%ssua_%s_local_%s%s_%d.file'%(mua_range,spk_posi,sync_n, sfx, name_sfx, loop_num)
                pyfile_name = 'rg%d_%ssua_%s_local_subM%d_%d.file'%(mua_range,spk_posi,sync_n,  substract_mean, loop_num)

                data_anly.load(dataAnaly_path+pyfile_name) # data_anly_onoff_thres_cor; data_anly_onoff_thres
                                        
                MUA_1_all_net_[loop_num%repeat] = data_anly.MUA_1_all.T
                MUA_2_all_net_[loop_num%repeat] = data_anly.MUA_2_all.T
            
                        
            
            spk_count = {'a1_rg%d_%ssua_%s_spon_lc_%s%s'%(mua_range,spk_posi,sync_n, sfx, name_sfx):MUA_1_all_net_, \
                         'a2_rg%d_%ssua_%s_spon_lc_%s%s'%(mua_range,spk_posi,sync_n, sfx, name_sfx):MUA_2_all_net_} # a2_mua_sync_ext_cor
            
            # matfile_name = 'spon_rg%d_%ssua_%s_local_%s%s.mat'%(mua_range,spk_posi,sync_n, sfx, name_sfx)
            matfile_name = 'spon_rg%d_%ssua_%s_local_subM%d_comb.mat'%(mua_range,spk_posi,sync_n, substract_mean)
            
                #%
            sio.savemat('raw_data/'+matfile_name, spk_count)
        
