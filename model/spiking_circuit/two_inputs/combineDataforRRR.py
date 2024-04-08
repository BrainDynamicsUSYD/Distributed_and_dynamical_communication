# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:00 2024

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
dataAnaly_path = dataAnaly_dir # path to the SUA data of each network

repeat = 30 # number of random network realizations

data_anly = mydata.mydata()

name_sfx = '' # 
unsyncType = 'bothOff'  #  
mua_range = 5
substract_mean = True
stn = 'st2'
for spk_posi in ['ctr']: # 'cor' ctr
    for sfx in ['ctr']:
        for get_att in [0, 1]:
            for analy_Sync in [0, 1]:
                print('spk_posi:',spk_posi, 'sfx:',sfx, 'get_att:',get_att, 'analy_Sync:',analy_Sync)

                if analy_Sync: sync_n = 'sync'
                else: 
                    if unsyncType == 'eitherOff':
                        sync_n = 'unsyncEtr'
                    else:
                        sync_n = 'unsync'
                        
                if get_att: att_n = 'att'
                else: att_n = 'noatt'


                MUA_1_all_net_ = np.empty(repeat, dtype=object)
                MUA_2_all_net_ = np.empty(repeat, dtype=object)

                
                for loop_num in range(0, repeat):
                    # pyfile_name = '%s_rg%d_%ssua_%s_local_%s%s_%d.file'%(att_n, mua_range,spk_posi,sync_n, sfx, name_sfx, loop_num)
                    pyfile_name = '%s_rg%d_%ssua_%s_local_subM%d_%d.file'%(att_n, mua_range,spk_posi,sync_n,  substract_mean, loop_num)
                    data_anly.load(dataAnaly_path+pyfile_name) # data_anly_onoff_thres_cor; data_anly_onoff_thres
                                            
                    MUA_1_all_net_[loop_num%repeat] = data_anly.MUA_1_all.T
                    MUA_2_all_net_[loop_num%repeat] = data_anly.MUA_2_all.T
                
                                            
                spk_count = {'a1_rg%d_%ssua_%s_%s_lc_%s%s'%(mua_range,spk_posi,sync_n,att_n, sfx, name_sfx):MUA_1_all_net_, \
                             'a2_rg%d_%ssua_%s_%s_lc_%s%s'%(mua_range,spk_posi,sync_n,att_n, sfx, name_sfx):MUA_2_all_net_} # a2_mua_sync_ext_cor
                
                matfile_name = '%s_rg%d_%ssua_%s_local_subM%d_%s.mat'%(att_n, mua_range,spk_posi,sync_n, substract_mean, stn)

                sio.savemat('raw_data/'+matfile_name, spk_count)
