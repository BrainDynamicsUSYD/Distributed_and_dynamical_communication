#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:26:48 2022

@author: shni2598
"""



import numpy as np
import firing_rate_analysis as fra
import connection as cn

#%%

def detect_target_T(spk_mat_1, spk_mat_2, stim_posi, stimNum_reach_thre_min, trg_exist, \
                    e_lattice, analy_dura, mua_win, threshold):

    # threshold = {'bottom':[6000,8000,10000],
    #              'top':[6000,8000,10000],
    #              'both':[6000,8000,10000],}

    mua_neuron = []
    mua_range = 5 
    width = 64
    for posi in stim_posi:
        mua_neuron.append(cn.findnearbyneuron.findnearbyneuron(e_lattice, posi, mua_range, width))
    
    # print(mua_neuron)
    
    n_stim = len(stim_posi)
    detect_t={}
    for key in threshold.keys():
        detect_t[key] = np.zeros([len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    
    for tri, dura in enumerate(analy_dura):
        mua1, mua2 = get_mua_multigroup(spk_mat_1, spk_mat_2, mua_neuron, dura[0], dura[1], mua_win)
        '''bottom'''
        thre = np.array(threshold['bottom'])
        thre_sortarg = np.argsort(thre)
        thre_sort = thre[thre_sortarg]
        
        detect_t_i = np.zeros([n_stim, len(thre)])
        detect_t_i[:] = np.nan
        
        for sti, mua in enumerate(mua1):
                        
            fr_sum = 0
            thre_i = 0
            for tt in range(mua.shape[0]):
                fr_sum += mua[tt]
                while fr_sum >= thre_sort[thre_i]:
                    detect_t_i[sti, thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
                    # print(thre[thre_i])
                    thre_i += 1
                    
                    if thre_i == len(thre):
                        break
                else: continue
                break
        
        detect_t['bottom'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
        
        '''top'''
        thre = np.array(threshold['top'])
        thre_sortarg = np.argsort(thre)
        thre_sort = thre[thre_sortarg]
        
        detect_t_i = np.zeros([n_stim, len(thre)])
        detect_t_i[:] = np.nan
        
        for sti, mua in enumerate(mua2):
                        
            fr_sum = 0
            thre_i = 0
            for tt in range(mua.shape[0]):
                fr_sum += mua[tt]
                while fr_sum >= thre_sort[thre_i]:
                    detect_t_i[sti, thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
                    thre_i += 1
                    
                    if thre_i == len(thre):
                        break
                else: continue
                break        
        
        detect_t['top'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
            
        '''both'''
        thre = np.array(threshold['both'])
        thre_sortarg = np.argsort(thre)
        thre_sort = thre[thre_sortarg]
        
        detect_t_i = np.zeros([n_stim, len(thre)])
        detect_t_i[:] = np.nan
        
        for sti, mua in enumerate(zip(mua1,mua2)):
            
            mua = np.sqrt(mua[0]*mua[1])
            fr_sum = 0
            thre_i = 0
            for tt in range(mua.shape[0]):
                fr_sum += mua[tt]
                while fr_sum >= thre_sort[thre_i]:
                    detect_t_i[sti, thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
                    thre_i += 1
                    
                    if thre_i == len(thre):
                        break
                else: continue
                break        
        
        detect_t['both'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)        
        
        
    return detect_t
        
def detect_target_T_diffusion(spk_mat_1, spk_mat_2, stim_posi, stimNum_reach_thre_min, trg_exist, \
                    e_lattice, analy_dura, mua_win, threshold, tau, scale_fr, dt):

    # threshold = {'bottom':[6000,8000,10000],
    #              'top':[6000,8000,10000],
    #              'both':[6000,8000,10000],}

    mua_neuron = []
    mua_range = 5 
    width = 64
    for posi in stim_posi:
        mua_neuron.append(cn.findnearbyneuron.findnearbyneuron(e_lattice, posi, mua_range, width))
    
    # print(mua_neuron)
    
    # n_stim = len(stim_posi)
    detect_t={}
    for key in threshold.keys():
        detect_t[key] = np.zeros([len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    
    for tri, dura in enumerate(analy_dura):
        mua1, mua2 = get_mua_multigroup(spk_mat_1, spk_mat_2, mua_neuron, dura[0]-mua_win//2, dura[1]+mua_win//2, mua_win)
        '''bottom'''
        thre = np.array(threshold['bottom'])
        
        detect_t_i = detect_singleTrial_trg_diffusion(mua1, thre, tau, scale_fr, dt)
        
        detect_t['bottom'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
        
        '''top'''
        thre = np.array(threshold['top'])
    
        detect_t_i = detect_singleTrial_trg_diffusion(mua2, thre, tau, scale_fr, dt)   
        
        detect_t['top'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
            
        '''both'''
        thre = np.array(threshold['both'])
        
        mua1 = np.array(mua1)
        mua2 = np.array(mua2)
        detect_t_i = detect_singleTrial_trg_diffusion(np.sqrt(mua1*mua2), thre, tau, scale_fr, dt)          
        
        detect_t['both'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)        
                
    return detect_t

#%%
# mua_all = np.zeros([1,2000])

# for ii in range(0,2000,200):
#     mua_all[0,ii:ii+100] = 50

# plt.figure()
# plt.plot(mua_all[0]) 

# #%%

# #np.random.randn(100).reshape(1,100)*10
# tau = 1000
# k = 0
# dt = 1 
# k_t = np.zeros(2000)
# for tt in range(mua_all[0].shape[0]):
#     k += (-k + mua_all[0][tt])/tau*dt
#     k_t[tt] = k
    
# plt.figure()
# plt.plot(k_t) 

# #%%

# #np.random.randn(100).reshape(1,100)*10
# tau = 5000
# k = 0
# dt = 1 
# k_t = np.zeros(2000)
# scale_fr = 1
# for tt in range(mua_all[0].shape[0]):
#     # k += (-k)/tau*dt + mua_all[0][tt]/1000*dt
#     # k += (-k/tau + mua_all[0][tt]/1000)*dt
#     k += (-k/tau + scale_fr*mua_all[0][tt]/1000)*dt       

#     k_t[tt] = k
    
# plt.figure()
# plt.plot(k_t) 

#%%
def detect_singleTrial_trg_diffusion(mua, thre, tau, scale_fr, dt):
    
    # detect_t_i = np.zeros([len(thre)])
    n_stim = len(mua)
    detect_t_i = np.zeros([n_stim, len(thre)])

    detect_t_i[:] = np.nan
    
    thre_sortarg = np.argsort(thre)
    thre_sort = thre[thre_sortarg]
    
    
    for sti, mua_i in enumerate(mua):
        thre_i = 0   
        k = 0
               
        for tt in range(mua_i.shape[0]):
            # k += (-k + scale_fr*mua_i[tt]/1000)/tau*dt       
            k += (-k/tau + scale_fr*mua_i[tt]/1000)*dt       

            while k >= thre_sort[thre_i]:
                detect_t_i[sti, thre_sortarg[thre_i]] = tt  # 0.5: half mua_win
                # print(thre[thre_i])
                thre_i += 1
                
                if thre_i == len(thre):
                    break
            else: continue
            break
    
    return detect_t_i


def get_mua_multigroup(spk_mat_1, spk_mat_2, mua_neuron, start, end, mua_win):
    
    mua1 = []
    mua2 = []
    for mua_n in mua_neuron:
        mua1.append(fra.get_spkcount_sum_sparmat(spk_mat_1[mua_n], start, end,\
                               sample_interval = 1,  window = mua_win, dt = 0.1)/mua_n.shape[0]/(mua_win/1000))
    
        mua2.append(fra.get_spkcount_sum_sparmat(spk_mat_2[mua_n], start, end,\
                               sample_interval = 1,  window = mua_win, dt = 0.1)/mua_n.shape[0]/(mua_win/1000))
            
    return mua1, mua2
    
    
def detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist):
    
    if trg_exist:
        
        return np.vstack([detect_t_i[:1], np.sort(detect_t_i[1:], 0)[:stimNum_reach_thre_min-1]]).max(0)
    
    else:
        
        return np.sort(detect_t_i[:], 0)[:stimNum_reach_thre_min].max(0)
    
    
    
    
def detect_target_T_distractorAccuf(spk_mat_1, spk_mat_2, stim_posi,  \
                    e_lattice, analy_dura, mua_win, threshold):

    # threshold = {'bottom':[6000,8000,10000],
    #              'top':[6000,8000,10000],
    #              'both':[6000,8000,10000],}

    mua_neuron = []
    mua_range = 5 
    width = 64
    for posi in stim_posi:
        mua_neuron.append(cn.findnearbyneuron.findnearbyneuron(e_lattice, posi, mua_range, width))
    
    # print(mua_neuron)
    
    n_stim = len(stim_posi)
    detect_t={}
    for key in threshold.keys():
        detect_t[key] = np.zeros([len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    distractor_accuf = {}
    for key in threshold.keys():
        distractor_accuf[key] = np.zeros([n_stim-1, len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    
    for tri, dura in enumerate(analy_dura):
        mua1, mua2 = get_mua_multigroup(spk_mat_1, spk_mat_2, mua_neuron, dura[0], dura[1], mua_win)
        mua1 = np.array(mua1)
        mua2 = np.array(mua2)
        
        '''bottom'''
        thre = threshold['bottom']
        
        detect_t_i = detect_singleTrial_trg(mua1, thre, mua_win)
        
        # detect_t_i = np.zeros([len(thre)])
        # detect_t_i[:] = np.nan
        
        # # for sti, mua in enumerate(mua1):
                        
        # fr_sum = 0
        # thre_i = 0
        # for tt in range(mua1[0].shape[0]):
        #     fr_sum += mua1[0][tt]
        #     if fr_sum >= thre[thre_i]:
        #         detect_t_i[thre_i] = tt + 0.5*mua_win # 0.5: half mua_win
        #         # print(thre[thre_i])
        #         thre_i += 1
                
        #         if thre_i == len(thre):
        #             break
        
        detect_t['bottom'][:,tri] = detect_t_i #detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
        
        for dtt_i, dtt in enumerate(detect_t_i):
            if np.isnan(dtt):
                distractor_accuf['bottom'][:,dtt_i,tri] = np.nan
            else:
                distractor_accuf['bottom'][:,dtt_i,tri] = mua1[1:,:round(dtt - 0.5*mua_win)].sum(1)
        
        '''top'''
        thre = threshold['top']

        detect_t_i = detect_singleTrial_trg(mua2, thre, mua_win)
                
        detect_t['top'][:,tri] = detect_t_i #detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)

        for dtt_i, dtt in enumerate(detect_t_i):
            if np.isnan(dtt):
                distractor_accuf['top'][:,dtt_i,tri] = np.nan
            else:
                distractor_accuf['top'][:,dtt_i,tri] = mua2[1:,:round(dtt - 0.5*mua_win)].sum(1)
            
        '''both'''
        thre = threshold['both']
        
        mua_geomean = np.sqrt(mua1*mua2)
        detect_t_i = detect_singleTrial_trg(mua_geomean, thre, mua_win)  
        
        detect_t['both'][:,tri] = detect_t_i #detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)        

        for dtt_i, dtt in enumerate(detect_t_i):
            if np.isnan(dtt):
                distractor_accuf['both'][:,dtt_i,tri] = np.nan
            else:
                distractor_accuf['both'][:,dtt_i,tri] = mua_geomean[1:,:round(dtt - 0.5*mua_win)].sum(1)
        
        
    return detect_t, distractor_accuf

    
def detect_singleTrial_trg(mua_all, thre, mua_win):
    
    '''target must be the first row in mua_all'''
    detect_t_i = np.zeros([len(thre)])
    detect_t_i[:] = np.nan
    
    thre_sortarg = np.argsort(thre)
    thre_sort = thre[thre_sortarg]

    fr_sum = 0
    thre_i = 0
    for tt in range(mua_all[0].shape[0]):
        fr_sum += mua_all[0][tt]
        while fr_sum >= thre_sort[thre_i]:
            detect_t_i[thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
            # print(thre[thre_i])
            thre_i += 1
            
            if thre_i == len(thre):
                break
        else: continue
        break
    
    return detect_t_i


#%%
def detect_target_T_diffusion_distractorAccuf(spk_mat_1, spk_mat_2, stim_posi,  \
                    e_lattice, analy_dura, mua_win, threshold, tau, scale_fr, dt):

    # threshold = {'bottom':[6000,8000,10000],
    #              'top':[6000,8000,10000],
    #              'both':[6000,8000,10000],}

    mua_neuron = []
    mua_range = 5 
    width = 64
    for posi in stim_posi:
        mua_neuron.append(cn.findnearbyneuron.findnearbyneuron(e_lattice, posi, mua_range, width))
    
    # print(mua_neuron)
    
    n_stim = len(stim_posi)
    detect_t={}
    for key in threshold.keys():
        detect_t[key] = np.zeros([len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    distractor_accuf = {}
    for key in threshold.keys():
        distractor_accuf[key] = np.zeros([n_stim-1, len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    
    for tri, dura in enumerate(analy_dura):
        mua1, mua2 = get_mua_multigroup(spk_mat_1, spk_mat_2, mua_neuron, dura[0]-mua_win//2, dura[1]+mua_win//2, mua_win)
        mua1 = np.array(mua1)
        mua2 = np.array(mua2)
        
        '''bottom'''
        thre = np.array(threshold['bottom'])
        
        detect_t_i, distrAcf = detect_singleTrial_trg_diffusion_distractorAccuf(mua1, thre, tau, scale_fr, dt)
        
        # detect_t_i = np.zeros([len(thre)])
        # detect_t_i[:] = np.nan
        
        # # for sti, mua in enumerate(mua1):
                        
        # fr_sum = 0
        # thre_i = 0
        # for tt in range(mua1[0].shape[0]):
        #     fr_sum += mua1[0][tt]
        #     if fr_sum >= thre[thre_i]:
        #         detect_t_i[thre_i] = tt + 0.5*mua_win # 0.5: half mua_win
        #         # print(thre[thre_i])
        #         thre_i += 1
                
        #         if thre_i == len(thre):
        #             break
        
        detect_t['bottom'][:,tri] = detect_t_i #detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)

        distractor_accuf['bottom'][:,:,tri] = distrAcf
        
        # for dtt_i, dtt in enumerate(detect_t_i):
        #     if np.isnan(dtt):
        #         distractor_accuf['bottom'][:,dtt_i,tri] = np.nan
        #     else:
        #         distractor_accuf['bottom'][:,dtt_i,tri] = mua1[1:,:round(dtt - 0.5*mua_win)].sum(1)
        
        '''top'''
        thre = np.array(threshold['top'])

        detect_t_i, distrAcf = detect_singleTrial_trg_diffusion_distractorAccuf(mua2, thre, tau, scale_fr, dt)
                    
        detect_t['top'][:,tri] = detect_t_i #detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)

        distractor_accuf['top'][:,:,tri] = distrAcf

            
        '''both'''
        thre = np.array(threshold['both'])
        
        mua_geomean = np.sqrt(mua1*mua2)
        detect_t_i, distrAcf = detect_singleTrial_trg_diffusion_distractorAccuf(mua_geomean, thre, tau, scale_fr, dt)
        
        detect_t['both'][:,tri] = detect_t_i #detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)        

        distractor_accuf['both'][:,:,tri] = distrAcf        
        
    return detect_t, distractor_accuf

#%%
def detect_singleTrial_trg_diffusion_distractorAccuf(mua, thre, tau, scale_fr, dt):
    
    '''Target firing rate must be the first row in 'mua' '''
    # detect_t_i = np.zeros([len(thre)])
    n_stim = len(mua)
    detect_t_i = np.zeros(len(thre))

    detect_t_i[:] = np.nan

    distrAcf = np.zeros([n_stim-1, len(thre)])
    distrAcf[:] = np.nan
    
    thre_sortarg = np.argsort(thre)
    thre_sort = thre[thre_sortarg]
    
    
    # for sti, mua_i in enumerate(mua):
    thre_i = 0   
    k = np.zeros(n_stim)
           
    for tt in range(mua.shape[1]):
        # k += (-k + scale_fr*mua_i[tt]/1000)/tau*dt       
        k += (-k/tau + scale_fr*mua[:,tt]/1000)*dt       

        while k[0] >= thre_sort[thre_i]:
            detect_t_i[thre_sortarg[thre_i]] = tt  # 0.5: half mua_win
            distrAcf[:, thre_sortarg[thre_i]] = k[1:]
            # print(thre[thre_i])
            thre_i += 1
            
            if thre_i == len(thre):
                break
        else: continue
        break
    
    return detect_t_i, distrAcf


#%%
def getRandThreFromCDF(thre_perc):
    
    # thre = np.zeros(len(thre_perc))
    if len(thre_perc.shape) == 1:
        thre_perc = thre_perc.reshape(1,-1)
    
    uni_rand = np.random.rand(len(thre_perc))/(1/(thre_perc.shape[1] - 1))
    
    # uni_rand = np.random.rand(len(thre_perc))*100
    uni_rand_floor = np.floor(uni_rand).astype(int)
    uni_rand_ceil = np.ceil(uni_rand).astype(int)
    
    row_ind = np.arange(len(thre_perc))
    
    thre = thre_perc[row_ind, uni_rand_floor] + (thre_perc[row_ind, uni_rand_ceil] - thre_perc[row_ind, uni_rand_floor]) * (uni_rand%1)
    
    # thre = (thre_perc[row_ind, uni_rand_floor] + thre_perc[row_ind, uni_rand_ceil])/2
    
    return thre

#%%
def detect_singleTrial_multiSt_diffusion(mua_all, thre_perc,  tau, scale_fr, dt, sameThreForEachSt = False):
    
    # '''target must be the first row in mua_all'''
    if sameThreForEachSt:
        thre = getRandThreFromCDF(thre_perc)
        
        thre_sortarg = np.argsort(thre)
        thre_sort = thre[thre_sortarg]
        
        detect_t_i = np.zeros([len(mua_all), len(thre)])
        detect_t_i[:] = np.nan
    
        for sti, mua in enumerate(mua_all):
                        
            thre_i = 0
            k = 0
            for tt in range(mua.shape[0]):
                k += (-k/tau + scale_fr*mua[tt]/1000)*dt       
                while k >= thre_sort[thre_i]:
                    detect_t_i[sti, thre_sortarg[thre_i]] = tt # 0.5: half mua_win
                    # print(thre[thre_i])
                    thre_i += 1
                    
                    if thre_i == len(thre):
                        break
                else: continue
                break
    else:        
        detect_t_i = np.zeros([len(mua_all), len(thre_perc)])
        detect_t_i[:] = np.nan
    
        for sti, mua in enumerate(mua_all):

            thre = getRandThreFromCDF(thre_perc)                
            thre_sortarg = np.argsort(thre)
            thre_sort = thre[thre_sortarg]                        

            thre_i = 0
            k = 0
            for tt in range(mua.shape[0]):
                k += (-k/tau + scale_fr*mua[tt]/1000)*dt       
                while k >= thre_sort[thre_i]:
                    detect_t_i[sti, thre_sortarg[thre_i]] = tt # 0.5: half mua_win
                    # print(thre[thre_i])
                    thre_i += 1
                    
                    if thre_i == len(thre):
                        break
                else: continue
                break    
                
    return detect_t_i    



def detect_singleTrial_multiSt(mua_all, thre_perc, mua_win, sameThreForEachSt = False):
    
    # '''target must be the first row in mua_all'''
    if sameThreForEachSt:
        thre = getRandThreFromCDF(thre_perc)
        
        thre_sortarg = np.argsort(thre)
        thre_sort = thre[thre_sortarg]
        
        detect_t_i = np.zeros([len(mua_all), len(thre)])
        detect_t_i[:] = np.nan
    
        for sti, mua in enumerate(mua_all):
                        
            fr_sum = 0
            thre_i = 0
            for tt in range(mua.shape[0]):
                fr_sum += mua[tt]
                while fr_sum >= thre_sort[thre_i]:
                    detect_t_i[sti, thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
                    # print(thre[thre_i])
                    thre_i += 1
                    
                    if thre_i == len(thre):
                        break
                else: continue
                break
    else:        
        detect_t_i = np.zeros([len(mua_all), len(thre_perc)])
        detect_t_i[:] = np.nan
    
        for sti, mua in enumerate(mua_all):

            thre = getRandThreFromCDF(thre_perc)                
            thre_sortarg = np.argsort(thre)
            thre_sort = thre[thre_sortarg]                        
            fr_sum = 0
            thre_i = 0
            for tt in range(mua.shape[0]):
                fr_sum += mua[tt]
                while fr_sum >= thre_sort[thre_i]:
                    detect_t_i[sti, thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
                    # print(thre[thre_i])
                    thre_i += 1
                    
                    if thre_i == len(thre):
                        break
                else: continue
                break    
                
    return detect_t_i    

#%%
def detect_target_T_randThrefromCDF_diffusion(spk_mat_1, spk_mat_2, stim_posi, stimNum_reach_thre_min, trg_exist, \
                    e_lattice, analy_dura, mua_win, threshold, sameThreForEachSt, tau, scale_fr, dt):

    # threshold = {'bottom':[6000,8000,10000],
    #              'top':[6000,8000,10000],
    #              'both':[6000,8000,10000],}

    mua_neuron = []
    mua_range = 5 
    width = 64
    for posi in stim_posi:
        mua_neuron.append(cn.findnearbyneuron.findnearbyneuron(e_lattice, posi, mua_range, width))
    
    # print(mua_neuron)
    
    # n_stim = len(stim_posi)
    detect_t={}
    for key in threshold.keys():
        detect_t[key] = np.zeros([len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    
    for tri, dura in enumerate(analy_dura):
        mua1, mua2 = get_mua_multigroup(spk_mat_1, spk_mat_2, mua_neuron, dura[0]-mua_win//2, dura[1]+mua_win//2, mua_win)
        mua1 = np.array(mua1)
        mua2 = np.array(mua2)

        '''bottom'''
        thre_perc = threshold['bottom']
        
        # thre = getRandThreFromCDF(thre_perc)

        detect_t_i = detect_singleTrial_multiSt_diffusion(mua1, thre_perc, tau, scale_fr, dt, sameThreForEachSt)
        
        detect_t['bottom'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
        
        '''top'''
        thre_perc = threshold['top']
        
        # thre = getRandThreFromCDF(thre_perc)

        detect_t_i = detect_singleTrial_multiSt_diffusion(mua2, thre_perc, tau, scale_fr, dt, sameThreForEachSt)
        
        detect_t['top'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
            
        '''both'''
        thre_perc = threshold['both']
        
        # thre = getRandThreFromCDF(thre_perc)

        detect_t_i = detect_singleTrial_multiSt_diffusion(np.sqrt(mua1*mua2), thre_perc, tau, scale_fr, dt, sameThreForEachSt)        
        
        detect_t['both'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)        
        
        
    return detect_t   


#%%
    

def detect_target_T_randThrefromCDF(spk_mat_1, spk_mat_2, stim_posi, stimNum_reach_thre_min, trg_exist, \
                    e_lattice, analy_dura, mua_win, threshold, sameThreForEachSt):

    # threshold = {'bottom':[6000,8000,10000],
    #              'top':[6000,8000,10000],
    #              'both':[6000,8000,10000],}

    mua_neuron = []
    mua_range = 5 
    width = 64
    for posi in stim_posi:
        mua_neuron.append(cn.findnearbyneuron.findnearbyneuron(e_lattice, posi, mua_range, width))
    
    # print(mua_neuron)
    
    # n_stim = len(stim_posi)
    detect_t={}
    for key in threshold.keys():
        detect_t[key] = np.zeros([len(threshold[key]), analy_dura.shape[0]], dtype=float)
    
    
    for tri, dura in enumerate(analy_dura):
        mua1, mua2 = get_mua_multigroup(spk_mat_1, spk_mat_2, mua_neuron, dura[0], dura[1], mua_win)
        mua1 = np.array(mua1)
        mua2 = np.array(mua2)

        '''bottom'''
        thre_perc = threshold['bottom']
        
        # thre = getRandThreFromCDF(thre_perc)

        detect_t_i = detect_singleTrial_multiSt(mua1, thre_perc, mua_win, sameThreForEachSt)
        
        detect_t['bottom'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
        
        '''top'''
        thre_perc = threshold['top']
        
        # thre = getRandThreFromCDF(thre_perc)

        detect_t_i = detect_singleTrial_multiSt(mua2, thre_perc, mua_win, sameThreForEachSt)
        
        detect_t['top'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)
            
        '''both'''
        thre_perc = threshold['both']
        
        # thre = getRandThreFromCDF(thre_perc)

        detect_t_i = detect_singleTrial_multiSt(np.sqrt(mua1*mua2), thre_perc, mua_win, sameThreForEachSt)        
        
        detect_t['both'][:,tri] = detect_singleTrial(detect_t_i, stimNum_reach_thre_min, trg_exist) #detect_t_i.max(0)        
        
        
    return detect_t    
    
    
# #%%
# thre_path = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/project2/data/targetseacrh/newparam/'

# data_thre = mydata.mydata()
# data_thre.load(thre_path + "data_varythreStN_for_notrg_C_percentile.file")

# choose_area = ['bottom', 'top', 'both']
# threshold_fr = {}

# # for area in choose_area:
# #     threshold_fr[area] = data_thre.distAccuf_stnum[0][area][:,:,0,stim_num-1].reshape(-1)

# #%%
# stim_num = 2
# for area in choose_area:
#     row = data_thre.distAccuf_stnum_perc[0][area].shape[0] * data_thre.distAccuf_stnum_perc[0][area].shape[1] 
#     threshold_fr[area] = data_thre.distAccuf_stnum_perc[0][area][:,:,stim_num-1,:].reshape(row, -1)

# #%%

# plt.figure()
# plt.plot(distAccuf_perc_att_trg_c[0]['bottom'][2][0][2])

# #%%
# pdf = np.diff(np.arange(101))/np.diff(distAccuf_perc_att_trg_c[0]['bottom'][2][0][2])

# plt.figure()
# plt.plot((distAccuf_perc_att_trg_c[0]['bottom'][2][0][2][1:]+distAccuf_perc_att_trg_c[0]['bottom'][2][0][2][:-1])/2, pdf)
# #%%
# thre = []
# for i in range(10000):
#     thre.append(getRandThreFromCDF(threshold_fr['bottom'])[7])

# thre = np.array(thre)
# #%%
# plt.figure()
# plt.hist(thre,20)
# #%%
# np.random.seed(10)
# getRandThreFromCDF(distAccuf_perc_att_trg_c[0]['bottom'][2][0][2])[0]

# #%%
# thre = []
# for i in range(10000):
#     thre.append(getRandThreFromCDF(distAccuf_perc_att_trg_c[0]['bottom'][:,:,1].reshape(12,-1)))

# thre = np.array(thre)
# #%%

# plt.figure()
# plt.hist(thre[:,6], 20, density=1)


# #%% 
# thre = getRandThreFromCDF(distAccuf_perc_att_trg_c[0]['bottom'][:,:,1].reshape(12,-1))
# thre_sortarg = np.argsort(thre)
# thre_sort = thre[thre_sortarg]
# #%%
# mua1 = np.tile(np.arange(0, 15000, 100),2).reshape(2,-1)
# mua1[1] += 500

# detect_t_i = np.zeros([2, len(thre)])
# detect_t_i[:] = np.nan

# mua_win = 1
# for sti, mua in enumerate(mua1):
                
#     fr_sum = 0
#     thre_i = 0
#     for tt in range(mua.shape[0]):
#         fr_sum += mua[tt]
#         while fr_sum >= thre_sort[thre_i]:
#             detect_t_i[sti, thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
#             # print(thre[thre_i])
#             thre_i += 1
            
#             if thre_i == len(thre):
#                 break
#         else: continue
#         break
# # for sti, mua in enumerate(mua1):
                
# #     fr_sum = 0
# #     thre_i = 0
# #     for tt in range(mua.shape[0]):
# #         fr_sum += mua[tt]
# #         if fr_sum >= thre_sort[thre_i]:
# #             detect_t_i[sti, thre_sortarg[thre_i]] = tt + 0.5*mua_win # 0.5: half mua_win
# #             # print(thre[thre_i])
# #             thre_i += 1
            
# #             if thre_i == len(thre):
# #                 break    
# #%%
# detect_t_i_2 = detect_singleTrial_multiSt(mua1, thre, mua_win=1)    
    