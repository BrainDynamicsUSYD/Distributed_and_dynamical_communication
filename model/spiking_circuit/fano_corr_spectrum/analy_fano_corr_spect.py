#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:38:45 2021

@author: Shencong Ni
"""

'''
1. get the power spectrum and the theta-gamma coupling of spontaneous activity
2. analyse the top-down attention's effect on Fano factor and noise correlation

run 'two_inputs_simu.py' before running this script.
'''



import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
#import load_data_dict
import mydata
import numpy as np

import firing_rate_analysis as fra
import frequency_analysis as fqa
from cfc_analysis import cfc
import fano_mean_match
import connection as cn
import pywt
import sys
import matplotlib.pyplot as plt
#%%
data_dir = 'raw_data/'
datapath = data_dir
sys_argv = int(sys.argv[1])
loop_num = sys_argv #

savefile_name = 'data_anly_spect_fano_corr_' 
save_apd = ''
stim_loc_2 = np.array([-32, -32]) # location of the second input


save_img = True # if true, save figures
# get_movie = True # if true, make and save movie

fano_mua_range = 5
nc_mua_range = 5

save_analy_file = 1

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

title = ''
    
#%%

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

#%%
n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp

stim_amp = np.unique(data.a1.param.stim1.stim_amp_scale)*200
#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
'''A demo for calculating power spectrum of spontaneous MUA''' 

start_t = np.arange(5000, 9001, 1000)
end_t = start_t + 1000
analy_dura = np.array([start_t, end_t]).T
data_anly.a1 = mydata.mydata()
data_anly.a2 = mydata.mydata()

for area in ['a1', 'a2']:
    coef = []
    for dura in analy_dura:
        
        mua = fra.get_spkcount_sum_sparmat(data.__dict__[area].ge.spk_matrix[mua_neuron], dura[0], dura[1],\
                            sample_interval = 1,  window = 1, dt = 0.1)
            
        mua = mua/mua_neuron.shape[0]/(1/1000)  
        coef_, freq = fqa.myfft(mua, Fs=1000, amp=False, power=True)
        coef.append(np.abs(coef_))
    
    data_anly.__dict__[area].spon_coef = np.array(coef)
    data_anly.__dict__[area].spon_freq = freq

fig, ax = plt.subplots(1,2, figsize=[8,5])

freq_plt = (data_anly.a1.spon_freq > 0) & (data_anly.a1.spon_freq < 100)
ax[0].loglog(data_anly.a1.spon_freq[freq_plt], data_anly.a1.spon_coef.mean(0)[freq_plt], label='area 1')
ax[1].loglog(data_anly.a2.spon_freq[freq_plt], data_anly.a2.spon_coef.mean(0)[freq_plt], label='area 2')

ax[0].set_xlabel('Frequency (Hz)')
ax[1].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Power spectrum (a.u.)')
ax[1].set_ylabel('Power spectrum (a.u.)')
ax[0].legend()
ax[1].legend()

fig.savefig('spectrum_spon_%.d.png'%loop_num)
plt.close()

#%%
'''
Wavelet spectrogram of spontaneous activity
'''
area = 'a1'
start_time = 5500
end_time = 6500
t_ext = 500
mua1 = fra.get_spkcount_sum_sparmat(data.__dict__[area].ge.spk_matrix[mua_neuron], start_time-t_ext, end_time+t_ext,
                   sample_interval = 1,  window = 1, dt = 0.1)/mua_neuron.shape[0]/0.001
area = 'a2'
mua2 = fra.get_spkcount_sum_sparmat(data.__dict__[area].ge.spk_matrix[mua_neuron], start_time-t_ext, end_time+t_ext,
                   sample_interval = 1,  window = 1, dt = 0.1)/mua_neuron.shape[0]/0.001

cmor = pywt.ContinuousWavelet('cmor15-1')
cmor.lower_bound = -8
cmor.upper_bound = 8
sampling_period = 0.001
freq = np.arange(30, 81, 2)[::-1]
scale = 1/sampling_period/freq

coef1, freq = fqa.mycwt(mua1, cmor, sampling_period, scale = scale,  method = 'fft', L1_norm = True)
coef1 = np.abs(coef1)
coef2, freq = fqa.mycwt(mua2, cmor, sampling_period, scale = scale,  method = 'fft', L1_norm = True)
coef2 = np.abs(coef2)


fig, ax = plt.subplots(2,1, figsize=[8,6])

im1 = ax[0].imshow(coef1[:,t_ext:-t_ext],aspect='auto',extent=[-0.5,1000+0.5, 29,81])#, vmax=1.1,vmin=0)#,origin='lower')
im2 = ax[1].imshow(coef2[:,t_ext:-t_ext],aspect='auto',extent=[-0.5,1000+0.5, 29,81])#, vmax=1.1,vmin=0)#,origin='lower')

ax[0].set_title('area 1')
ax[1].set_title('area 2')
ax[1].set_xlabel('Time (ms)')
ax[0].set_ylabel('Frequency (Hz)')
ax[1].set_ylabel('Frequency (Hz)')

cax = ax[0].inset_axes([1.01, 0, 0.013, 1], transform=ax[0].transAxes)
fig.colorbar(im1, cax=cax, orientation='vertical')
cax.text(-3.2,1.03,'Amp. (a.u.)', ha='left',fontsize=10, transform=cax.transAxes)

cax = ax[1].inset_axes([1.01, 0, 0.013, 1], transform=ax[1].transAxes)
fig.colorbar(im2, cax=cax, orientation='vertical')
cax.text(-3.2,1.03,'Amp. (a.u.)', ha='left',fontsize=10, transform=cax.transAxes)


fig.savefig('wavelet_spon_%.d.png'%loop_num)
plt.close()
#%%
'''A demo for calculating theta-gamma coupling'''

findcfc = cfc.cfc()
Fs = 1000;
phaseBand = np.arange(1,14.1,0.5)
ampBand = np.arange(20,101,5) 
phaseBandWid = 0.5 ;
ampBandWid = 5 ;

band1 = np.concatenate((phaseBand - phaseBandWid, ampBand - ampBandWid)).reshape(1,-1)
band2 = np.concatenate((phaseBand + phaseBandWid, ampBand + ampBandWid)).reshape(1,-1)
subBand = np.concatenate((band1,band2),0)
subBand = subBand.T
#
##%%
findcfc.timeDim = -1;
findcfc.Fs = Fs; 
findcfc.phaseBand = subBand[:len(phaseBand)];
findcfc.ampBand = subBand[len(phaseBand):]
findcfc.section_input_to_find_MI_cfc = None #[4000,40000]
findcfc.optionSur = 2

area = 'a1'
start_time = 5000
end_time = 10000
mua1 = fra.get_spkcount_sum_sparmat(data.__dict__[area].ge.spk_matrix[mua_neuron], start_time, end_time,
                   sample_interval = 1,  window = 1, dt = 0.1)/mua_neuron.shape[0]/0.001

MI_raw, MI_surr, meanBinAmp = findcfc.find_cfc_from_rawsig(mua1,return_Ampdist=True)

fig, ax1 = plt.subplots(1,1, figsize=[6,6])
imcf = ax1.contourf(phaseBand, ampBand, MI_surr.T, 15)#, aspect='auto')
imc = ax1.contour(phaseBand, ampBand, MI_surr.T, 15, colors='k', linewidths=0.6)#, aspect='auto')

plt.colorbar(im1, ax=ax1)

ax1.set_xlabel('Phase frequency (Hz)')
ax1.set_ylabel('Amp. frequency (Hz)')
ax1.set_title('area 1')

fig.savefig('theta_gam_coup_%.d.png'%loop_num)
plt.close()


#%%
'''Fano factor'''
neu_range = fano_mua_range 
bin_count_interval_hz = 5

data_anly.fano = mydata.mydata()

stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool = [None]*1

neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
data_anly.fano.neu_range = neu_range

    
simu_time_tot = data.param.simutime#29000

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

fanomm = fano_mean_match.fano_mean_match()

fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'regression' # 'mean' or 'regression'
fanomm.mean_match_across_condition = True # do mean matching across different condition e.g. cued or uncued condition
fanomm.seed = 100

N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
win_lst = [50, ] # ms
data_anly.fano.win_lst = win_lst
data_anly.fano.bin_count_interval_hz = bin_count_interval_hz
win_id = -1
for win in win_lst:#[50,100]:
    win_id += 1
    for st in range(n_StimAmp):
        fanomm.bin_count_interval = win*10**-3*bin_count_interval_hz
        fanomm.win = win
        fanomm.stim_onoff = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
        fanomm.stim_onoff_2 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy() #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True
        

        fanomm.t_bf = -(win/2)
        fanomm.t_aft = -(win/2)
        #fano_mean_noatt, fano_std_noatt, _ = fanomm.get_fano()
        fano_mean_noatt, fano_sem_noatt, _, fano_mean_att, fano_sem_att, _ = fanomm.get_fano()
        
        if win_id ==0 and st == 0:
            data_anly.fano.fano_mean_sem = [None]*len(win_lst)#np.zeros([n_StimAmp*2, fano_mean_noatt.shape[0], 2, len(win_lst)])
        if st == 0:
            data_anly.fano.fano_mean_sem[win_id] = np.zeros([n_StimAmp*2, fano_mean_noatt.shape[0], 2])
        data_anly.fano.fano_mean_sem[win_id][st,:,0] = fano_mean_noatt
        data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,0] = fano_mean_att
        data_anly.fano.fano_mean_sem[win_id][st,:,1] = fano_sem_noatt
        data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,1] = fano_sem_att
        
    
    fig, ax = plt.subplots(1,1, figsize=[8,6])
    for st in range(n_StimAmp):
        ax.errorbar(np.arange(data_anly.fano.fano_mean_sem[win_id].shape[1])*10+(win/2), \
                    data_anly.fano.fano_mean_sem[win_id][st,:,0],data_anly.fano.fano_mean_sem[win_id][st,:,1], \
                    fmt='--', c=clr[st], marker='o', label='uncued, stim_amp: %.1f Hz'%(stim_amp[st]))
        ax.errorbar(np.arange(data_anly.fano.fano_mean_sem[win_id].shape[1])*10+(win/2),\
                    data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,0],data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,1],\
                    fmt='-', c=clr[st], marker='o', label='cued, stim_amp: %.1f Hz'%(stim_amp[st]))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Fano factor')
    plt.legend()
    title3 = 'fano_win%.1f_bin%d\n_range%d'%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    fig.suptitle(title3)
    savetitle = title3.replace('\n','')
    fanofile = savetitle+save_apd+'_%d'%(loop_num)+'.png'
    fig.savefig(fanofile)
    plt.close()
        

#%%
'''noise correlation'''
get_coefDiffDist = True

neuron = np.arange(data.a1.param.Ne)
neu_pool = [None, None]
stim_loc = np.array([0,0])
dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
stim_loc = stim_loc_2 #np.array([-32,-32])

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool[1] = neuron[(dist >= 0) & (dist <= neu_range)]

win_lst = [ 50]# 
data_anly.nscorr_t = mydata.mydata()
data_anly.nscorr_t.win_lst = win_lst
data_anly.nscorr_t.neu_pool = neu_pool
data_anly.nscorr_t.param = []
data_anly.nscorr_t.nscorr_t = []
if get_coefDiffDist:
    data_anly.nscorr_t.nscorr_diff = []
    
for win in win_lst:
    nscorr = fra.noise_corr()
    nscorr.win = win # ms sliding window length to count spikes
    nscorr.move_step = 10 # ms sliding window move step, (sampling interval for time varying noise correlation)
    nscorr.t_bf = -nscorr.win/2 # ms; time before stimulus onset to start to sample noise correlation
    nscorr.t_aft = -nscorr.win/2 # ms; time after stimulus off to finish sampling noise correlation
    data_anly.nscorr_t.param.append(data.class2dict(nscorr))
    
    simu_time_tot = data.param.simutime#29000
        
        #N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
    nscorr.spk_matrix1 = data.a1.ge.spk_matrix[neu_pool[0],:]
    nscorr.spk_matrix2 = data.a1.ge.spk_matrix[neu_pool[1],:]
    
    nscorr.return_every_coef = get_coefDiffDist
    
    for st in range(n_StimAmp):  
        '''no-att; within 1 group'''
        nscorr.spk_matrix1 = data.a1.ge.spk_matrix[neu_pool[0],:]
        nscorr.spk_matrix2 = None
        nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
        #nscorr.dura2 = None
        if get_coefDiffDist:
            corr_, corr_all_noatt = nscorr.get_nc_withingroup_t()
        else:
            corr_ = nscorr.get_nc_withingroup_t()
            
        if st == 0:
            if get_coefDiffDist:
                ns_t = np.zeros([3,2,corr_.shape[1],n_StimAmp*2])
                ns_diff = []
            else:
                ns_t = np.zeros([3,2,corr_.shape[1],n_StimAmp*2])
    
        ns_t[0, :, :, st] = corr_
        
        '''att; within 1 group'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
        #nscorr.dura2 = None
        if get_coefDiffDist:
            corr_, corr_all_att = nscorr.get_nc_withingroup_t()
        else:
            corr_ = nscorr.get_nc_withingroup_t()
        ns_t[0, :, :, st+n_StimAmp] = corr_
        if get_coefDiffDist:
            ns_diff.append(corr_all_att-corr_all_noatt)
                    
        '''no-att; within group 2'''
        nscorr.spk_matrix1 = data.a1.ge.spk_matrix[neu_pool[1],:]
        nscorr.spk_matrix2 = None
        nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
        #nscorr.dura2 = None
        if get_coefDiffDist:
            corr_, corr_all_noatt = nscorr.get_nc_withingroup_t()
        else:
            corr_ = nscorr.get_nc_withingroup_t()    
        ns_t[2, :, :, st] = corr_
        '''att; within group 2'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
        #nscorr.dura2 = None
        if get_coefDiffDist:
            corr_, corr_all_att = nscorr.get_nc_withingroup_t()
        else:
            corr_ = nscorr.get_nc_withingroup_t()
        ns_t[2, :, :, st+n_StimAmp] = corr_
        if get_coefDiffDist:
            ns_diff.append(corr_all_att-corr_all_noatt)

        '''no-att; between 2 groups'''
        nscorr.spk_matrix1 = data.a1.ge.spk_matrix[neu_pool[0],:]
        nscorr.spk_matrix2 = data.a1.ge.spk_matrix[neu_pool[1],:]
        nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
        #nscorr.dura2 = nscorr.dura1
        if get_coefDiffDist:
            corr_, corr_all_noatt = nscorr.get_nc_betweengroups_t()
        else:
            corr_ = nscorr.get_nc_betweengroups_t()
        ns_t[1, :, :, st] = corr_
        '''att; between 2 groups'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
        #nscorr.dura2 = nscorr.dura1
        if get_coefDiffDist:
            corr_, corr_all_att = nscorr.get_nc_betweengroups_t()
        else:
            corr_ = nscorr.get_nc_betweengroups_t()
        ns_t[1, :, :, st+n_StimAmp] = corr_
        if get_coefDiffDist:
            ns_diff.append(corr_all_att-corr_all_noatt)
    
    data_anly.nscorr_t.nscorr_t.append(ns_t)
    if get_coefDiffDist:
        data_anly.nscorr_t.nscorr_diff.append(ns_diff)
    
    fig, ax = plt.subplots(1,1, figsize=[8,5])
    ax = [ax]
    sample_t = np.arange(ns_t.shape[2])*nscorr.move_step-nscorr.t_bf
    for st in range(n_StimAmp):  
        ax[0].errorbar(sample_t, ns_t[0, 0, :, st],ns_t[0, 1, :, st], c=clr[st], fmt='--', marker='o', label='uncued;amp:%.1fHz'%stim_amp[st])
        ax[0].errorbar(sample_t, ns_t[0, 0, :, st+n_StimAmp],ns_t[0, 1, :, st+n_StimAmp], c=clr[st], fmt='-', marker='o', label='cued;amp:%.1fHz'%stim_amp[st])
    
    #ax.legend()
    ax[0].legend()
    ax[0].set_xlim([sample_t.min()-20,sample_t.max()+150])
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Noise correlation')

    fig.suptitle(title + ' win:%.1f'%win)
    nsfile = 'NoisCorr_win%.1f'%win+save_apd+'_%d'%(loop_num)+'.png'
    if save_img: fig.savefig(nsfile)
    plt.close()
   
#%%
if save_analy_file:
    data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)

#%%
"""
'''animation'''
if get_movie:
    
    #%
    '''spontaneous'''
    
    first_stim = 0; last_stim = 0
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 1000
    end_time = data.a1.param.stim1.stim_on[last_stim,0] 
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=None, adpt=None)
    
    ani.save('spontaneous_%d.mp4'%loop_num)
    del ani
    
        
    #%%
    '''2 inputs; uncued'''
    
    stim_ani = 0
    first_stim = stim_ani*n_perStimAmp; 
    last_stim = stim_ani*n_perStimAmp 
    
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = data.a1.param.stim1.stim_on[last_stim,1] + 500
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    stim_on_off = data.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    #stim = [[[[31.5,31.5],[31.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    adpt = None
    #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    
    ani.save('twoInputs_uncued_%d.mp4'%loop_num)
    del ani
    
    #%%
    '''2 inputs; cued'''
    stim_ani = 0
    first_stim = (n_StimAmp + stim_ani)*n_perStimAmp 
    last_stim = (n_StimAmp + stim_ani)*n_perStimAmp 
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = data.a1.param.stim1.stim_on[last_stim,1] + 500
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    stim_on_off = data.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    #stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
    #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    #stim = [[[[31.5,31.5],[31.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    if hasattr(data.param, 'chg_adapt_range'):
        chg_adapt_range = data.param.chg_adapt_range
    else:
        chg_adapt_range = 7
        
    adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[chg_adapt_range]]]]
    #adpt = None
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    
    ani.save('twoInputs_cued_%d.mp4'%loop_num)
    del ani
"""
