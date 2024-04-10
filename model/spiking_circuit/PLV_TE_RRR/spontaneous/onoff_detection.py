# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:37:53 2024

@author: Shencong Ni
"""



import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
#import load_data_dict
import mydata
#import brian2.numpy_ as np
import numpy as np
#from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
import frequency_analysis as fqa
import fano_mean_match
#import find_change_pts
import detect_onoff
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
data_dir = 'raw_data/'
datapath = data_dir
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num

onoff_thre_method = '1'
sfx = 'win10_min10_smt1_mtd%s_ctr'%onoff_thre_method #'smt_ctr'

savefile_name = 'data_anly_onoff_testthres_%s'%sfx #'data_anly' data_anly_temp data_anly_onoff_thres_samesens data_anly_onoff_onoffsetsamelen
#save_apd = '' 
save_apd_sens='_sens_thre_%s'%sfx  # _sens_thre_samesens
save_apd_asso='_asso_thre_%s'%sfx # _asso_thre_samesens
save_apd_alignedmua = '_thre_%s'%sfx
save_apd_thre = '_thre_%s'%sfx

mua_loca = [0, 0] # [0, 0] #[-32, -32] # [0, 0]
#onoff_detect_method = 'threshold'
save_fig_dir = './fig_movi/'
if not os.path.exists(save_fig_dir):
    try:os.makedirs(save_fig_dir)
    except FileExistsError:
        pass
#%%
def rate2mua(r): # convet firing rate (Hz) to MUA spike counts
    r=np.array(r); 
    return r*0.01*80 # 80 MUA neurons; 10 ms time bin

def mua2rate(mua):
    mua=np.array(mua); 
    return mua/0.01/80
#%%
thre_spon_sens_hz = np.arange(3,15.1,0.5) #rate2mua(7)#4
#thre_stim_sens_hz = np.tile(np.arange(15,40.1,0.5),(1,2,1)) #rate2mua([[25,30]]) #[12, 15, 30] rate2mua([[25,30]]) rate2mua([[30,30]])
thre_spon_asso_hz = np.arange(5,30.1,0.5) # np.arange(3,15.1,0.5) #rate2mua(10)
#thre_stim_asso_hz = np.tile(np.arange(10,90.1,0.5),(1,2,1)) # np.tile(np.arange(10,50.1,0.5),(1,2,1)) #np.vstack((np.arange(10,50.1,0.5), np.arange(10,50.1,0.5))) #rate2mua([[20, 40]]) #[12, 15, 30] [[stim1_noatt,stim2_att],...]


thre_spon_sens_mua = rate2mua(thre_spon_sens_hz)#4
#thre_stim_sens_mua = rate2mua(thre_stim_sens_hz) #[12, 15, 30] rate2mua([[25,30]]) rate2mua([[30,30]])
thre_spon_asso_mua = rate2mua(thre_spon_asso_hz)
#thre_stim_asso_mua = rate2mua(thre_stim_asso_hz) #[12, 15, 30] [[stim1_noatt,stim2_att],...]

# fftplot = 1; getfano = 1
# get_nscorr = 1; get_nscorr_t = 1
# get_TunningCurve = 1; get_HzTemp = 1
# firing_rate_long = 1

if loop_num%1 == 0: save_img = 1
else: save_img = 0


save_analy_file = True
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

#n_StimAmp = data.a1.param.stim1.n_StimAmp
#n_perStimAmp = data.a1.param.stim1.n_perStimAmp
#stim_amp = [400] #200*2**np.arange(n_StimAmp)

title = ''
#%%
def onoff_analysis(findonoff, spk_mat, mua_neuron,  \
                   onoff_detect_method, onoffThreshold_spon_list, \
               title=None, save_apd=None, savefig=False):
# def onoff_analysis(findonoff, spk_mat, mua_neuron, analy_dura_stim, ignore_respTransient, n_StimAmp, n_perStimAmp, stim_amp, \
#                    onoff_detect_method, onoffThreshold_spon_list, onoffThreshold_stim_list, \
#                title=None, save_apd=None, savefig=False):

    '''spon onoff'''
    
    
    dt_samp = findonoff.mua_sampling_interval
    start = 5000; end = 205000
    analy_dura_spon = np.array([[start,end]])
    # analy_dura_spon = np.arange(5000, 205000-1, 10000)
    # analy_dura_spon = np.array([analy_dura_spon, analy_dura_spon + 10000]).T
    
    
    
    #spon_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura_spon, cross_validation=False) # cross_validation must be False here since spontaneous activity only has one trial
    findonoff.MinThreshold = None # None # 1000
    findonoff.MaxNumChanges = int(round(((analy_dura_spon[0,1]-analy_dura_spon[0,0]))/1000*12)) #15
    findonoff.smooth_window = None #52

    if onoff_detect_method == 'threshold':
        spon_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura_spon, method = 'threshold', threshold_list=onoffThreshold_spon_list)
    else:
        spon_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura_spon, method = 'comparing')
    
    R = mydata.mydata()
    R.spon = spon_onoff
    
    
    stim_num = 0
    end_plot_time = 3000
    n_ind, t = spk_mat[mua_neuron,analy_dura_spon[stim_num,0]*10:(analy_dura_spon[stim_num,0]+end_plot_time)*10].nonzero()
    
    
    
    fig,ax = plt.subplots(2,1, figsize=[15,6])
    # ax[0].plot(t/10, n_ind, '|')
    # ax[0].plot(show_stim_on)  
    # ax[1].plot(np.arange(mua_.shape[0]),mua_)
    #ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
    plt_mua_t = np.arange(len(spon_onoff.mua_smt[stim_num]))*dt_samp
    
    ax[0].plot(plt_mua_t[:int(round(end_plot_time/dt_samp))], spon_onoff.mua_smt[stim_num][:int(round(end_plot_time/dt_samp))]/mua_neuron.shape[0]/(findonoff.mua_win/1000), c=clr[0], label='smooth')
    ax[0].plot(plt_mua_t[:int(round(end_plot_time/dt_samp))], spon_onoff.mua[stim_num][:int(round(end_plot_time/dt_samp))]/mua_neuron.shape[0]/(findonoff.mua_win/1000), c=clr[2])
    ax[0].legend()
    
    for i in range(spon_onoff.cpts[stim_num].shape[0]):
        if spon_onoff.cpts[stim_num][i] >= end_plot_time : break
        #ax[0].plot([spon_onoff.cpts[stim_num][i],spon_onoff.cpts[stim_num][i]],[0,spon_onoff.mua[stim_num][:end_plot_time].max()], c=clr[1])
        ax[0].axvline(spon_onoff.cpts[stim_num][i], c=clr[1])
    ax[1].plot(plt_mua_t[:int(round(end_plot_time/dt_samp))], spon_onoff.onoff_bool[stim_num][:int(round(end_plot_time/dt_samp))]*len(mua_neuron))
    ax[1].plot(t/10, n_ind, '|')
    fig.suptitle(title + save_apd + '_spon_on-off; plot')
    
    savetitle = title.replace('\n','')
    onofffile = savetitle+'_spon_t'+save_apd+'_%d'%(loop_num)+'.png'
    if savefig : 
        fig.savefig(save_fig_dir + onofffile)
        plt.close()
    
    
    fig, ax = plt.subplots(1,4, figsize=[15,6])
    hr = ax[0].hist(np.concatenate(spon_onoff.on_t),bins=20, density=True)
    mu = np.concatenate(spon_onoff.on_t).mean()
    ax[0].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[0].set_title('on period; spon; mean:%.2f'%mu)
    hr = ax[1].hist(np.concatenate(spon_onoff.off_t),bins=20, density=True)
    mu = np.concatenate(spon_onoff.off_t).mean()
    ax[1].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[1].set_title('off period; spon; mean:%.2f'%mu)
    hr = ax[2].hist(np.concatenate(spon_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = np.concatenate(spon_onoff.on_amp).mean()
    ax[2].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[2].set_title('on rate; spon; mean:%.2f'%mu)
    hr = ax[3].hist(np.concatenate(spon_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = np.concatenate(spon_onoff.off_amp).mean()
    ax[3].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[3].set_title('off rate; spon; mean:%.2f'%mu)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    fig.suptitle(title + save_apd + '_spon_on-off dist')
    
    savetitle = title.replace('\n','')
    onofffile = savetitle+'_spon_dis'+save_apd+'_%d'%(loop_num)+'.png'
    if savefig :
        fig.savefig(save_fig_dir + onofffile)
        plt.close()
    
    return R
    
    
#%%

#%%
'''onoff'''

#%%

mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
# simu_time_tot = data.param.simutime#29000

# data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
# data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10])

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




findonoff_cpts = detect_onoff.MUA_findchangepts()
findonoff_cpts.mua_win = 10 # ms 
findonoff_cpts.MinDistance = 10
findonoff_cpts.smooth_data = 1
findonoff_cpts.smooth_method = 'sgolay' # input for 'smoothdata' in matlab ; sgolay  rlowess
findonoff_cpts.onoff_thre_method = onoff_thre_method

#%%


data_anly.onoff_sens = onoff_analysis(findonoff_cpts, data.a1.ge.spk_matrix, mua_neuron, \
                    onoff_detect_method='threshold', onoffThreshold_spon_list=thre_spon_sens_mua,  \
                    title=title, save_apd=save_apd_sens, savefig=save_img)


#%%


data_anly.onoff_asso = onoff_analysis(findonoff_cpts, data.a2.ge.spk_matrix, mua_neuron, \
                    onoff_detect_method='threshold', onoffThreshold_spon_list=thre_spon_asso_mua, \
                    title=title, save_apd=save_apd_asso, savefig=save_img)
    
#%%
'''
fig, ax = plt.subplots(2,2,figsize=[12,7])

ax[0,0].plot(thre_spon_sens_hz, data_anly.onoff_sens.spon.sse[0], label='spon sens opt thre: %.2f'%mua2rate(data_anly.onoff_sens.spon.threshold_opt))
ax[1,0].plot(thre_spon_asso_hz, data_anly.onoff_asso.spon.sse[0], label='spon asso opt thre: %.2f'%mua2rate(data_anly.onoff_asso.spon.threshold_opt))


for axx in ax:
    for axy in axx:
        axy.legend()
        
figtitle = title + '_testthre_spon' + save_apd_thre
fig.suptitle(figtitle)

savetitle = title + '_testthre_spon' + save_apd_thre
savetitle = savetitle.replace('\n','')
savetitle = savetitle+'_%d'%(loop_num)+'.png'

if save_img :
    fig.savefig(save_fig_dir + savetitle)
    plt.close() 
'''

#%%

    
if save_analy_file:
    data_anly.save(data_anly.class2dict(), data_dir+savefile_name+'_%d.file'%loop_num)


