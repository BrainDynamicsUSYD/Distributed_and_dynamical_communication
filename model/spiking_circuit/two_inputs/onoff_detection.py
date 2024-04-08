#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:00:58 2021

@author: Shencong Ni
"""

'''
Detect the On/Off states.
Run 'two_inputs_simu.py' before running this script.
'''



import matplotlib as mpl
mpl.use('Agg')
import mydata
import numpy as np
import firing_rate_analysis as fra
# import frequency_analysis as fqa
import detect_onoff
import connection as cn
import sys
import os
import matplotlib.pyplot as plt

#%%
datapath = 'raw_data/' # path to data
sys_argv = int(sys.argv[1])
loop_num = sys_argv 

onoff_thre_method = '1'
apd = 'win10_min10_smt1_mtd%s_ctr_'%onoff_thre_method #

savefile_name = 'data_anly_onoff_testthres_%s'%apd 

save_apd_sens='_sens_thre_%s'%apd  # 
save_apd_asso='_asso_thre_%s'%apd # 
save_apd_alignedmua = '_thre_%s'%apd
save_apd_thre = '_thre_%s'%apd

mua_loca = [0, 0] # [0, 0] [-32, -32]

save_fig_dir = './fig_movi/ctr/'
if not os.path.exists(save_fig_dir):
    try:os.makedirs(save_fig_dir)
    except FileExistsError:
        pass
#%%
'''80 MUA neurons; 10 ms time bin !!!'''
def rate2mua(r): # convet firing rate (Hz) to MUA spike counts
    r=np.array(r); 
    return r*0.01*80 # 80 MUA neurons; 10 ms time bin !!!

def mua2rate(mua):
    mua=np.array(mua); 
    return mua/0.01/80
#%%
thre_spon_sens_hz = np.arange(3,15.1,0.5) 
thre_stim_sens_hz = np.tile(np.arange(15,40.1,0.5),(1,2,1)) 
thre_spon_asso_hz = np.arange(5,30.1,0.5) # 
thre_stim_asso_hz = np.tile(np.arange(10,90.1,0.5),(1,2,1)) # 

thre_spon_sens_mua = rate2mua(thre_spon_sens_hz)#
thre_stim_sens_mua = rate2mua(thre_stim_sens_hz) #
thre_spon_asso_mua = rate2mua(thre_spon_asso_hz)
thre_stim_asso_mua = rate2mua(thre_stim_asso_hz) #
#%%
if loop_num%2 == 0: save_img = 1
else: save_img = 0

# if loop_num%10 ==0: get_ani = 0
# else: get_ani = 0

save_analy_file = True
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = [400] #200*2**np.arange(n_StimAmp)

title = '' # title for plots  
#%%
def onoff_analysis(findonoff, spk_mat, mua_neuron, analy_dura_stim, ignore_respTransient, n_StimAmp, n_perStimAmp, stim_amp, \
                   onoff_detect_method, onoffThreshold_spon_list, onoffThreshold_stim_list, \
               title=None, save_apd=None, savefig=False):

    '''spon onoff'''
    
    print('Analysing spontaneous activity ...')
    dt_samp = findonoff.mua_sampling_interval
    start = 5000; end = 20000
    analy_dura_spon = np.array([[start,end]])
    
    
    findonoff.MinThreshold = None # None # 1000
    findonoff.MaxNumChanges = int(round(((analy_dura_spon[0,1]-analy_dura_spon[0,0]))/1000*12)) #15
    findonoff.smooth_window = None #

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

    
    '''stim;'''
    R.stim_noatt = []
    R.stim_att = []
    R.ignore_respTransient = ignore_respTransient # length of transient response to be ignored; ms
    print(n_StimAmp)
    for n in range(n_StimAmp):
        print ('stimulus amplitude: %.1f Hz'%stim_amp[n])
        '''no att/uncued'''
        print('Analysing stimulus evoked activity without cue ...')

        analy_dura = analy_dura_stim[n*n_perStimAmp:(n+1)*n_perStimAmp].copy()
        analy_dura[:,0] += ignore_respTransient #200 #500
        
        findonoff.MinThreshold = None # None # 1000
        findonoff.MaxNumChanges = int(round(((analy_dura[0,1]-analy_dura[0,0]))/1000*12)) #15
        print(analy_dura[0,1]-analy_dura[0,0])
        findonoff.smooth_window = None #52
    
        if onoff_detect_method == 'threshold':
            stim_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold_list=onoffThreshold_stim_list[n][0])
        else:
            stim_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'comparing')
    
        
        R.stim_noatt.append(stim_onoff)
        
        stim_num = 0
        plt_dura = 2000 #ms
        n_ind, t = spk_mat[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+plt_dura)*10].nonzero()
        
        '''on-off plot'''
        fig, ax = plt.subplots(4,1, figsize=[15,12])
        '''no att'''
        plt_mua_t = np.arange(len(stim_onoff.mua_smt[stim_num]))*dt_samp
        plt_dura_dt = int(round(plt_dura/dt_samp))
        print(plt_dura_dt)
        ax[0].plot(plt_mua_t[:plt_dura_dt], stim_onoff.mua_smt[stim_num][:plt_dura_dt]/mua_neuron.shape[0]/(findonoff.mua_win/1000), c=clr[0], label='smooth')
        ax[0].plot(plt_mua_t[:plt_dura_dt], stim_onoff.mua[stim_num][:plt_dura_dt]/mua_neuron.shape[0]/(findonoff.mua_win/1000), c=clr[2])
        ax[0].legend()
        for i in range(stim_onoff.cpts[stim_num].shape[0]):
            if stim_onoff.cpts[stim_num][i] >= plt_dura: break
            ax[0].axvline(stim_onoff.cpts[stim_num][i], c=clr[1])
        ax[0].set_title('stim; no att')
        ax[1].plot(plt_mua_t[:plt_dura_dt], stim_onoff.onoff_bool[stim_num][:plt_dura_dt]*len(mua_neuron))
        ax[1].plot(t/10, n_ind, '|')
        ax[0].xaxis.set_visible(False)
        ax[1].xaxis.set_visible(False) 
       
        
        '''att/cued'''
        print('Analysing stimulus evoked activity with cue ...')
        
        analy_dura = analy_dura_stim[(n+n_StimAmp)*n_perStimAmp:(n+n_StimAmp+1)*n_perStimAmp].copy()
        analy_dura[:,0] += ignore_respTransient #
        
        findonoff.MinThreshold = None # None # 1000
        findonoff.MaxNumChanges = int(round(((analy_dura[0,1]-analy_dura[0,0]))/1000*12)) #15
        findonoff.smooth_window = None #
    
        if onoff_detect_method == 'threshold':
            stim_onoff_att = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold_list=onoffThreshold_stim_list[n][1])
        else:
            stim_onoff_att = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'comparing')

        
        R.stim_att.append(stim_onoff_att)
        
        stim_num = 0
        plt_dura = 2000
        n_ind, t = spk_mat[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+plt_dura)*10].nonzero()
         
        '''att'''
        plt_mua_t = np.arange(len(stim_onoff_att.mua_smt[stim_num]))*dt_samp
        plt_dura_dt = int(round(plt_dura/dt_samp))
        ax[2].plot(plt_mua_t[:plt_dura_dt], stim_onoff_att.mua_smt[stim_num][:plt_dura_dt]/mua_neuron.shape[0]/(findonoff.mua_win/1000), c=clr[0], label='smooth')
        ax[2].plot(plt_mua_t[:plt_dura_dt], stim_onoff_att.mua[stim_num][:plt_dura_dt]/mua_neuron.shape[0]/(findonoff.mua_win/1000), c=clr[2])
        ax[2].legend()
        for i in range(stim_onoff_att.cpts[stim_num].shape[0]):
            if stim_onoff_att.cpts[stim_num][i] >= plt_dura: break
            ax[2].axvline(stim_onoff_att.cpts[stim_num][i], c=clr[1])
    
        ax[2].set_title('stim; att')
        ax[3].plot(plt_mua_t[:plt_dura_dt], stim_onoff_att.onoff_bool[stim_num][:plt_dura_dt]*len(mua_neuron))
    
        ax[3].plot(t/10, n_ind, '|')
        ax[2].xaxis.set_visible(False)      
        fig.suptitle(title + save_apd + '_stim: %.1f hz'%stim_amp[n])
    
        savetitle = title.replace('\n','')
        onofffile = savetitle+'_stim%d_t'%n+save_apd+'_%d'%(loop_num)+'.png'
        if savefig :
            fig.savefig(save_fig_dir + onofffile)
            plt.close()      
        
        
        fig,ax = plt.subplots(2,4, figsize=[15,6])
        hr = ax[0,0].hist(np.concatenate(stim_onoff.on_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff.on_t).mean()
        ax[0,0].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,0].set_title('on period; no att; mean:%.2f'%mu)
        hr = ax[0,1].hist(np.concatenate(stim_onoff.off_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff.off_t).mean()
        ax[0,1].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,1].set_title('off period; no att; mean:%.2f'%mu)
        hr = ax[0,2].hist(np.concatenate(stim_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[0,2].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,2].set_title('on rate; no att; mean:%.2f'%mu)
        hr = ax[0,3].hist(np.concatenate(stim_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[0,3].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,3].set_title('off rate; no att; mean:%.2f'%mu)
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        
        hr = ax[1,0].hist(np.concatenate(stim_onoff_att.on_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff_att.on_t).mean()
        ax[1,0].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,0].set_title('on period; att; mean:%.2f'%mu)
        hr = ax[1,1].hist(np.concatenate(stim_onoff_att.off_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff_att.off_t).mean()
        ax[1,1].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,1].set_title('off period; att; mean:%.2f'%mu)
        hr = ax[1,2].hist(np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[1,2].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,2].set_title('on rate; att; mean:%.2f'%mu)
        hr = ax[1,3].hist(np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[1,3].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,3].set_title('off rate; att; mean:%.2f'%mu)
        ax[1,0].set_yscale('log')
        ax[1,1].set_yscale('log')
        fig.suptitle(title + save_apd + '_stim: %.1f hz'%stim_amp[n])
    
        savetitle = title.replace('\n','')
        onofffile = savetitle+'_stim%d_dis'%n+save_apd+'_%d'%(loop_num)+'.png'
        if savefig :
            fig.savefig(save_fig_dir + onofffile)
            plt.close() 
    
    return R

#%%
'''onoff'''

mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)


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



analy_dura_stim = data.a1.param.stim1.stim_on.copy()

findonoff_cpts = detect_onoff.MUA_findchangepts()
findonoff_cpts.mua_win = 10 # ms 
findonoff_cpts.MinDistance = 10
findonoff_cpts.smooth_data = 1
findonoff_cpts.smooth_method = 'sgolay' # input for 'smoothdata' in matlab ; sgolay  rlowess
findonoff_cpts.onoff_thre_method = onoff_thre_method
#%%
ignore_respTransient = 200 # ms
data_anly.onoff_sens = onoff_analysis(findonoff_cpts, data.a1.ge.spk_matrix, mua_neuron, analy_dura_stim, ignore_respTransient, n_StimAmp, n_perStimAmp, stim_amp, \
                    onoff_detect_method='threshold', onoffThreshold_spon_list=thre_spon_sens_mua, onoffThreshold_stim_list=thre_stim_sens_mua, \
                    title=title, save_apd=save_apd_sens, savefig=save_img)

#%%

data_anly.onoff_asso = onoff_analysis(findonoff_cpts, data.a2.ge.spk_matrix, mua_neuron, analy_dura_stim, ignore_respTransient, n_StimAmp, n_perStimAmp, stim_amp, \
                    onoff_detect_method='threshold', onoffThreshold_spon_list=thre_spon_asso_mua, onoffThreshold_stim_list=thre_stim_asso_mua, \
                    title=title, save_apd=save_apd_asso, savefig=save_img)
#%%    
#%
'''optimal threshold'''
print('optimal threshold for ON-OFF state classification')
for n in range(len(stim_amp)):
    fig, ax = plt.subplots(2,2,figsize=[12,7])
    
    ax[0,0].plot(thre_spon_sens_hz, data_anly.onoff_sens.spon.sse[0], label='spon sens opt thre: %.2f'%mua2rate(data_anly.onoff_sens.spon.threshold_opt))
    ax[1,0].plot(thre_spon_asso_hz, data_anly.onoff_asso.spon.sse[0], label='spon asso opt thre: %.2f'%mua2rate(data_anly.onoff_asso.spon.threshold_opt))
    
    ax[0,1].plot(thre_stim_sens_hz[n,0,:], data_anly.onoff_sens.stim_noatt[n].sse.mean(0), label='noatt sens opt thre: %.2f'%mua2rate(data_anly.onoff_sens.stim_noatt[0].threshold_opt))
    ax[0,1].plot(thre_stim_sens_hz[n,1,:], data_anly.onoff_sens.stim_att[n].sse.mean(0), label='att sens opt thre: %.2f'%mua2rate(data_anly.onoff_sens.stim_att[0].threshold_opt))
    
                 
    ax[1,1].plot(thre_stim_asso_hz[n,0,:], data_anly.onoff_asso.stim_noatt[n].sse.mean(0), label='noatt asso opt thre: %.2f'%mua2rate(data_anly.onoff_asso.stim_noatt[0].threshold_opt))
    ax[1,1].plot(thre_stim_asso_hz[n,1,:], data_anly.onoff_asso.stim_att[n].sse.mean(0), label='att asso opt thre: %.2f'%mua2rate(data_anly.onoff_asso.stim_att[0].threshold_opt))
    
    for axx in ax:
        for axy in axx:
            axy.legend()
            
    figtitle = title + '_testthre_stim%.1fhz'%stim_amp[n] + save_apd_thre
    fig.suptitle(figtitle)
    
    savetitle = title + '_testthre_stim%d'%n + save_apd_thre
    savetitle = savetitle.replace('\n','')
    savetitle = savetitle+'_%d'%(loop_num)+'.png'
    
    if save_img :
        fig.savefig(save_fig_dir + savetitle)
        plt.close() 
#%%
if save_analy_file:
    data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)

#%%
# if get_ani:
    # #%
    # '''spon'''
    # #first_stim = 0 #1*n_perStimAmp -1; 
    # #last_stim = 0 #1*n_perStimAmp
    # start_time = 5000 #data.a1.param.stim1.stim_on[first_stim,0] - 300
    # end_time = start_time + 1000        #data.a1.param.stim1.stim_on[last_stim,1] + 300
    
    # data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    # data.a1.ge.get_centre_mass()
    # data.a1.ge.overlap_centreandspike()
    
    # data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    # data.a2.ge.get_centre_mass()
    # data.a2.ge.overlap_centreandspike()
    
    # frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    
    # stim = None 
    # adpt = None
    # ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
    #                                         frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    # savetitle = title.replace('\n','')
    
    # moviefile = savetitle+'_spon_%d'%loop_num+'.mp4'
    # #%
    # ani.save(save_fig_dir + moviefile)
    
    # del ani
    # #%%
    # '''no att'''
    # first_stim = 0 #1*n_perStimAmp -1; 
    # last_stim = 0 #1*n_perStimAmp
    # start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    # end_time = start_time + 1000        #data.a1.param.stim1.stim_on[last_stim,1] + 300
    
    # data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    # data.a1.ge.get_centre_mass()
    # data.a1.ge.overlap_centreandspike()
    
    # data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    # data.a2.ge.get_centre_mass()
    # data.a2.ge.overlap_centreandspike()
    
    # frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    # stim_on_off = data.a1.param.stim1.stim_on-start_time
    # stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    # stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    # adpt = None
    # ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
    #                                         frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    # savetitle = title.replace('\n','')
    
    # moviefile = savetitle+'_noatt_%d'%loop_num+'.mp4'
    
    # ani.save(save_fig_dir + moviefile)
    # #%%
    # del ani
    # #%%
    # '''att'''
    # first_stim = n_StimAmp*n_perStimAmp #1*n_perStimAmp -1 + n_perStimAmp*n_StimAmp; 
    # last_stim = n_StimAmp*n_perStimAmp #1*n_perStimAmp + n_perStimAmp*n_StimAmp
    # start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    # end_time = start_time + 1000        #data.a1.param.stim1.stim_on[last_stim,1] + 300
    
    # data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    # data.a1.ge.get_centre_mass()
    # data.a1.ge.overlap_centreandspike()
    
    # data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    # data.a2.ge.get_centre_mass()
    # data.a2.ge.overlap_centreandspike()
    
    # frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    # stim_on_off = data.a1.param.stim1.stim_on-start_time
    # stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    # stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]

    # adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[7]]]]
    # #adpt = None
    # ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
    #                                         frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    # savetitle = title.replace('\n','')
    
    # moviefile = savetitle+'_att_%d'%loop_num+'.mp4'
    # #%%
    # ani.save(save_fig_dir + moviefile)
    # del ani
    
#%%








