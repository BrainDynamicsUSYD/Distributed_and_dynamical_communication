# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:22:51 2023

@author: Shencong Ni

"""

'''
Simulations for 2-input condition with/without cue

To run simulations for 1-input condition, set stim2 = False at line 97.
'''



import brian2.numpy_ as np

from scipy import sparse
from brian2.only import *
import time
import mydata
import os
import datetime
import connection as cn
import poisson_stimuli as psti
import pre_process_sc
import preprocess_2area
import build_two_areas
import get_stim_scale
import adapt_gaussian
import adapt_logistic

import sys
import pickle

#%%
prefs.codegen.target = 'cython'
#%%
'''create folder for saving cache'''
dir_cache = './cache/cache%s' %sys.argv[1]
prefs.codegen.runtime.cython.cache_dir = dir_cache
prefs.codegen.max_cache_dir_size = 120
#%%
data_dir = 'raw_data/'
if not os.path.exists(data_dir):
    try: os.makedirs(data_dir)
    except FileExistsError:
        pass
#%%    **
sys_argv = int(sys.argv[1])
#%%
get_movie = True # if true, make and save movie
#%%
record_LFP = True # if true, record LFP signal
#%%
loop_num = -1

repeat = 30 # the number of total random realizations of the network;
for rp in [None]*repeat:
    loop_num += 1
    if loop_num == sys_argv: 
        # script input argument 'sys_argv' from 0 to repeat-1
        # each 'sys_argv' corresponds to one random realization of the network
        print('loop_num:',loop_num)
        break
    else: continue
    break

if loop_num != sys_argv: sys.exit("Error: wrong command line argument! \nThe argument must be an integer from 0 to %d; to change its upper bound, please change the 'repeat' variable."%(repeat-1))                    

const_seed = True # if True, manually set the seed of random number generator as the 'loop_num'
    
tau_k_ = 60 # ms; decay time constant of adaptation current

w_extnl_ = 5 # nS; synaptic coupling weight of external Poisson inputs
tau_s_r_ = 1 # ms; rising time constant of post-synaptic current


ie_r_i1 = 0.777076  # for tuning the IE ratio of the inhibitory neuron group in the bottom area (area 1; sensory V4)

ie_r_i2 = 0.6784  # for tuning the IE ratio of the inhibitory neuron group in the top area (area 2; association FEF) 


tau_s_di_ = 4.5 # ms; decay time constant of post-synaptic inhibitory current
tau_s_de_ = 5 # ms; decay time constant of post-synaptic excitatory current

delta_gk_1 = 1.9 # nS; baseline value of adaptation strength in sensory area

stim_dura = 10000 # ms; duration of the presentation of each stimulus on each trial
num_trials = 20 # number of trials

t_ref = 4 # ms; refractory period

adapt_change_shape = 'logi' # 'logi', 'gaus'; spatial profile of the adaptaion reduction  
chg_adapt_sharpness = 2.2 # sharpness of the spatial profile of the adaptaion reduction if adapt_change_shape = 'logi';


stim2 = True # if 'True', add 2 stimuli/inputs (one at the center of sensory area and another at the corner), otherwise 1 stimulus (at center)

pois_bckgrdExt = True # if True, use the Poisson spike train for the background external inputs
I_extnl_ = 0 #0.51 # nA; amplitude of the constant current of the background external inputs, taking effects only if pois_bckgrdExt = False



peak_p_e1_e2 = 0.4 # peak connection probability for e1 to e2 (e for excitatory neurons; i for inhibitory neurons; 1 for sensory area; 2 for association area)
peak_p_e1_i2 = 0.4 # peak connection probability for e1 to i2 
peak_p_e2_e1 = 0.4 # peak connection probability for e2 to e1 
peak_p_e2_i1 = 0.4 # peak connection probability for e2 to i1 


pois_extnl_r_1 = 8 # Hz; Poisson rate of each external neuron which provides background Poisson inputs to each neuron in sensory area
pois_extnl_r_2 = 8  # Hz; Poisson rate of each external neuron which provides background Poisson inputs to each neuron in association area
delta_gk_2 = 6.5  # nS; baseline value of adaptation strength in association area
new_delta_gk_2 = 0.5  # minimum of adaptation strength after reducing adaptation
ie_r_e2 = 0.7475*0.99 # used for tunning IE-ratio of excitatory neurons in association area
ie_r_e1 = 0.88 # used for tunning IE-ratio of excitatory neurons in sensory area
tau_p_d_e1_e2 = 8  # decay constant for e1-e2 inter-areal connection probability
tau_p_d_e1_i2 = tau_p_d_e1_e2 # decay constant for e1-i2 inter-areal connection probability
tau_p_d_e2_e1 = 8  # decay constant for e2-e1 inter-areal connection probability
tau_p_d_e2_i1 = tau_p_d_e2_e1  # decay constant for e2-i1 inter-areal connection probability 
scale_w_12_e = 2.6*0.5/((tau_p_d_e1_e2/6)**2)# for tunning e1-e2 connection strength
scale_w_12_i = scale_w_12_e # for tunning e1-i2 connection strength
scale_w_21_e = 0.35/4*0.5/((tau_p_d_e2_e1/13)**2) # for tunning e2-e1 connection strength
scale_w_21_i = scale_w_21_e*1 # for tunning e2-i1 connection strength                            
stim_amp =  400 # Hz; stimulus amplitude
chg_adapt_range =  8.2 # range of the adaptation strength reduction


if not pois_bckgrdExt:
    I_extnl_crt2e = I_extnl_ #0.51 0.40
    I_extnl_crt2i = I_extnl_ #0.51 0.40


#%%
if const_seed:
    seed_num = loop_num # set the seed of random number generator
    seed(seed_num)
else:
    seed_num = None
#%%
'''generate connectivities for sensory area (area 1)'''
def find_w_e(w_i, num_i, num_e, ie_ratio):
    return (w_i*num_i)/num_e/ie_ratio
               
def find_w_i(w_e, num_e, num_i, ie_ratio):
    return (w_e*num_e)*ie_ratio/num_i

w_ee_1 = 7.857 # nS; e1-e1 connection weight
w_ei_1 = 10.847 # e1-i1 connection weight

ie_r_e = 2.76*6.5/5.8; # 
ie_r_i = 2.450*6.5/5.8; #

ijwd1 = pre_process_sc.get_ijwd()
ijwd1.Ne = 64*64; ijwd1.Ni = 32*32
ijwd1.width = 64#

ijwd1.decay_p_ee = 7.5 # # decay constant of e to e connection probability as distance increases
ijwd1.decay_p_ei = 9.5 # decay constant of e to i connection probability as distance increases
ijwd1.decay_p_ie = 19 # decay constant of i to e connection probability as distance increases
ijwd1.decay_p_ii = 19 # decay constant of i to i connection probability as distance increases
ijwd1.delay = [0.5,2.5]



num_ee = 270 #  in-degree of e-e connection
num_ei = 350 # in-degree of e-i connection
num_ie = 130 #  in-degree of i-e connection
num_ii = 180 # in-degree of i-i connection



ijwd1.mean_SynNumIn_ee = num_ee    
ijwd1.mean_SynNumIn_ei = num_ei 
ijwd1.mean_SynNumIn_ie = num_ie  
ijwd1.mean_SynNumIn_ii = num_ii

ijwd1.w_ee_mean = w_ee_1 
ijwd1.w_ei_mean = w_ei_1  
ijwd1.w_ie_mean = find_w_i(w_ee_1, num_ee, num_ie, ie_r_e*ie_r_e1)   
ijwd1.w_ii_mean = find_w_i(w_ei_1, num_ei, num_ii, ie_r_i*ie_r_i1) 
print('bottom area: ee:%.3f, ei:%.3f, ie:%.3f, ii:%.3f'%(ijwd1.w_ee_mean, ijwd1.w_ei_mean, ijwd1.w_ie_mean, ijwd1.w_ii_mean))
#%     
ijwd1.generate_ijw() # generate connectivity and coupling weight; 
# 'i' represents the index of presynaptic neurons, 'j' the index of postsynaptic neurons, 'w' the coupling weights
ijwd1.generate_d_rand_lowerHigherBound() # generate the spike transmission time ('d') 
param_a1 = {**ijwd1.__dict__} # save parameters



del param_a1['i_ee'], param_a1['j_ee'], param_a1['w_ee'], param_a1['d_ee'], param_a1['dist_ee']  
del param_a1['i_ei'], param_a1['j_ei'], param_a1['w_ei'], param_a1['d_ei'], param_a1['dist_ei'] 
del param_a1['i_ie'], param_a1['j_ie'], param_a1['w_ie'], param_a1['d_ie'], param_a1['dist_ie'] 
del param_a1['i_ii'], param_a1['j_ii'], param_a1['w_ii'], param_a1['d_ii'], param_a1['dist_ii']
#%
'''generate connectivities for association area (area 2)'''

w_ee_2 = 11 
w_ei_2 = 13.805 

ie_r_e = 2.76*6.5/5.8;  
ie_r_i = 2.450*6.5/5.8;  

ijwd2 = pre_process_sc.get_ijwd()
ijwd2.Ne = 64*64; ijwd2.Ni = 32*32
ijwd2.width = 64#79

ijwd2.decay_p_ee = 7.5# decay constant of e to e connection probability as distance increases
ijwd2.decay_p_ei = 9.5# # decay constant of e to i connection probability as distance increases
ijwd2.decay_p_ie = 19# # decay constant of i to e connection probability as distance increases
ijwd2.decay_p_ii = 19# # decay constant of i to i connection probability as distance increases
ijwd2.delay = [0.5,2.5]

num_ee = 270 #285 # 240; 
num_ei = 350 #380 #320; 
num_ie = 130 #117 #130; # 150
num_ii = 180 #162 #180 # 180 170


ijwd2.mean_SynNumIn_ee = num_ee
ijwd2.mean_SynNumIn_ei = num_ei
ijwd2.mean_SynNumIn_ie = num_ie
ijwd2.mean_SynNumIn_ii = num_ii


ijwd2.w_ee_mean = w_ee_2 
ijwd2.w_ei_mean = w_ei_2 
ijwd2.w_ie_mean = find_w_i(w_ee_2, num_ee, num_ie, ie_r_e*ie_r_e2) 
ijwd2.w_ii_mean = find_w_i(w_ei_2, num_ei, num_ii, ie_r_i*ie_r_i2)
print('top area: ee:%.3f, ei:%.3f, ie:%.3f, ii:%.3f'%(ijwd2.w_ee_mean, ijwd2.w_ei_mean, ijwd2.w_ie_mean, ijwd2.w_ii_mean))
#%
ijwd2.generate_ijw()
ijwd2.generate_d_rand_lowerHigherBound()

param_a2 = {**ijwd2.__dict__}

del param_a2['i_ee'], param_a2['j_ee'], param_a2['w_ee'], param_a2['d_ee'], param_a2['dist_ee'] 
del param_a2['i_ei'], param_a2['j_ei'], param_a2['w_ei'], param_a2['d_ei'], param_a2['dist_ei']
del param_a2['i_ie'], param_a2['j_ie'], param_a2['w_ie'], param_a2['d_ie'], param_a2['dist_ie'] 
del param_a2['i_ii'], param_a2['j_ii'], param_a2['w_ii'], param_a2['d_ii'], param_a2['dist_ii']

#%
'''generate connectivity for inter-areal connections'''
ijwd_inter = preprocess_2area.get_ijwd_2()

ijwd_inter.Ne1 = 64*64; ijwd_inter.Ne2 = 64*64; 
ijwd_inter.width1 = 64; ijwd_inter.width2 = 64;
ijwd_inter.p_inter_area_1 = 1/2; ijwd_inter.p_inter_area_2 = 1/2
ijwd_inter.section_width_1 = 4;  ijwd_inter.section_width_2 = 4; 
ijwd_inter.peak_p_e1_e2 = peak_p_e1_e2; ijwd_inter.tau_p_d_e1_e2 = tau_p_d_e1_e2
ijwd_inter.peak_p_e1_i2 = peak_p_e1_i2; ijwd_inter.tau_p_d_e1_i2 = tau_p_d_e1_i2        
ijwd_inter.peak_p_e2_e1 = peak_p_e2_e1; ijwd_inter.tau_p_d_e2_e1 = tau_p_d_e2_e1
ijwd_inter.peak_p_e2_i1 = peak_p_e2_i1; ijwd_inter.tau_p_d_e2_i1 = tau_p_d_e2_i1

ijwd_inter.w_e1_e2_mean = 5*scale_w_12_e; ijwd_inter.w_e1_i2_mean = 5*scale_w_12_i
ijwd_inter.w_e2_e1_mean = 5*scale_w_21_e; ijwd_inter.w_e2_i1_mean = 5*scale_w_21_i

ijwd_inter.generate_ijwd() # generate connectivity, coupling weight and spike transmission time

print('e1e2:%.3f, e1i2:%.3f, e2e1:%.3f, e2i1:%.3f'%(ijwd_inter.w_e1_e2_mean, ijwd_inter.w_e1_i2_mean,ijwd_inter.w_e2_e1_mean,ijwd_inter.w_e2_i1_mean))

param_inter = {**ijwd_inter.__dict__}

del param_inter['i_e1_e2'], param_inter['j_e1_e2'], param_inter['w_e1_e2'], param_inter['d_e1_e2'] 
del param_inter['i_e1_i2'], param_inter['j_e1_i2'], param_inter['w_e1_i2'], param_inter['d_e1_i2'] 
del param_inter['i_e2_e1'], param_inter['j_e2_e1'], param_inter['w_e2_e1'], param_inter['d_e2_e1'] 
del param_inter['i_e2_i1'], param_inter['j_e2_i1'], param_inter['w_e2_i1'], param_inter['d_e2_i1']

#%%
start_scope()

twoarea_net = build_two_areas.two_areas()

group_e_1, group_i_1, syn_ee_1, syn_ei_1, syn_ie_1, syn_ii_1,\
group_e_2, group_i_2, syn_ee_2, syn_ei_2, syn_ie_2, syn_ii_2,\
syn_e1e2, syn_e1i2, syn_e2e1, syn_e2i1 = twoarea_net.build(ijwd1, ijwd2, ijwd_inter)
#%%
'''record LFP'''
        
if record_LFP:
    import get_LFP
    
    LFP_elec = np.array([[0,0],[-32,-32]]) # positions for LFP recordings
    i_LFP, j_LFP, w_LFP = get_LFP.get_LFP(ijwd2.e_lattice, LFP_elec, width = ijwd2.width, LFP_sigma = 7, LFP_effect_range = 2.5)
    
    group_LFP_record_1 = NeuronGroup(len(LFP_elec), model = get_LFP.LFP_recordneuron)
    syn_LFP_1 = Synapses(group_e_1, group_LFP_record_1, model = get_LFP.LFP_syn)
    syn_LFP_1.connect(i=i_LFP, j=j_LFP)
    syn_LFP_1.w[:] = w_LFP[:]

    group_LFP_record_2 = NeuronGroup(len(LFP_elec), model = get_LFP.LFP_recordneuron)
    syn_LFP_2 = Synapses(group_e_2, group_LFP_record_2, model = get_LFP.LFP_syn)
    syn_LFP_2.connect(i=i_LFP, j=j_LFP)
    syn_LFP_2.w[:] = w_LFP[:]
#%%

chg_adapt_loca = [0, 0]

'''gaussian shape'''
if adapt_change_shape == 'gaus':
    adapt_value_new = adapt_gaussian.get_adaptation(base_amp = delta_gk_2, \
        max_decrease = [delta_gk_2-new_delta_gk_2], sig=[chg_adapt_range], position=[chg_adapt_loca], n_side=int(round((ijwd2.Ne)**0.5)), width=ijwd2.width)

    '''logistic shape'''

elif adapt_change_shape == 'logi':
    
    adapt_value_new = adapt_logistic.get_adaptation(base_amp = delta_gk_2, max_decrease = [delta_gk_2-new_delta_gk_2], \
                                                rang = [chg_adapt_range], sharpness =[chg_adapt_sharpness], position=[chg_adapt_loca],\
                                                    n_side=round((ijwd2.Ne)**0.5), width=ijwd2.width)


#%%


#%%
'''stimulus'''

'''stim 1'''
'''uncued'''
stim_scale_cls = get_stim_scale.get_stim_scale()
stim_scale_cls.seed = 10
n_perStimAmp = num_trials # number of trials for each stimulus strength
if not isinstance(stim_amp, np.ndarray):
    stim_amp = np.array([stim_amp])
n_StimAmp = stim_amp.shape[0] # number of different stimulus strength

stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = stim_amp[i]/200 # 200 Hz is the default rate

stim_scale_cls.stim_amp_scale = stim_amp_scale
stim_scale_cls.stim_dura = stim_dura
stim_scale_cls.separate_dura = np.array([800,1500]) # ms; time interval between two trials
stim_scale_cls.get_scale()
stim_scale_cls.n_StimAmp = n_StimAmp
stim_scale_cls.n_perStimAmp = n_perStimAmp
stim_scale_cls.stim_amp = stim_amp


transient = 10000 # ms; onset time of the first trial
init = np.zeros(transient//stim_scale_cls.dt_stim)
stim_scale_cls.scale_stim = np.concatenate((init, stim_scale_cls.scale_stim))
stim_scale_cls.stim_on += transient
'''cued'''
stim_scale_cls_att = get_stim_scale.get_stim_scale()
stim_scale_cls_att.seed = 15
n_StimAmp = stim_amp.shape[0]
n_perStimAmp = num_trials
stim_amp_scale = np.ones(n_StimAmp*n_perStimAmp)
for i in range(n_StimAmp):
    stim_amp_scale[i*n_perStimAmp:i*n_perStimAmp+n_perStimAmp] = stim_amp[i]/200

stim_scale_cls_att.stim_amp_scale = stim_amp_scale
stim_scale_cls_att.stim_dura = stim_dura
stim_scale_cls_att.separate_dura = np.array([800,1500])
stim_scale_cls_att.get_scale()

inter_time = 4000 # ms; time interval between uncued trial block and cued trial block
suplmt = (inter_time // stim_scale_cls.dt_stim) - (stim_scale_cls.scale_stim.shape[0] - stim_scale_cls.stim_on[-1,1] // stim_scale_cls.dt_stim) # supply '0' between non-attention and attention stimuli amplitude

stim_scale_cls.scale_stim = np.concatenate((stim_scale_cls.scale_stim, np.zeros(suplmt), stim_scale_cls_att.scale_stim))
stim_scale_cls.stim_amp_scale = np.concatenate((stim_scale_cls.stim_amp_scale, stim_scale_cls_att.stim_amp_scale))
stim_scale_cls_att.stim_on += stim_scale_cls.stim_on[-1,1] + inter_time
stim_scale_cls.stim_on = np.vstack((stim_scale_cls.stim_on, stim_scale_cls_att.stim_on))
#%%
scale_1 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)
data_ = mydata.mydata()
param_a1 = {**param_a1, 'stim1':data_.class2dict(stim_scale_cls)}
#%%
'''stim 2'''
if stim2:
    scale_2 = TimedArray(stim_scale_cls.scale_stim, dt=10*ms)
else:
    scale_2 = TimedArray(np.zeros(stim_scale_cls.scale_stim.shape), dt=10*ms)

data_ = mydata.mydata()
param_a1 = {**param_a1, 'stim2':data_.class2dict(stim_scale_cls)}

#%%
posi_stim_e1 = NeuronGroup(ijwd1.Ne, \
                        '''rates =  bkg_rates + stim_1*scale_1(t) + stim_2*scale_2(t) : Hz
                        bkg_rates : Hz
                        stim_1 : Hz
                        stim_2 : Hz
                        ''', threshold='rand()<rates*dt')

posi_stim_e1.bkg_rates = 0*Hz
posi_stim_e1.stim_1 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]])*Hz
posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, -32]])*Hz

synapse_e_extnl = cn.model_neu_syn_AD.synapse_e_AD
syn_extnl_e1 = Synapses(posi_stim_e1, group_e_1, model=synapse_e_extnl, on_pre='x_E_extnl_post += w')
syn_extnl_e1.connect('i==j')
syn_extnl_e1.w = w_extnl_*nS

#%%
'''background Poisson Input; '''

if pois_bckgrdExt:
    pois_bkgExt_e1 = PoissonInput(group_e_1, 'x_E_extnl', 200, pois_extnl_r_1*Hz, weight=5*nS)
    pois_bkgExt_i1 = PoissonInput(group_i_1, 'x_E_extnl', 200, pois_extnl_r_1*Hz, weight=5*nS)
    
    pois_bkgExt_e2 = PoissonInput(group_e_2, 'x_E_extnl', 200, pois_extnl_r_2*Hz, weight=5*nS)
    pois_bkgExt_i2 = PoissonInput(group_i_2, 'x_E_extnl', 200, pois_extnl_r_2*Hz, weight=5*nS)

#%%

if const_seed:
    seed(seed_num)

group_e_1.tau_s_de = tau_s_de_*ms; 
group_e_1.tau_s_di = tau_s_di_*ms
group_e_1.tau_s_re = group_e_1.tau_s_ri = tau_s_r_*ms

group_e_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_e_1.tau_s_re_inter = tau_s_r_*ms
group_e_1.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_e_1.tau_s_re_extnl = tau_s_r_*ms

group_i_1.tau_s_de = tau_s_de_*ms
group_i_1.tau_s_di = tau_s_di_*ms
group_i_1.tau_s_re = group_i_1.tau_s_ri = tau_s_r_*ms

group_i_1.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_1.tau_s_re_inter = tau_s_r_*ms
group_i_1.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_i_1.tau_s_re_extnl = tau_s_r_*ms


group_e_1.v = np.random.random(ijwd1.Ne)*35*mV-85*mV
group_i_1.v = np.random.random(ijwd1.Ni)*35*mV-85*mV
group_e_1.delta_gk = delta_gk_1*nS
group_e_1.tau_k = tau_k_*ms
if pois_bckgrdExt:
    group_e_1.I_extnl_crt = 0*nA # 0.25 0.51*nA 0.35
    group_i_1.I_extnl_crt = 0*nA # 0.25 0.60*nA 0.35

else:
    group_e_1.I_extnl_crt = I_extnl_crt2e*nA # 0.25 0.51*nA 0.35
    group_i_1.I_extnl_crt = I_extnl_crt2i*nA # 0.25 0.60*nA 0.35


group_e_2.tau_s_de = tau_s_de_*ms; 
group_e_2.tau_s_di = tau_s_di_*ms
group_e_2.tau_s_re = group_e_2.tau_s_ri = tau_s_r_*ms

group_e_2.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_e_2.tau_s_re_inter = tau_s_r_*ms
group_e_2.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_e_2.tau_s_re_extnl = tau_s_r_*ms

group_i_2.tau_s_de = tau_s_de_*ms
group_i_2.tau_s_di = tau_s_di_*ms
group_i_2.tau_s_re = group_i_2.tau_s_ri = tau_s_r_*ms

group_i_2.tau_s_de_inter = tau_s_de_*ms #5.0*ms; 
group_i_2.tau_s_re_inter = tau_s_r_*ms
group_i_2.tau_s_de_extnl = tau_s_de_*ms #5.0*ms
group_i_2.tau_s_re_extnl = tau_s_r_*ms

group_e_2.v = np.random.random(ijwd2.Ne)*35*mV-85*mV
group_i_2.v = np.random.random(ijwd2.Ni)*35*mV-85*mV
group_e_2.delta_gk = delta_gk_2*nS
group_e_2.tau_k = tau_k_*ms
if pois_bckgrdExt:
    group_e_2.I_extnl_crt = 0*nA #0.51*nA  0.40 0.35
    group_i_2.I_extnl_crt = 0*nA #0.60*nA  0.40 0.35
else:
    group_e_2.I_extnl_crt = I_extnl_crt2e*nA #0.51*nA  0.40 0.35
    group_i_2.I_extnl_crt = I_extnl_crt2i*nA #0.60*nA  0.40 0.35
    
#%%
spk_e_1 = SpikeMonitor(group_e_1, record = True)
spk_i_1 = SpikeMonitor(group_i_1, record = True)
spk_e_2 = SpikeMonitor(group_e_2, record = True)
spk_i_2 = SpikeMonitor(group_i_2, record = True)


if record_LFP:
    lfp_moni_1 = StateMonitor(group_LFP_record_1, ('lfp'), dt = 1*ms, record = True)
    lfp_moni_2 = StateMonitor(group_LFP_record_2, ('lfp'), dt = 1*ms, record = True)

#%%

C = 0.25*nF # capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -70*mV # reset membrane potential
v_rev_I = -80*mV # # reversal membrane potential for inhibitory synaptic current
v_rev_E = 0*mV # reversal membrane potential for excitatory synaptic current
v_k = -85*mV # reversal membrane potential for adaptation (potassium) current


group_e_2.g_l = 16.7*nS #16.7*nS # leaky conductance
group_i_2.g_l = 25*nS # leaky conductance

group_e_2.t_ref = t_ref*ms # refractory period
group_i_2.t_ref = t_ref*ms # 

group_e_1.g_l = 16.7*nS #16.7*nS # leaky conductance
group_i_1.g_l = 25*nS # leaky conductance

group_e_1.t_ref = t_ref*ms #
group_i_1.t_ref = t_ref*ms # 


#%%
net = Network(collect())
net.store('state1')


#%%
tic = time.perf_counter()
simu_time_tot = (stim_scale_cls.stim_on[-1,1] + 500)*ms
simu_time1 = (stim_scale_cls.stim_on[n_StimAmp*n_perStimAmp-1,1] + 2000)*ms

simu_time2 = simu_time_tot - simu_time1

'''run uncued trials'''
net.run(simu_time1, profile=False) #,namespace={'tau_k': 80*ms}

'''run cued trials'''
group_e_2.delta_gk[:] = adapt_value_new*nS # set the new spike frequency adaptation for the cued condition

net.run(simu_time2, profile=False) #,namespace={'tau_k': 80*ms}

print('total time elapsed:',np.round((time.perf_counter() - tic)/60,2), 'min')

#%%
'''save data'''
now = datetime.datetime.now()

param_all = {'I_extnl':I_extnl_,
             'pois_extnl_r_1':pois_extnl_r_1,
             'pois_extnl_r_2':pois_extnl_r_2,
        'delta_gk_1':delta_gk_1,
             'delta_gk_2':delta_gk_2,
         'new_delta_gk_2':new_delta_gk_2,
         'tau_k': tau_k_,
         'stim2': stim2,
         'tau_s_di':tau_s_di_,
         'tau_s_de':tau_s_de_,
         'tau_s_r':tau_s_r_,
         'num_ee':num_ee,
         'num_ei':num_ei,
         'num_ii':num_ii,
         'num_ie':num_ie,
         'simutime':int(round(simu_time_tot/ms)),
         'chg_adapt_range': chg_adapt_range,
         'chg_adapt_loca': chg_adapt_loca,
         'ie_r_e': ie_r_e,
         'ie_r_e1':ie_r_e1,   
         'ie_r_e2':ie_r_e2,
         'ie_r_i': ie_r_i,
         'ie_r_i1': ie_r_i1,
         'ie_r_i2': ie_r_i2,
         't_ref': t_ref,
         'stim_dura': stim_dura,
         'seed_num':seed_num}

if adapt_change_shape == 'logi':
    param_all['chg_adapt_sharpness'] = chg_adapt_sharpness

def get_sparse_spk_matrix(i, t, shape):    
    spk_matrix = sparse.coo_matrix((np.ones(i.shape[0],dtype=bool),(i,t)), shape=shape)
    return spk_matrix.tocsr()

simu_time_tot = int(round(simu_time_tot/ms))

spk_tstep_e1 = np.round(spk_e_1.t/(0.1*ms)).astype(int)
spk_tstep_i1 = np.round(spk_i_1.t/(0.1*ms)).astype(int)
spk_tstep_e2 = np.round(spk_e_2.t/(0.1*ms)).astype(int)
spk_tstep_i2 = np.round(spk_i_2.t/(0.1*ms)).astype(int)

spk_mat_e1 = get_sparse_spk_matrix(spk_e_1.i[:], spk_tstep_e1, [ijwd1.Ne, simu_time_tot*10])
spk_mat_i1 = get_sparse_spk_matrix(spk_i_1.i[:], spk_tstep_i1, [ijwd1.Ni, simu_time_tot*10])
spk_mat_e2 = get_sparse_spk_matrix(spk_e_2.i[:], spk_tstep_e2, [ijwd2.Ne, simu_time_tot*10])
spk_mat_i2 = get_sparse_spk_matrix(spk_i_2.i[:], spk_tstep_i2, [ijwd2.Ni, simu_time_tot*10])


data_save = {'datetime':now.strftime("%Y-%m-%d %H:%M:%S"), 'dt':0.1, 'loop_num':loop_num, 'data_dir': os.getcwd(),
        'param':param_all,
        'a1':{'param':param_a1,
              'ge':{'t_ind': spk_mat_e1.indices, 't_indptr': spk_mat_e1.indptr},    
              'gi':{'t_ind': spk_mat_i1.indices, 't_indptr': spk_mat_i1.indptr}},
        'a2':{'param':param_a2,
              'ge':{'t_ind': spk_mat_e2.indices, 't_indptr': spk_mat_e2.indptr},
              'gi':{'t_ind': spk_mat_i2.indices, 't_indptr': spk_mat_i2.indptr}},
        'inter':{'param':param_inter}}

if record_LFP:
    data_save['a1']['ge']['LFP'] = lfp_moni_1.lfp[:]/nA
    data_save['a2']['ge']['LFP'] = lfp_moni_2.lfp[:]/nA


with open(data_dir+'data%d.file'%loop_num, 'wb') as file:
    pickle.dump(data_save, file)


#%%

if get_movie:
    import firing_rate_analysis as fra
    import matplotlib.pyplot as plt
    
    '''load data'''
    data = mydata.mydata()
    data.load(data_dir+'data%d.file'%loop_num)
    
    simu_time_tot = data.param.simutime
    data.a1.ge.get_sparse_spk_matrix_csrindptr([data.a1.param.Ne, simu_time_tot*10], mat_type='csc')
    data.a2.ge.get_sparse_spk_matrix_csrindptr([data.a2.param.Ne, simu_time_tot*10], mat_type='csc')
    data.a1.ge.get_spk_it()
    data.a2.ge.get_spk_it()
    
    ''' make movie '''
    
    #%
    chg_adapt_range = data.param.chg_adapt_range
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

    
