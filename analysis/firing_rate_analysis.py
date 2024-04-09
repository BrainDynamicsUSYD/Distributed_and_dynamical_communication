# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:18:39 2020

@author: Shencong Ni
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.stats import sem
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import connection as cn
#%%
def get_spike_rate(spike, start_time, end_time, indiv_rate = False, popu_rate = False, \
                   sample_interval = 1, n_neuron = 4096, window = 10, dt = 0.1, reshape_indiv_rate = True, save_results_to_input = False):
    '''
    input: 
    spike: an object that contains timestep index and neuron index for spikes
    start_time: (ms) the time in recorded spike data to start to calculate the firing rate 
    end_time: (ms) the time in recorded spike data to stop the calculation of firing rate 
    indiv_rate: whether to output the number of spikes for each neuron in 'window'
    popu_rate: whether to output the population average firing rate
    sample_interval: (ms) time interval between each firing rate recording point
    n_neuron: total number of neurons
    window: (ms) length of window to count spikes
    dt: (ms) simulation time step
    reshape_indiv_rate: (Boolean), whether to reshape the indiv_rate to grid-shape (N*N*t array, N=sqrt(n_neuron), t=number of sampled time points)
    save_results_to_input: whether to save output as the attributes of input object-"spike"
    output:
    indiv_rate: the number of spikes for each neuron within the period of 'window' at each sampling point 
    pop_rate: population average firing rate
    '''

    sample_interval = int(np.round(sample_interval/dt))
    start_time = int(np.round(start_time/dt))
    end_time = int(np.round(end_time/dt))
    window_step = int(np.round(window/dt))

    sample_t = np.arange(0, end_time-start_time-window_step+1, sample_interval)

    spiket = spike.t #np.round(spike.t/dt).astype(int)
    spikei = spike.i[(spiket >= start_time) & (spiket < end_time)]
    spiket = spiket[(spiket >= start_time) & (spiket < end_time)]

    spk_mat = csc_matrix((np.ones(len(spikei),dtype=int),(spikei,spiket-start_time)),(n_neuron,end_time-start_time))
    spk_rate = np.zeros([n_neuron, len(sample_t)], dtype=np.int8)
        
    for i in range(len(sample_t)):
        
        neu, counts = np.unique(spk_mat.indices[spk_mat.indptr[sample_t[i]]:spk_mat.indptr[sample_t[i]+window_step]],return_counts=True)
        spk_rate[:, i][neu] += counts
    
    #for stp in range(spk.shape[1]):
#    for t in range(len(sample_t)):    
#        neu, counts = np.unique(spike.i[(t_spk >= (sample_t[t])) & (t_spk < (window_step+sample_t[t]))], return_counts=True)
#        spk_rate[:, t][neu] += counts
    if indiv_rate:
        if reshape_indiv_rate:
            spk_rate = spk_rate.reshape(int(n_neuron**0.5),int(n_neuron**0.5),spk_rate.shape[1])
            pop_rate = (spk_rate.sum(0).sum(0))/(window/(1000))/n_neuron
        else:
            pop_rate = (spk_rate.sum(0))/(window/(1000))/n_neuron
    else:
        pop_rate = spk_rate.sum(0)/(window/(1000))/n_neuron
    
    if not(save_results_to_input):
        if indiv_rate and (not popu_rate): return spk_rate
        elif (not indiv_rate) and popu_rate: return pop_rate
        elif indiv_rate and popu_rate: return spk_rate, pop_rate
        else: return 0
    else:
        if indiv_rate and (not popu_rate):
            spike.spk_rate = spk_rate
            return spike        
        elif (not indiv_rate) and popu_rate: 
            spike.pop_rate = pop_rate
            return spike
        elif indiv_rate and popu_rate:           
            spike.spk_rate = spk_rate
            spike.pop_rate = pop_rate
            return spike
        else: return 0
#%%
"""
def get_centre_mass(spk, dt, slide_interval, jump_interval):
    '''
    input:
    spk: N*N*t array; a 3D array containing the number of spikes for each neuron in a short period of time window.
    dt: (ms) sampling interval used in "spk"
    slide_interval: (ms) the time interval between two successive centre-of-mass calculation.
    jump_interval: (ms) the time interval between two successive centre-of-mass jump distance calculation
    output:
    centre_ind: coordinate of the centre of mass (use the format of matrix coordinate) 
    centre: exact position of the centre of mass
    jump_size: jump size of two centre-of-mass with time interval "jump_interval" (can be possitive or negative, cooresponding to different jumping directions) 
    jump_dist: the obsolute value of jump distance of two successive centre-of-mass (always positive)    
    '''  
#        slide_interval = round(slide_interval/dt)
#        jump_interval = round(jump_interval/dt)
    slide_interval = int(np.round(slide_interval/dt))
    len_hori = spk.shape[1]; len_vert = spk.shape[0];
    x_hori = np.cos(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    y_hori = np.sin(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    xy_hori = np.concatenate((x_hori,y_hori),axis=1)
    
    x_vert = np.cos(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    y_vert = np.sin(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    xy_vert = np.concatenate((x_vert,y_vert),axis=1)
    
    slide_ary = np.arange(0,spk.shape[2],slide_interval)
    centre_ind = np.zeros([len(slide_ary), 2], dtype=np.int16)
    #jump_size = np.zeros([len(slide_ary), 2])
    centre = np.zeros([len(slide_ary), 2], dtype=float)
    for ind in range(len(slide_ary)):
        if np.all(spk[:,:,slide_ary[ind]] == 0):
            if ind == 0:
                centre[ind] = np.array([0,0])
                ind_vert = int(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi))
                ind_hori = int(wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi))
                centre_ind[ind] = [ind_vert, ind_hori]
            else:
                centre[ind] = centre[ind-1]
                centre_ind[ind] = centre_ind[ind-1]
        else:
            sum_hori = np.sum(spk[:,:,slide_ary[ind]], axis=0)
            sum_vert = np.sum(spk[:,:,slide_ary[ind]], axis=1)
            ctr_hori = np.dot(sum_hori, xy_hori)
            ctr_vert = np.dot(sum_vert, xy_vert)
            '''
            if ctr_hori[1] >= 0:
                ind_hori = int((npa.arctan2(ctr_hori[1],ctr_hori[0])*len_hori)/(2*np.pi))
            else:
                ind_hori = int(((2*np.pi+np.arctan2(ctr_hori[1],ctr_hori[0]))*len_hori)/(2*np.pi))
            if ctr_vert[1] >= 0:
                ind_vert = int((np.arctan2(ctr_vert[1],ctr_vert[0])*len_vert)/(2*np.pi))
            else:
                ind_vert = int(((2*np.pi+np.arctan2(ctr_vert[1],ctr_vert[0]))*len_vert)/(2*np.pi))
            '''
            centre[ind,1] = np.angle(np.array([ctr_hori[0] + 1j*ctr_hori[1]]))[0]
            centre[ind,0] = np.angle(np.array([ctr_vert[0] + 1j*ctr_vert[1]]))[0]
            ind_vert = int(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi))
    #        if ind_vert > 62: 
    #            ind_vert = 62
            ind_hori = int(wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi))
    #        if ind_hori > 62: 
    #            ind_hori = 62
            centre_ind[ind] = [ind_vert, ind_hori]
            
    jump_size = centre[int(jump_interval/slide_interval):, :] - centre[:len(centre)-int(jump_interval/slide_interval), :]
    jump_size = wrapToPi(jump_size)
    jump_size[:,0] = jump_size[:,0]*len_vert/(2*np.pi)
    jump_size[:,1] = jump_size[:,1]*len_hori/(2*np.pi)
    jump_dist = np.sqrt(np.sum(jump_size**2,1))
    centre[:,0] = wrapTo2Pi(centre[:,0])*len_vert/(2*np.pi)
    centre[:,1] = wrapTo2Pi(centre[:,1])*len_hori/(2*np.pi)
    
    return centre_ind, centre, jump_size, jump_dist
"""
#%%
def get_centre_mass_2(spk, dt, slide_interval, jump_interval, detect_pattern=False):
    '''
    input:
    spk: N*N*t array; a 3D array containing the number of spikes for each neuron in a short period of time window.
    dt: (ms) sampling interval used in "spk"
    slide_interval: (ms) the time interval between two successive centre-of-mass calculation.
    jump_interval: (ms) the time interval between two successive centre-of-mass jump distance calculation
    output:
    centre_ind: coordinate of the centre of mass (use the format of matrix coordinate) 
    centre: exact position of the centre of mass
    jump_size: jump size of two centre-of-mass with time interval "jump_interval" (can be possitive or negative, cooresponding to different jumping directions) 
    jump_dist: the obsolute value of jump distance of two successive centre-of-mass (always positive)    
    '''  
#        slide_interval = round(slide_interval/dt)
#        jump_interval = round(jump_interval/dt)
    slide_interval = int(np.round(slide_interval/dt))
    len_hori = spk.shape[1]; len_vert = spk.shape[0];
    x_hori = np.cos(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    y_hori = np.sin(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    xy_hori = np.concatenate((x_hori,y_hori),axis=1)
    
    x_vert = np.cos(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    y_vert = np.sin(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    xy_vert = np.concatenate((x_vert,y_vert),axis=1)
    
    slide_ary = np.arange(0,spk.shape[2],slide_interval)
    centre_ind = np.zeros([len(slide_ary), 2], dtype=np.int16)
    #jump_size = np.zeros([len(slide_ary), 2])
    centre = np.zeros([len(slide_ary), 2], dtype=float)
    if detect_pattern:
        pattern_size = np.zeros(slide_ary.shape)
        pattern = np.zeros(slide_ary.shape, bool)
        lattice = cn.coordination.makelattice(len_hori, len_hori, [int(round(len_hori/2))-0.5,int(round(len_hori/2))-0.5])
        hw_h = 0.5*(len_hori); hw_v = 0.5*(len_vert)
        hori_n = np.arange(len_hori)
        vert_n = np.arange(len_vert)

    for ind in range(len(slide_ary)):
        if np.all(spk[:,:,slide_ary[ind]] == 0):
            if ind == 0:
                centre[ind] = np.array([0,0])
                ind_vert = int(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi))
                ind_hori = int(wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi))
                centre_ind[ind] = [ind_vert, ind_hori]
                if detect_pattern:
                    pattern[ind] = False
                    pattern_size[ind] = 0
            else:
                centre[ind] = centre[ind-1]
                centre_ind[ind] = centre_ind[ind-1]
                if detect_pattern:
                    pattern[ind] = False
                    pattern_size[ind] = 0
        else:
            sum_hori = np.sum(spk[:,:,slide_ary[ind]], axis=0)
            sum_vert = np.sum(spk[:,:,slide_ary[ind]], axis=1)
            ctr_hori = np.dot(sum_hori, xy_hori)
            ctr_vert = np.dot(sum_vert, xy_vert)
            '''
            if ctr_hori[1] >= 0:
                ind_hori = int((npa.arctan2(ctr_hori[1],ctr_hori[0])*len_hori)/(2*np.pi))
            else:
                ind_hori = int(((2*np.pi+np.arctan2(ctr_hori[1],ctr_hori[0]))*len_hori)/(2*np.pi))
            if ctr_vert[1] >= 0:
                ind_vert = int((np.arctan2(ctr_vert[1],ctr_vert[0])*len_vert)/(2*np.pi))
            else:
                ind_vert = int(((2*np.pi+np.arctan2(ctr_vert[1],ctr_vert[0]))*len_vert)/(2*np.pi))
            '''
            centre[ind,1] = np.angle(np.array([ctr_hori[0] + 1j*ctr_hori[1]]))[0]
            centre[ind,0] = np.angle(np.array([ctr_vert[0] + 1j*ctr_vert[1]]))[0]
            # ind_vert = int(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi))
            ind_vert = wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi)
            if ind_vert >= len_vert-0.5:
                ind_vert = int(ind_vert - len_vert)
            else:
                ind_vert = int(ind_vert)            
            
    #        if ind_vert > 62: 
    #            ind_vert = 62
            # ind_hori = int(wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi))
            ind_hori = wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi)
              
            if ind_hori >= len_hori-0.5:
                ind_hori = int(ind_hori - len_hori)
            else:
                ind_hori = int(ind_hori)  
                
    #        if ind_hori > 62: 
    #            ind_hori = 62
            centre_ind[ind] = [ind_vert, ind_hori]
            if detect_pattern:
                size_hori = np.sqrt(np.sum(sum_hori*((hori_n - centre_ind[ind,1] + hw_h)%(2*hw_h) - hw_h)**2)/sum_hori.sum())
                size_vert = np.sqrt(np.sum(sum_vert*((vert_n - centre_ind[ind,0] + hw_v)%(2*hw_v) - hw_v)**2)/sum_vert.sum())
                size = np.max([size_hori, size_vert])                
                neuron = cn.findnearbyneuron.findnearbyneuron(lattice, [centre_ind[ind,1], len_vert-1-centre_ind[ind,0]], size, len_hori)
                if spk[:,:,slide_ary[ind]].reshape(-1)[neuron].sum() > ((np.pi*size**2)*0.5): # 0.7: threshold
                    pattern[ind] = True
                
                pattern_size[ind] = size   

            
    jump_size = centre[int(jump_interval/slide_interval):, :] - centre[:len(centre)-int(jump_interval/slide_interval), :]
    jump_size = wrapToPi(jump_size)
    jump_size[:,0] = jump_size[:,0]*len_vert/(2*np.pi)
    jump_size[:,1] = jump_size[:,1]*len_hori/(2*np.pi)
    jump_dist = np.sqrt(np.sum(jump_size**2,1))
    centre[:,0] = wrapTo2Pi(centre[:,0])*len_vert/(2*np.pi)
    centre[:,1] = wrapTo2Pi(centre[:,1])*len_hori/(2*np.pi)
    
    centre[:,0][centre[:,0] >= len_vert-0.5] -= len_vert
    centre[:,1][centre[:,1] >= len_hori-0.5] -= len_hori
    
    if not detect_pattern:
        return centre_ind, centre, jump_size, jump_dist
    else :
        return centre_ind, centre, jump_size, jump_dist, pattern, pattern_size

#%%
def get_centre_mass(spk, dt, slide_interval, jump_interval, detect_pattern=False):
    '''
    input:
    spk: N*N*t array; a 3D array containing the number of spikes for each neuron in a short period of time window.
    dt: (ms) sampling interval used in "spk"
    slide_interval: (ms) the time interval between two successive centre-of-mass calculation.
    jump_interval: (ms) the time interval between two successive centre-of-mass jump distance calculation
    output:
    centre_ind: coordinate of the centre of mass (use the format of matrix coordinate) 
    centre: exact position of the centre of mass
    jump_size: jump size of two centre-of-mass with time interval "jump_interval" (can be possitive or negative, cooresponding to different jumping directions) 
    jump_dist: the obsolute value of jump distance of two successive centre-of-mass (always positive)    
    '''  
#        slide_interval = round(slide_interval/dt)
#        jump_interval = round(jump_interval/dt)
    slide_interval_step = int(np.round(slide_interval/dt)); #print(slide_interval_step)
    len_hori = spk.shape[1]; len_vert = spk.shape[0];
    x_hori = np.cos((np.arange(len_hori)+0.5)/len_hori*2*np.pi).reshape([-1,1])
    y_hori = np.sin((np.arange(len_hori)+0.5)/len_hori*2*np.pi).reshape([-1,1])
    xy_hori = np.concatenate((x_hori,y_hori),axis=1)
    
    x_vert = np.cos((np.arange(len_vert)+0.5)/len_vert*2*np.pi).reshape([-1,1])
    y_vert = np.sin((np.arange(len_vert)+0.5)/len_vert*2*np.pi).reshape([-1,1])
    xy_vert = np.concatenate((x_vert,y_vert),axis=1)
    
    slide_ary = np.arange(0,spk.shape[2],slide_interval_step)
    centre_ind = np.zeros([len(slide_ary), 2], dtype=np.int16)
    #jump_size = np.zeros([len(slide_ary), 2])
    centre = np.zeros([len(slide_ary), 2], dtype=float)
    if detect_pattern:
        pattern_size = np.zeros(slide_ary.shape)
        pattern = np.zeros(slide_ary.shape, bool)
        lattice = cn.coordination.makelattice(len_hori, len_hori, [int(round(len_hori/2))-0.5,int(round(len_hori/2))-0.5])
        hw_h = 0.5*(len_hori); hw_v = 0.5*(len_vert)
        hori_n = np.arange(len_hori)
        vert_n = np.arange(len_vert)

    for ind in range(len(slide_ary)):
        if np.all(spk[:,:,slide_ary[ind]] == 0):
            if ind == 0:
                centre[ind] = np.array([0,0])
                ind_vert = int(np.round(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi)-0.5))
                ind_hori = int(np.round(wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi)-0.5))
                centre_ind[ind] = [ind_vert, ind_hori]
                if detect_pattern:
                    pattern[ind] = False
                    pattern_size[ind] = 0
            else:
                centre[ind] = centre[ind-1]
                centre_ind[ind] = centre_ind[ind-1]
                if detect_pattern:
                    pattern[ind] = False
                    pattern_size[ind] = 0
        else:
            sum_hori = np.sum(spk[:,:,slide_ary[ind]], axis=0)
            sum_vert = np.sum(spk[:,:,slide_ary[ind]], axis=1)
            ctr_hori = np.dot(sum_hori, xy_hori)
            ctr_vert = np.dot(sum_vert, xy_vert)
            '''
            if ctr_hori[1] >= 0:
                ind_hori = int((npa.arctan2(ctr_hori[1],ctr_hori[0])*len_hori)/(2*np.pi))
            else:
                ind_hori = int(((2*np.pi+np.arctan2(ctr_hori[1],ctr_hori[0]))*len_hori)/(2*np.pi))
            if ctr_vert[1] >= 0:
                ind_vert = int((np.arctan2(ctr_vert[1],ctr_vert[0])*len_vert)/(2*np.pi))
            else:
                ind_vert = int(((2*np.pi+np.arctan2(ctr_vert[1],ctr_vert[0]))*len_vert)/(2*np.pi))
            '''
            centre[ind,1] = np.angle(np.array([ctr_hori[0] + 1j*ctr_hori[1]]))[0]
            centre[ind,0] = np.angle(np.array([ctr_vert[0] + 1j*ctr_vert[1]]))[0]
            # ind_vert = int(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi))
            ind_vert = int(np.round(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi)-0.5))
          
            ind_hori = int(np.round(wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi)-0.5))
                            
            centre_ind[ind] = [ind_vert, ind_hori]
            if detect_pattern:
                size_hori = np.sqrt(np.sum(sum_hori*((hori_n - centre_ind[ind,1] + hw_h)%(2*hw_h) - hw_h)**2)/sum_hori.sum())
                size_vert = np.sqrt(np.sum(sum_vert*((vert_n - centre_ind[ind,0] + hw_v)%(2*hw_v) - hw_v)**2)/sum_vert.sum())
                size = np.max([size_hori, size_vert])                
                neuron = cn.findnearbyneuron.findnearbyneuron(lattice, [centre_ind[ind,1], len_vert-1-centre_ind[ind,0]], size, len_hori)
                if spk[:,:,slide_ary[ind]].reshape(-1)[neuron].sum() > ((np.pi*size**2)*0.5): # 0.7: threshold
                    pattern[ind] = True
                
                pattern_size[ind] = size   

            
    jump_size = centre[round(jump_interval/slide_interval):, :] - centre[:len(centre)-round(jump_interval/slide_interval), :]
    jump_size = wrapToPi(jump_size)
    jump_size[:,0] = jump_size[:,0]*len_vert/(2*np.pi)
    jump_size[:,1] = jump_size[:,1]*len_hori/(2*np.pi)
    jump_dist = np.sqrt(np.sum(jump_size**2,1))
    centre[:,0] = wrapTo2Pi(centre[:,0])*len_vert/(2*np.pi)
    centre[:,1] = wrapTo2Pi(centre[:,1])*len_hori/(2*np.pi)
    
    # centre[:,0][centre[:,0] >= len_vert-0.5] -= len_vert
    # centre[:,1][centre[:,1] >= len_hori-0.5] -= len_hori
    
    if not detect_pattern:
        return centre_ind, centre, jump_size, jump_dist
    else :
        return centre_ind, centre, jump_size, jump_dist, pattern, pattern_size
#%%
def get_MSD(spike, start_time, end_time, \
                   sample_interval = 1, n_neuron = 4096, window = 10, dt = 0.1, \
                   slide_interval = 1, jump_interval = np.array([15]),fit_stableDist = None, mat_eng = None, mat_eng_quit=True):
    if fit_stableDist is not None and fit_stableDist != 'Matlab' and fit_stableDist != 'pylevy':
        raise Exception('Error: fitting method for stable distribution must be either "Matlab" or "pylevy"!')
    if fit_stableDist == 'Matlab': 
        import fitStable_mat
        if mat_eng is None:
            mat_eng = fitStable_mat.start_Mat_Eng()
        #eng = matlab.engine.start_matlab('-nodisplay') # '-nodisplay'
        distparam = np.zeros([len(jump_interval), 3, 4])
    elif  fit_stableDist == 'pylevy':
        import levy
        distparam = np.zeros([len(jump_interval), 4])
    else: pass  
#    print(jump_interval)        
    spk_rate = get_spike_rate(spike, start_time, end_time, indiv_rate = True, popu_rate = False, \
                   sample_interval = sample_interval, n_neuron = n_neuron, window = window, dt = dt, reshape_indiv_rate = True, save_results_to_input = False)
    
    '''
    ###############
    slide_interval = int(np.round(slide_interval/sample_interval))
    len_hori = spk_rate.shape[1]; len_vert = spk_rate.shape[0];
    x_hori = np.cos(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    y_hori = np.sin(np.arange(len_hori)/len_hori*2*np.pi).reshape([-1,1])
    xy_hori = np.concatenate((x_hori,y_hori),axis=1)
    
    x_vert = np.cos(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    y_vert = np.sin(np.arange(len_vert)/len_vert*2*np.pi).reshape([-1,1])
    xy_vert = np.concatenate((x_vert,y_vert),axis=1)
    
    slide_ary = np.arange(0,spk_rate.shape[2],slide_interval)
    #centre_ind = np.zeros([len(slide_ary), 2], dtype=np.int16)
    #jump_size = np.zeros([len(slide_ary), 2])
    centre = np.zeros([len(slide_ary), 2], dtype=float)
    for ind in range(len(slide_ary)):
        if np.all(spk_rate[:,:,slide_ary[ind]] == 0):
            if ind == 0:
                centre[ind] = np.array([0,0])
            else:
                centre[ind] = centre[ind-1]
        else:
            sum_hori = np.sum(spk_rate[:,:,slide_ary[ind]], axis=0)
            sum_vert = np.sum(spk_rate[:,:,slide_ary[ind]], axis=1)
            ctr_hori = np.dot(sum_hori, xy_hori)
            ctr_vert = np.dot(sum_vert, xy_vert)
            centre[ind,1] = np.angle(np.array([ctr_hori[0] + 1j*ctr_hori[1]]))[0]
            centre[ind,0] = np.angle(np.array([ctr_vert[0] + 1j*ctr_vert[1]]))[0]
            #ind_vert = int(wrapTo2Pi(np.array([centre[ind,0]]))*len_vert/(2*np.pi))
    #        if ind_vert > 62: 
    #            ind_vert = 62
            #ind_hori = int(wrapTo2Pi(np.array([centre[ind,1]]))*len_hori/(2*np.pi))
    #        if ind_hori > 62: 
    #            ind_hori = 62
            #centre_ind[ind] = [ind_vert, ind_hori]

    ##########################
    '''

    # slide_interval = int(np.round(slide_interval/dt))
    slide_interval_step = int(np.round(slide_interval/sample_interval))

    len_hori = spk_rate.shape[1]; len_vert = spk_rate.shape[0];
    x_hori = np.cos((np.arange(len_hori)+0.5)/len_hori*2*np.pi).reshape([-1,1])
    y_hori = np.sin((np.arange(len_hori)+0.5)/len_hori*2*np.pi).reshape([-1,1])
    xy_hori = np.concatenate((x_hori,y_hori),axis=1)
    
    x_vert = np.cos((np.arange(len_vert)+0.5)/len_vert*2*np.pi).reshape([-1,1])
    y_vert = np.sin((np.arange(len_vert)+0.5)/len_vert*2*np.pi).reshape([-1,1])
    xy_vert = np.concatenate((x_vert,y_vert),axis=1)
    
    slide_ary = np.arange(0,spk_rate.shape[2],slide_interval_step)
    # centre_ind = np.zeros([len(slide_ary), 2], dtype=np.int16)
    #jump_size = np.zeros([len(slide_ary), 2])
    centre = np.zeros([len(slide_ary), 2], dtype=float)

    for ind in range(len(slide_ary)):
        if np.all(spk_rate[:,:,slide_ary[ind]] == 0):
            if ind == 0:
                centre[ind] = np.array([0,0])
            else:
                centre[ind] = centre[ind-1]
        else:
            sum_hori = np.sum(spk_rate[:,:,slide_ary[ind]], axis=0)
            sum_vert = np.sum(spk_rate[:,:,slide_ary[ind]], axis=1)
            ctr_hori = np.dot(sum_hori, xy_hori)
            ctr_vert = np.dot(sum_vert, xy_vert)

            centre[ind,1] = np.angle(np.array([ctr_hori[0] + 1j*ctr_hori[1]]))[0]
            centre[ind,0] = np.angle(np.array([ctr_vert[0] + 1j*ctr_vert[1]]))[0]

    MSD = np.zeros(jump_interval.shape)
    #print(MSD)        
    print(slide_interval, slide_interval_step)
    for i in range(len(jump_interval)):
        
        jump_size = centre[round(jump_interval[i]/slide_interval):, :] - centre[:len(centre)-round(jump_interval[i]/slide_interval), :]
        jump_size = wrapToPi(jump_size)
        jump_size[:,0] = jump_size[:,0]*len_vert/(2*np.pi)
        jump_size[:,1] = jump_size[:,1]*len_hori/(2*np.pi)
        MSD[i] = np.mean(np.sum(jump_size**2,1))
        #print(MSD)
        if fit_stableDist == 'Matlab':            
            #print(jump_size.size)
            #jumpsize_mat = matlab.double(list(jump_size.reshape(-1)), size=[jump_size.size,1])
            #distparam[i] = my_fitstable.fitstable(inputIn)
            
            # if i == 0: eng = fitStable_mat.start_Mat_Eng()
            fitparam, mat_eng = fitStable_mat.fitStable(jump_size.reshape(-1), mat_eng)
            #jumpsize_mat = jump_size.reshape(-1)
            #plt.figure()
            #plt.hist(jumpsize_mat)
            #jumpsize_mat = matlab.double(list(jumpsize_mat))
            #jumpsize_mat.reshape([2*len(jump_size),1])
            
            #eng.fitdist(jumpsize_mat,'stable',nargout=0)
            #eng.workspace['result_fit'] = eng.fitdist(jumpsize_mat,'stable')#,nargout=0)
            
            #eng.eval("result_fit=fitdist(jumpsize_mat,'stable')", nargout=0)
            #paramci = eng.eval('result_fit.paramci')
            #fitparam = eng.eval('[result_fit.alpha result_fit.beta result_fit.gam result_fit.delta]')
            #fitparam = np.concatenate((np.array(fitparam),np.array(paramci)),0)       
            distparam[i] = fitparam
        
        elif fit_stableDist == 'pylevy':               
            fitparam = levy.fit_levy(jump_size.reshape(-1))
            distparam[i] = fitparam[0].get('0')
        else: pass            
    #print(MSD)        
    if fit_stableDist == 'Matlab' and mat_eng_quit: mat_eng.quit()
    
    if fit_stableDist is not None:
        return MSD, distparam
    else: return MSD
#%%
def wrapTo2Pi(angle):
    #positiveinput = (angle > 0)
    angle = np.mod(angle, 2*np.pi)
    #angle[(angle==0) & positiveinput] = 2*np.pi
    return angle

def wrapToPi(angle):
    select = (angle < -np.pi) | (angle > np.pi) 
    angle[select] = wrapTo2Pi(angle[select] + np.pi) - np.pi
    return angle    
#%%
def overlap_centreandspike(centre, img, show_trajectory = False):
    '''
    overlap centre of mass and network firing rate for visualization
    input: 
    centre: coordinate of centre of mass
    img: (N*N*t) network firing rate or spike counts in a time window
    show_trajectory: if show the historic location of centre of mass
    output:
    img: overlapped result    
    '''
    #centre = centre_mass(spk)
    #centre = centre)
    if not show_trajectory:
        for ind in range(img.shape[2]):
            img[:,:,ind][centre[ind,0], centre[ind,1]] = -1 # make the centre of mass on the plot show different colour        
        return img
    else:
        for ind in range(img.shape[2]):
            img[centre[ind,0],centre[ind,1],ind:] = -1
        return img
#%%
def plot_traj(ax, centre_ind, corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), coordinate_type_is_matrix=True, lw=1):
    '''
    plot trajactory; periodic boundary is considered
    ax: input plot ax
    centre_ind: coordinates of points of trajactory; N*2 array
    corner: the down-left and upper-right corner of grid
    coordinate_type_is_matrix: if the format of coordinates uses the convention of matrix coordinate
    '''
    #wd = width; ht = height
    #corner = np.array([[-0.5,-0.5],[wd-0.5, ht-0.5]])
    
    centre_ind = np.copy(centre_ind) 
    # copy the centre_ind to avoid changing the value of original centre_ind out of the function 
    # if don't do that, the original centre_ind out of the function will be changed, not sure if it is a bug
    wd = corner[1,0]-corner[0,0]; ht = corner[1,1]-corner[0,1]

    if coordinate_type_is_matrix:
        centre_ind[:,0] = ht - 1 - centre_ind[:,0]
        centre_ind = np.flip(centre_ind, 1)
    else: pass

    ax.plot(centre_ind[0,0], centre_ind[0,1], 'og', label='start')
    
    for i in range (len(centre_ind)-1):
        
        if abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2) and abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):           
            
            if np.sign(centre_ind[i,1]-centre_ind[i+1,1]) == np.sign(centre_ind[i,0]-centre_ind[i+1,0]):        
                tr = int(centre_ind[i,0] > centre_ind[i+1,0])
                upright = tr*centre_ind[i] + (1-tr)*centre_ind[i+1] 
                downleft = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0]) > (corner[1,1] - upright[1])/(corner[1,0] - upright[0]):
                    x = [[upright[0], upright[0], upright[0]-wd],  
                         [downleft[0]+wd, downleft[0]+wd, downleft[0]]]
                    y = [[upright[1], upright[1]-ht, upright[1]-ht],
                         [downleft[1]+ht, downleft[1], downleft[1]]]
                else:
                    x = [[upright[0], upright[0]-wd, upright[0]-wd],  
                         [downleft[0]+wd, downleft[0], downleft[0]]]
                    y = [[upright[1], upright[1], upright[1]-ht],
                         [downleft[1]+ht, downleft[1]+ht, downleft[1]]]
                ax.plot(x,y,'b',lw=lw)           
            else:   
                tr = int(centre_ind[i,0] < centre_ind[i+1,0])
                upleft = tr*centre_ind[i] + (1-tr)*centre_ind[i+1]
                downright = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if abs((downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])) > abs((corner[1,1] - upleft[1])/(corner[0,0] - upleft[0])):
                    x = [[upleft[0], upleft[0], upleft[0]+wd],  
                         [downright[0]-wd, downright[0]-wd, downright[0]]]
                    y = [[upleft[1], upleft[1]-ht, upleft[1]-ht],
                         [downright[1]+ht, downright[1], downright[1]]]
                else:
                    x = [[upleft[0], upleft[0]+wd, upleft[0]+wd],  
                         [downright[0]-wd, downright[0], downright[0]]]
                    y = [[upleft[1], upleft[1], upleft[1]-ht],
                         [downright[1]+ht, downright[1]+ht, downright[1]]]                    
                ax.plot(x,y,'b',lw=lw)
                
        elif abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2):
            
            if centre_ind[i,0] > centre_ind[i+1,0]:
                ax.plot([centre_ind[i,0]-wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]],'b', lw=lw)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]+wd], [centre_ind[i,1], centre_ind[i+1,1]],'b', lw=lw)
            else:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]-wd], [centre_ind[i,1], centre_ind[i+1,1]],'b', lw=lw)
                ax.plot([centre_ind[i,0]+wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]],'b', lw=lw)
        
        elif abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):
            if centre_ind[i,1]-centre_ind[i+1,1]>0:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]-ht, centre_ind[i+1,1]],'b', lw=lw)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]+ht],'b', lw=lw)
            else:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]-ht],'b', lw=lw)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]+ht, centre_ind[i+1,1]],'b', lw=lw)
        else:
            ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]],'b', lw=lw)
    ax.plot(centre_ind[i+1,0], centre_ind[i+1,1],'or',label='end')
    ax.legend()
    return ax
#%%
def plot_traj_simp_1(ax, centre_ind, clist, corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), 
                   coordinate_type_is_matrix=True, lw=1, ls='-'):
    '''
    plot trajactory; periodic boundary is considered
    ax: input plot ax
    centre_ind: coordinates of points of trajactory; N*2 array
    corner: the down-left and upper-right corner of grid
    coordinate_type_is_matrix: if the format of coordinates uses the convention of matrix coordinate
    '''
    #wd = width; ht = height
    #corner = np.array([[-0.5,-0.5],[wd-0.5, ht-0.5]])
    
    centre_ind = np.copy(centre_ind) 
    # copy the centre_ind to avoid changing the value of original centre_ind out of the function 
    # if don't do that, the original centre_ind out of the function will be changed, not sure if it is a bug
    wd = corner[1,0]-corner[0,0]; ht = corner[1,1]-corner[0,1]

    if coordinate_type_is_matrix:
        centre_ind[:,0] = ht - centre_ind[:,0]
        centre_ind = np.flip(centre_ind, 1)
    else: pass

    # ax.plot(centre_ind[0,0], centre_ind[0,1], 'og', label='start')
    
    for i in range (len(centre_ind)-1):
        
        if abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2) and abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):           
            
            if np.sign(centre_ind[i,1]-centre_ind[i+1,1]) == np.sign(centre_ind[i,0]-centre_ind[i+1,0]):        
                tr = int(centre_ind[i,0] > centre_ind[i+1,0])
                upright = tr*centre_ind[i] + (1-tr)*centre_ind[i+1] 
                downleft = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0]) > (corner[1,1] - upright[1])/(corner[1,0] - upright[0]):
                    x = [[upright[0], upright[0], upright[0]-wd],  
                         [downleft[0]+wd, downleft[0]+wd, downleft[0]]]
                    y = [[upright[1], upright[1]-ht, upright[1]-ht],
                         [downleft[1]+ht, downleft[1], downleft[1]]]
                else:
                    x = [[upright[0], upright[0]-wd, upright[0]-wd],  
                         [downleft[0]+wd, downleft[0], downleft[0]]]
                    y = [[upright[1], upright[1], upright[1]-ht],
                         [downleft[1]+ht, downleft[1]+ht, downleft[1]]]
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i])           
            else:   
                tr = int(centre_ind[i,0] < centre_ind[i+1,0])
                upleft = tr*centre_ind[i] + (1-tr)*centre_ind[i+1]
                downright = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if abs((downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])) > abs((corner[1,1] - upleft[1])/(corner[0,0] - upleft[0])):
                    x = [[upleft[0], upleft[0], upleft[0]+wd],  
                         [downright[0]-wd, downright[0]-wd, downright[0]]]
                    y = [[upleft[1], upleft[1]-ht, upleft[1]-ht],
                         [downright[1]+ht, downright[1], downright[1]]]
                else:
                    x = [[upleft[0], upleft[0]+wd, upleft[0]+wd],  
                         [downright[0]-wd, downright[0], downright[0]]]
                    y = [[upleft[1], upleft[1], upleft[1]-ht],
                         [downright[1]+ht, downright[1]+ht, downright[1]]]                    
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i])
                
        elif abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2):
            
            if centre_ind[i,0] > centre_ind[i+1,0]:
                ax.plot([centre_ind[i,0]-wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]+wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
            else:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]-wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                ax.plot([centre_ind[i,0]+wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
        
        elif abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):
            if centre_ind[i,1]-centre_ind[i+1,1]>0:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]-ht, centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]+ht], lw=lw, ls=ls, c=clist[i])
            else:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]-ht], lw=lw, ls=ls, c=clist[i])
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]+ht, centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
        else:
            ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
    # ax.plot(centre_ind[i+1,0], centre_ind[i+1,1],'or',label='end')
    # ax.legend()
    return ax
#%%
def plot_traj_simp(ax, centre_ind, clist, corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), 
                   coordinate_type_is_matrix=True, lw=1, ls='-',zorder=0):
    '''
    plot trajactory; periodic boundary is considered
    ax: input plot ax
    centre_ind: coordinates of points of trajactory; N*2 array
    corner: the down-left and upper-right corner of grid
    coordinate_type_is_matrix: if the format of coordinates uses the convention of matrix coordinate
    '''
    #wd = width; ht = height
    #corner = np.array([[-0.5,-0.5],[wd-0.5, ht-0.5]])
    
    centre_ind = np.copy(centre_ind) 
    # copy the centre_ind to avoid changing the value of original centre_ind out of the function 
    # if don't do that, the original centre_ind out of the function will be changed, not sure if it is a bug
    wd = corner[1,0]-corner[0,0]; ht = corner[1,1]-corner[0,1]

    if coordinate_type_is_matrix:
        centre_ind[:,0] = ht - centre_ind[:,0]
        centre_ind = np.flip(centre_ind, 1)
    else: pass

    # ax.plot(centre_ind[0,0], centre_ind[0,1], 'og', label='start')
    
    for i in range (len(centre_ind)-1):
        
        if abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2) and abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):           
            
            if np.sign(centre_ind[i,1]-centre_ind[i+1,1]) == np.sign(centre_ind[i,0]-centre_ind[i+1,0]):        
                tr = int(centre_ind[i,0] > centre_ind[i+1,0])
                upright = tr*centre_ind[i] + (1-tr)*centre_ind[i+1] 
                downleft = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0]) > (corner[1,1] - upright[1])/(corner[1,0] - upright[0]):
                    r = (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0])
                    x1 = upright[0] + (corner[1,1]-upright[1])/r
                    y1 = corner[1,1]
                    x2 = x1; y2 = corner[0,1]
                    x3 = corner[1,0]; y3 = (corner[1,0]-x2)*r + corner[0,1]
                    x4 = corner[0,0]; y4 = y3
                    x5 = downleft[0]; y5 = downleft[1]
                    x = [[upright[0],x2, x4],
                          [x1,        x3, x5]]
                    y = [[upright[1],y2, y4],
                          [y1,        y3, y5]]
                    # print(y)
                    # x = [[upright[0], upright[0], upright[0]-wd],  
                    #      [downleft[0]+wd, downleft[0]+wd, downleft[0]]]
                    # y = [[upright[1], upright[1]-ht, upright[1]-ht],
                    #      [downleft[1]+ht, downleft[1], downleft[1]]]
                else:
                    r = (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0])
                    x1 = corner[1,0] ; y1 =  upright[1]+(x1-upright[0])*r  
                    x2 = corner[0,0]; y2 = y1
                    # x3 = upright[0]-wd + (corner[1,1]-upright[1])/r; 
                    x3 = (corner[1,1]-y2)/r + corner[0,0]; y3 = corner[1,1]
                    x4 = x3; y4 = corner[0,1]
                    x5 = downleft[0]; y5 = downleft[1]
                    x = [[upright[0],x2, x4],
                          [x1,        x3, x5]]
                    y = [[upright[1],y2, y4],
                          [y1,        y3, y5]]
                    # x = [[upright[0], upright[0]-wd, upright[0]-wd],  
                    #      [downleft[0]+wd, downleft[0], downleft[0]]]
                    # y = [[upright[1], upright[1], upright[1]-ht],
                    #      [downleft[1]+ht, downleft[1]+ht, downleft[1]]]
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i],zorder=zorder)           
            else:   
                tr = int(centre_ind[i,0] < centre_ind[i+1,0])
                upleft = tr*centre_ind[i] + (1-tr)*centre_ind[i+1]
                downright = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if abs((downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])) > abs((corner[1,1] - upleft[1])/(corner[0,0] - upleft[0])):
                    r = (downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])
                    x0 = upleft[0]; y0 = upleft[1]
                    x1 = x0 + (corner[1,1]-y0)/r; y1 = corner[1,1]
                    x2 = x1; y2 = corner[0,1]
                    x3 = corner[0,0]; y3 = (corner[0,0]-x2)*r + corner[0,1]
                    x4 = corner[1,0]; y4 = y3
                    x5 = downright[0]; y5 = downright[1]
                    x = [[x0, x2, x4],
                          [x1, x3, x5]]
                    y = [[y0, y2, y4],
                          [y1, y3, y5]]                   
                    # x = [[upleft[0], upleft[0], upleft[0]+wd],  
                    #      [downright[0]-wd, downright[0]-wd, downright[0]]]
                    # y = [[upleft[1], upleft[1]-ht, upleft[1]-ht],
                    #      [downright[1]+ht, downright[1], downright[1]]]
                else:
                    r = (downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])
                    x0 = upleft[0]; y0 = upleft[1]
                    x1 = corner[0,0] ; y1 =  y0+(x1-x0)*r  
                    x2 = corner[1,0]; y2 = y1
                    # x3 = upright[0]-wd + (corner[1,1]-upright[1])/r; 
                    x3 = (corner[1,1]-y2)/r + x2; y3 = corner[1,1]
                    x4 = x3; y4 = corner[0,1]
                    x5 = downright[0]; y5 = downright[1]
                    x = [[x0, x2, x4],
                          [x1, x3, x5]]
                    y = [[y0, y2, y4],
                          [y1, y3, y5]]                    
                    # x = [[upleft[0], upleft[0]+wd, upleft[0]+wd],  
                    #      [downright[0]-wd, downright[0], downright[0]]]
                    # y = [[upleft[1], upleft[1], upleft[1]-ht],
                    #      [downright[1]+ht, downright[1]+ht, downright[1]]]                    
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i],zorder=zorder)
                
        elif abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2):            
            if centre_ind[i,0] > centre_ind[i+1,0]:
                r = (centre_ind[i+1,1]-centre_ind[i,1])/(centre_ind[i+1,0] + wd - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                x1 = corner[1,0]; y1 = (x1-x0)*r + y0
                x2 = corner[0,0]; y2 = y1;
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i],zorder=zorder)                    
                # ax.plot([centre_ind[i,0]-wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]+wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
            else:
                r = (centre_ind[i+1,1]-centre_ind[i,1])/(centre_ind[i+1,0] - wd - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                x1 = corner[0,0]; y1 = (x1-x0)*r + y0
                x2 = corner[1,0]; y2 = y1;
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i],zorder=zorder)                    
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]-wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0]+wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
        
        elif abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):
            if centre_ind[i,1]-centre_ind[i+1,1]>0:
                r = (centre_ind[i+1,1]+ht-centre_ind[i,1])/(centre_ind[i+1,0] - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                y1 = corner[1,1]; x1 = (y1-y0)/r + x0 
                x2 = x1; y2 = corner[0,1];
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i],zorder=zorder)                    
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]-ht, centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]+ht], lw=lw, ls=ls, c=clist[i])
            else:
                r = (centre_ind[i+1,1]-ht-centre_ind[i,1])/(centre_ind[i+1,0] - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                y1 = corner[0,1]; x1 = (y1-y0)/r + x0 
                x2 = x1; y2 = corner[1,1];
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                ax.plot(x,y,lw=lw, ls=ls, c=clist[i],zorder=zorder)                    
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]-ht], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]+ht, centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
        else:
            ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i],zorder=zorder)
    # ax.plot(centre_ind[i+1,0], centre_ind[i+1,1],'or',label='end')
    # ax.legend()
    return ax

#%%
def plot_traj_simp_2(ax, centre_ind, clist, corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), 
                   coordinate_type_is_matrix=True, lw=1, ls='-'):
    '''
    plot trajactory; periodic boundary is considered
    ax: input plot ax
    centre_ind: coordinates of points of trajactory; N*2 array
    corner: the down-left and upper-right corner of grid
    coordinate_type_is_matrix: if the format of coordinates uses the convention of matrix coordinate
    '''
    #wd = width; ht = height
    #corner = np.array([[-0.5,-0.5],[wd-0.5, ht-0.5]])
    
    centre_ind = np.copy(centre_ind) 
    # copy the centre_ind to avoid changing the value of original centre_ind out of the function 
    # if don't do that, the original centre_ind out of the function will be changed, not sure if it is a bug
    wd = corner[1,0]-corner[0,0]; ht = corner[1,1]-corner[0,1]

    if coordinate_type_is_matrix:
        centre_ind[:,0] = ht - centre_ind[:,0]
        centre_ind = np.flip(centre_ind, 1)
    else: pass

    # ax.plot(centre_ind[0,0], centre_ind[0,1], 'og', label='start')
    linex = [[]]; liney = [[]]
    seg_i = 0
    linex[seg_i].append(centre_ind[0,0]) 
    liney[seg_i].append(centre_ind[0,1]) 
    
    for i in range (len(centre_ind)-1):
        
        if abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2) and abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):           
            
            if np.sign(centre_ind[i,1]-centre_ind[i+1,1]) == np.sign(centre_ind[i,0]-centre_ind[i+1,0]):        
                tr = int(centre_ind[i,0] > centre_ind[i+1,0])
                upright = tr*centre_ind[i] + (1-tr)*centre_ind[i+1] 
                downleft = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0]) > (corner[1,1] - upright[1])/(corner[1,0] - upright[0]):
                    r = (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0])
                    x1 = upright[0] + (corner[1,1]-upright[1])/r
                    y1 = corner[1,1]
                    x2 = x1; y2 = corner[0,1]
                    x3 = corner[1,0]; y3 = (corner[1,0]-x2)*r + corner[0,1]
                    x4 = corner[0,0]; y4 = y3
                    x5 = downleft[0]; y5 = downleft[1]
                    x = [[upright[0],x2, x4],
                          [x1,        x3, x5]]
                    y = [[upright[1],y2, y4],
                          [y1,        y3, y5]]
                    # print(y)
                    # x = [[upright[0], upright[0], upright[0]-wd],  
                    #      [downleft[0]+wd, downleft[0]+wd, downleft[0]]]
                    # y = [[upright[1], upright[1]-ht, upright[1]-ht],
                    #      [downleft[1]+ht, downleft[1], downleft[1]]]
                else:
                    r = (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0])
                    x1 = corner[1,0] ; y1 =  upright[1]+(x1-upright[0])*r  
                    x2 = corner[0,0]; y2 = y1
                    # x3 = upright[0]-wd + (corner[1,1]-upright[1])/r; 
                    x3 = (corner[1,1]-y2)/r + corner[0,0]; y3 = corner[1,1]
                    x4 = x3; y4 = corner[0,1]
                    x5 = downleft[0]; y5 = downleft[1]
                    x = [[upright[0],x2, x4],
                          [x1,        x3, x5]]
                    y = [[upright[1],y2, y4],
                          [y1,        y3, y5]]
                    # x = [[upright[0], upright[0]-wd, upright[0]-wd],  
                    #      [downleft[0]+wd, downleft[0], downleft[0]]]
                    # y = [[upright[1], upright[1], upright[1]-ht],
                    #      [downleft[1]+ht, downleft[1]+ht, downleft[1]]]
                linex[seg_i].append(x1) 
                liney[seg_i].append(y1)
                linex = linex + [[x2,x3],[x4,x5]]
                liney = liney + [[y2,y3],[y4,y5]]
                seg_i += 2
                # ax.plot(x,y,lw=lw, ls=ls, c=clist[i])           
            else:   
                tr = int(centre_ind[i,0] < centre_ind[i+1,0])
                upleft = tr*centre_ind[i] + (1-tr)*centre_ind[i+1]
                downright = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if abs((downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])) > abs((corner[1,1] - upleft[1])/(corner[0,0] - upleft[0])):
                    r = (downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])
                    x0 = upleft[0]; y0 = upleft[1]
                    x1 = x0 + (corner[1,1]-y0)/r; y1 = corner[1,1]
                    x2 = x1; y2 = corner[0,1]
                    x3 = corner[0,0]; y3 = (corner[0,0]-x2)*r + corner[0,1]
                    x4 = corner[1,0]; y4 = y3
                    x5 = downright[0]; y5 = downright[1]
                    x = [[x0, x2, x4],
                          [x1, x3, x5]]
                    y = [[y0, y2, y4],
                          [y1, y3, y5]]                   
                    # x = [[upleft[0], upleft[0], upleft[0]+wd],  
                    #      [downright[0]-wd, downright[0]-wd, downright[0]]]
                    # y = [[upleft[1], upleft[1]-ht, upleft[1]-ht],
                    #      [downright[1]+ht, downright[1], downright[1]]]
                else:
                    r = (downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])
                    x0 = upleft[0]; y0 = upleft[1]
                    x1 = corner[0,0] ; y1 =  y0+(x1-x0)*r  
                    x2 = corner[1,0]; y2 = y1
                    # x3 = upright[0]-wd + (corner[1,1]-upright[1])/r; 
                    x3 = (corner[1,1]-y2)/r + x2; y3 = corner[1,1]
                    x4 = x3; y4 = corner[0,1]
                    x5 = downright[0]; y5 = downright[1]
                    x = [[x0, x2, x4],
                          [x1, x3, x5]]
                    y = [[y0, y2, y4],
                          [y1, y3, y5]]                    
                    # x = [[upleft[0], upleft[0]+wd, upleft[0]+wd],  
                    #      [downright[0]-wd, downright[0], downright[0]]]
                    # y = [[upleft[1], upleft[1], upleft[1]-ht],
                    #      [downright[1]+ht, downright[1]+ht, downright[1]]]                    
                linex[seg_i].append(x1) 
                liney[seg_i].append(y1)
                linex = linex + [[x2,x3],[x4,x5]]
                liney = liney + [[y2,y3],[y4,y5]]
                seg_i += 2                
                # ax.plot(x,y,lw=lw, ls=ls, c=clist[i])
                
        elif abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2):            
            if centre_ind[i,0] > centre_ind[i+1,0]:
                r = (centre_ind[i+1,1]-centre_ind[i,1])/(centre_ind[i+1,0] + wd - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                x1 = corner[1,0]; y1 = (x1-x0)*r + y0
                x2 = corner[0,0]; y2 = y1;
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                # ax.plot(x,y,lw=lw, ls=ls, c=clist[i])                    
                # ax.plot([centre_ind[i,0]-wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]+wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
            else:
                r = (centre_ind[i+1,1]-centre_ind[i,1])/(centre_ind[i+1,0] - wd - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                x1 = corner[0,0]; y1 = (x1-x0)*r + y0
                x2 = corner[1,0]; y2 = y1;
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                # ax.plot(x,y,lw=lw, ls=ls, c=clist[i])                    
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]-wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0]+wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
            linex[seg_i].append(x1) 
            liney[seg_i].append(y1)
            linex = linex + [[x2,x3]]
            liney = liney + [[y2,y3]]
            seg_i += 1
                
        elif abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):
            if centre_ind[i,1]-centre_ind[i+1,1]>0:
                r = (centre_ind[i+1,1]+ht-centre_ind[i,1])/(centre_ind[i+1,0] - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                y1 = corner[1,1]; x1 = (y1-y0)/r + x0 
                x2 = x1; y2 = corner[0,1];
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                # ax.plot(x,y,lw=lw, ls=ls, c=clist[i])                    
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]-ht, centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]+ht], lw=lw, ls=ls, c=clist[i])
            else:
                r = (centre_ind[i+1,1]-ht-centre_ind[i,1])/(centre_ind[i+1,0] - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                y1 = corner[0,1]; x1 = (y1-y0)/r + x0 
                x2 = x1; y2 = corner[1,1];
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = [[x0, x2],
                      [x1, x3]]
                y = [[y0, y2],
                      [y1, y3]]
                # ax.plot(x,y,lw=lw, ls=ls, c=clist[i])                    
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]-ht], lw=lw, ls=ls, c=clist[i])
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]+ht, centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
            linex[seg_i].append(x1) 
            liney[seg_i].append(y1)
            linex = linex + [[x2,x3]]
            liney = liney + [[y2,y3]]
            seg_i += 1
        
        else:
            linex[seg_i].append(centre_ind[i+1,0]) 
            liney[seg_i].append(centre_ind[i+1,1]) 
            # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, c=clist[i])
    # ax.plot(centre_ind[i+1,0], centre_ind[i+1,1],'or',label='end')
    # ax.legend()
    for x,y in zip(linex,liney): 
        ax.plot(x, y, lw=lw, ls=ls, c=clist[-1])

    return ax


#%%
def plot_traj_simp_3d(ax, centre_ind, clist, corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), 
                   coordinate_type_is_matrix=True, lw=1, ls='-', zs=0, zdir='z', zorder=10):
    '''
    plot trajactory; periodic boundary is considered
    ax: input plot ax
    centre_ind: coordinates of points of trajactory; N*2 array
    corner: the down-left and upper-right corner of grid
    coordinate_type_is_matrix: if the format of coordinates uses the convention of matrix coordinate
    '''
    #wd = width; ht = height
    #corner = np.array([[-0.5,-0.5],[wd-0.5, ht-0.5]])
    
    centre_ind = np.copy(centre_ind) 
    # copy the centre_ind to avoid changing the value of original centre_ind out of the function 
    # if don't do that, the original centre_ind out of the function will be changed, not sure if it is a bug
    wd = corner[1,0]-corner[0,0]; ht = corner[1,1]-corner[0,1]

    if coordinate_type_is_matrix:
        centre_ind[:,0] = ht - centre_ind[:,0]
        centre_ind = np.flip(centre_ind, 1)
    else: pass

    # ax.plot(centre_ind[0,0], centre_ind[0,1], 'og', label='start')
    
    for i in range (len(centre_ind)-1):
        
        if abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2) and abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):           
            
            if np.sign(centre_ind[i,1]-centre_ind[i+1,1]) == np.sign(centre_ind[i,0]-centre_ind[i+1,0]):        
                tr = int(centre_ind[i,0] > centre_ind[i+1,0])
                upright = tr*centre_ind[i] + (1-tr)*centre_ind[i+1] 
                downleft = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0]) > (corner[1,1] - upright[1])/(corner[1,0] - upright[0]):
                    r = (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0])
                    x1 = upright[0] + (corner[1,1]-upright[1])/r
                    y1 = corner[1,1]
                    x2 = x1; y2 = corner[0,1]
                    x3 = corner[1,0]; y3 = (corner[1,0]-x2)*r + corner[0,1]
                    x4 = corner[0,0]; y4 = y3
                    x5 = downleft[0]; y5 = downleft[1]
                    x = np.array([[upright[0],x2, x4],
                          [x1,        x3, x5]])
                    y = np.array([[upright[1],y2, y4],
                          [y1,        y3, y5]])
                    # x = [[upright[0], upright[0], upright[0]-wd],  
                    #      [downleft[0]+wd, downleft[0]+wd, downleft[0]]]
                    # y = [[upright[1], upright[1]-ht, upright[1]-ht],
                    #      [downleft[1]+ht, downleft[1], downleft[1]]]
                else:
                    r = (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0])
                    x1 = corner[1,0] ; y1 =  upright[1]+(x1-upright[0])*r  
                    x2 = corner[0,0]; y2 = y1
                    # x3 = upright[0]-wd + (corner[1,1]-upright[1])/r; 
                    x3 = (corner[1,1]-y2)/r + corner[0,0]; y3 = corner[1,1]
                    x4 = x3; y4 = corner[0,1]
                    x5 = downleft[0]; y5 = downleft[1]
                    x = np.array([[upright[0],x2, x4],
                          [x1,        x3, x5]])
                    y = np.array([[upright[1],y2, y4],
                          [y1,        y3, y5]])
                    # x = [[upright[0], upright[0]-wd, upright[0]-wd],  
                    #      [downleft[0]+wd, downleft[0], downleft[0]]]
                    # y = [[upright[1], upright[1], upright[1]-ht],
                    #      [downleft[1]+ht, downleft[1]+ht, downleft[1]]]
                for ii in range(x.shape[1]):
                    ax.plot(x[:,ii],y[:,ii],lw=lw, ls=ls, c=clist[i], zs=zs, zdir=zdir, zorder=zorder)
                # print(1)
            else:   
                tr = int(centre_ind[i,0] < centre_ind[i+1,0])
                upleft = tr*centre_ind[i] + (1-tr)*centre_ind[i+1]
                downright = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if abs((downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])) > abs((corner[1,1] - upleft[1])/(corner[0,0] - upleft[0])):
                    r = (downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])
                    x0 = upleft[0]; y0 = upleft[1]
                    x1 = x0 + (corner[1,1]-y0)/r; y1 = corner[1,1]
                    x2 = x1; y2 = corner[0,1]
                    x3 = corner[0,0]; y3 = (corner[0,0]-x2)*r + corner[0,1]
                    x4 = corner[1,0]; y4 = y3
                    x5 = downright[0]; y5 = downright[1]
                    x = np.array([[x0, x2, x4],
                          [x1, x3, x5]])
                    y = np.array([[y0, y2, y4],
                          [y1, y3, y5]])
                    # x = [[upleft[0], upleft[0], upleft[0]+wd],  
                    #      [downright[0]-wd, downright[0]-wd, downright[0]]]
                    # y = [[upleft[1], upleft[1]-ht, upleft[1]-ht],
                    #      [downright[1]+ht, downright[1], downright[1]]]
                else:
                    r = (downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])
                    x0 = upleft[0]; y0 = upleft[1]
                    x1 = corner[0,0] ; y1 =  y0+(x1-x0)*r  
                    x2 = corner[1,0]; y2 = y1
                    # x3 = upright[0]-wd + (corner[1,1]-upright[1])/r; 
                    x3 = (corner[1,1]-y2)/r + x2; y3 = corner[1,1]
                    x4 = x3; y4 = corner[0,1]
                    x5 = downright[0]; y5 = downright[1]
                    x = np.array([[x0, x2, x4],
                          [x1, x3, x5]])
                    y = np.array([[y0, y2, y4],
                          [y1, y3, y5]])
                    # x = [[upleft[0], upleft[0]+wd, upleft[0]+wd],  
                    #      [downright[0]-wd, downright[0], downright[0]]]
                    # y = [[upleft[1], upleft[1], upleft[1]-ht],
                    #      [downright[1]+ht, downright[1]+ht, downright[1]]]                    
                for ii in range(x.shape[1]):
                    ax.plot(x[:,ii],y[:,ii],lw=lw, ls=ls, c=clist[i], zs=zs, zdir=zdir, zorder=zorder)
                # print(1)
                
        elif abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2):
            
            if centre_ind[i,0] > centre_ind[i+1,0]:
                r = (centre_ind[i+1,1]-centre_ind[i,1])/(centre_ind[i+1,0] + wd - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                x1 = corner[1,0]; y1 = (x1-x0)*r + y0
                x2 = corner[0,0]; y2 = y1;
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = np.array([[x0, x2],
                      [x1, x3]])
                y = np.array([[y0, y2],
                      [y1, y3]])
                for ii in range(x.shape[1]):
                    ax.plot(x[:,ii],y[:,ii],lw=lw, ls=ls, c=clist[i], zs=zs, zdir=zdir, zorder=zorder)
                # print(1)
                # ax.plot([centre_ind[i,0]-wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]+wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
            else:
                r = (centre_ind[i+1,1]-centre_ind[i,1])/(centre_ind[i+1,0] - wd - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                x1 = corner[0,0]; y1 = (x1-x0)*r + y0
                x2 = corner[1,0]; y2 = y1;
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = np.array([[x0, x2],
                      [x1, x3]])
                y = np.array([[y0, y2],
                      [y1, y3]])
                for ii in range(x.shape[1]):
                    ax.plot(x[:,ii],y[:,ii],lw=lw, ls=ls, c=clist[i], zs=zs, zdir=zdir, zorder=zorder)
                # print(1)
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]-wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls,zs=zs, zdir=zdir, c=clist[i], zorder=zorder)
                # ax.plot([centre_ind[i,0]+wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                
        elif abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):
            if centre_ind[i,1]-centre_ind[i+1,1]>0:
                r = (centre_ind[i+1,1]+ht-centre_ind[i,1])/(centre_ind[i+1,0] - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                y1 = corner[1,1]; x1 = (y1-y0)/r + x0 
                x2 = x1; y2 = corner[0,1];
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = np.array([[x0, x2],
                      [x1, x3]])
                y = np.array([[y0, y2],
                      [y1, y3]])
                for ii in range(x.shape[1]):
                    ax.plot(x[:,ii],y[:,ii],lw=lw, ls=ls, c=clist[i], zs=zs, zdir=zdir, zorder=zorder)
                # print(1)
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]-ht, centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]+ht], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
            else:
                r = (centre_ind[i+1,1]-ht-centre_ind[i,1])/(centre_ind[i+1,0] - centre_ind[i,0])
                x0 = centre_ind[i,0]; y0 = centre_ind[i,1]
                y1 = corner[0,1]; x1 = (y1-y0)/r + x0 
                x2 = x1; y2 = corner[1,1];
                x3 = centre_ind[i+1,0]; y3 = centre_ind[i+1,1]
                x = np.array([[x0, x2],
                      [x1, x3]])
                y = np.array([[y0, y2],
                      [y1, y3]])
                # try:
                for ii in range(x.shape[1]):
                    ax.plot(x[:,ii],y[:,ii],lw=lw, ls=ls, c=clist[i], zs=zs, zdir=zdir, zorder=zorder)
                # except ValueError:
                #     print(x, '\n',y, '\n', i)
                #     return 0
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]-ht], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                # ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]+ht, centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
        else:
            ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
    # ax.plot(centre_ind[i+1,0], centre_ind[i+1,1],'or',label='end')
    # ax.legend()
    return ax
#%%
def plot_traj_simp_3d_(ax, centre_ind, clist, corner = np.array([[-0.5,-0.5],[63.5, 63.5]]), 
                   coordinate_type_is_matrix=True, lw=1, ls='-', zs=0, zdir='z', zorder=10):
    '''
    plot trajactory; periodic boundary is considered
    ax: input plot ax
    centre_ind: coordinates of points of trajactory; N*2 array
    corner: the down-left and upper-right corner of grid
    coordinate_type_is_matrix: if the format of coordinates uses the convention of matrix coordinate
    '''
    #wd = width; ht = height
    #corner = np.array([[-0.5,-0.5],[wd-0.5, ht-0.5]])
    
    centre_ind = np.copy(centre_ind) 
    # copy the centre_ind to avoid changing the value of original centre_ind out of the function 
    # if don't do that, the original centre_ind out of the function will be changed, not sure if it is a bug
    wd = corner[1,0]-corner[0,0]; ht = corner[1,1]-corner[0,1]

    if coordinate_type_is_matrix:
        centre_ind[:,0] = ht - centre_ind[:,0]
        centre_ind = np.flip(centre_ind, 1)
        print(centre_ind.min(), centre_ind.max())
    else: pass

    # ax.plot(centre_ind[0,0], centre_ind[0,1], 'og', label='start')
    
    for i in range (len(centre_ind)-1):
        # print(i)
        if abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2) and abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):           
            
            if np.sign(centre_ind[i,1]-centre_ind[i+1,1]) == np.sign(centre_ind[i,0]-centre_ind[i+1,0]):        
                tr = int(centre_ind[i,0] > centre_ind[i+1,0])
                upright = tr*centre_ind[i] + (1-tr)*centre_ind[i+1] 
                downleft = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if (downleft[1] + ht - upright[1])/(downleft[0] + wd - upright[0]) > (corner[1,1] - upright[1])/(corner[1,0] - upright[0]):
                    x = [[upright[0], upright[0], upright[0]-wd],  
                          [downleft[0]+wd, downleft[0]+wd, downleft[0]]]
                    y = [[upright[1], upright[1]-ht, upright[1]-ht],
                          [downleft[1]+ht, downleft[1], downleft[1]]]
                else:
                    x = [[upright[0], upright[0]-wd, upright[0]-wd],  
                          [downleft[0]+wd, downleft[0], downleft[0]]]
                    y = [[upright[1], upright[1], upright[1]-ht],
                          [downleft[1]+ht, downleft[1]+ht, downleft[1]]]
                ax.plot(x,y,lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)           
            else:   
                tr = int(centre_ind[i,0] < centre_ind[i+1,0])
                upleft = tr*centre_ind[i] + (1-tr)*centre_ind[i+1]
                downright = (1-tr)*centre_ind[i] + tr*centre_ind[i+1]
                if abs((downright[1] + ht -upleft[1])/(downright[0] - wd - upleft[0])) > abs((corner[1,1] - upleft[1])/(corner[0,0] - upleft[0])):
                    x = [[upleft[0], upleft[0], upleft[0]+wd],  
                          [downright[0]-wd, downright[0]-wd, downright[0]]]
                    y = [[upleft[1], upleft[1]-ht, upleft[1]-ht],
                          [downright[1]+ht, downright[1], downright[1]]]
                else:
                    x = [[upleft[0], upleft[0]+wd, upleft[0]+wd],  
                          [downright[0]-wd, downright[0], downright[0]]]
                    y = [[upleft[1], upleft[1], upleft[1]-ht],
                          [downright[1]+ht, downright[1]+ht, downright[1]]]                    
                ax.plot(x,y,lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                
        elif abs(centre_ind[i,0]-centre_ind[i+1,0])>(wd/2):
            
            if centre_ind[i,0] > centre_ind[i+1,0]:
                ax.plot([centre_ind[i,0]-wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]+wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
            else:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]-wd], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls,zs=zs, zdir=zdir, c=clist[i], zorder=zorder)
                ax.plot([centre_ind[i,0]+wd, centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
        
        elif abs(centre_ind[i,1]-centre_ind[i+1,1])>(ht/2):
            if centre_ind[i,1]-centre_ind[i+1,1]>0:
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]-ht, centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]+ht], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
            else:
                print(i)
                print([[centre_ind[i,0], centre_ind[i,0]], 
                       [centre_ind[i+1,0],centre_ind[i+1,0]]])
                print([[centre_ind[i,1], centre_ind[i,1]+ht], 
                       [centre_ind[i+1,1]-ht, centre_ind[i+1,1]]])
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]-ht], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
                ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1]+ht, centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
        else:
            # print(i)
            ax.plot([centre_ind[i,0], centre_ind[i+1,0]], [centre_ind[i,1], centre_ind[i+1,1]], lw=lw, ls=ls, zs=zs, zdir=zdir,c=clist[i], zorder=zorder)
    # ax.plot(centre_ind[i+1,0], centre_ind[i+1,1],'or',label='end')
    # ax.legend()
    return ax
#%%
# def show_pattern2(spkrate1, spkrate2=None, frames = 1000, start_time = 0, interval_movie=10, anititle='', show_pattern_size=False, pattern_ctr=None, pattern_size=None):
#     '''
#     create amination for the firing pattern of network
#     input:
#     spkrate1, spkrate2: (N*N*t array)
#     frames: number of frames of amination
#     start_time: the real-time of the simulation that the first frame of 'spkrate' cooresponds to
#     anititle: set the title of amination
#     '''
#     if spkrate2 is None:
        
#         from mpl_toolkits.axes_grid1 import make_axes_locatable
#         fig = plt.figure(figsize = (6,6))
#         ax1= fig.add_subplot(111, label="1",frame_on=False)
#         ax1.axis('off')
#         divider = make_axes_locatable(ax1)
#         cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
#         #fig.colorbar(img1, cax=cax1)
#         # if show_pattern_size:
#         #     ax2=fig.add_subplot(111, label="2",frame_on=False)
#         #     ax2.set_xlim([-0.5,spkrate1.shape[1]+0.5])
#         #     ax2.set_ylim([-0.5,spkrate1.shape[0]+0.5])       
#         #     ax2.axis('off')
            
#         cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
#         cmap_c = np.array([1.,0.,0.,1.])
#         cmap_stimulus = np.array([88/255,150/255.,0.,1.])
#         cmap = np.vstack((cmap_stimulus,cmap_c,cmap_spk(range(7))))
#         cmap = mpl.colors.ListedColormap(cmap)
#         #cmap.set_under('red')
#         bounds = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
#         norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#         #cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
        
#         cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
#                                         norm=norm,
#                                         boundaries=bounds,
#                                         ticks=np.array([-2,-1, 0, 1, 2, 3, 4, 5, 6])+0.5,
#                                         spacing='proportional',
#                                         orientation='horizontal') #horizontal vertical
#         cb.ax.set_xticklabels(['stimulus','ctr', 0, 1, 2, 3, 4, 5, 6])
#         cb.set_label('number of spikes')
        
#         #titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
#         titleaxes = divider.append_axes("top", size="5%", pad=0.01)
#         titleaxes.axis('off')
#         title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
#         time_title = np.arange(spkrate1.shape[2]) + start_time
        
#         value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
#         #value2=ax2.matshow(spk2[:,:,:], cmap=cb.cmap)
#         # if show_pattern_size: 
#         #     ax2 = fig.add_axes([0, 0, 1, 1]) 
#         #     #ax2= plt.subplot(111)
#         #     circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
#         #     ax2.add_patch(circle)
#         if show_pattern_size:
#             circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.,color='r',fill=False)
#             ax1.add_patch(circle)
#         # def init():
#         #     value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
#         #     title.set_text(u"time: {} ms".format(time_title[0]))
#         #     # if show_pattern_size:
#         #     #     circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
#         #     #     ax2.add_patch(circle)
#         #     #     return value1,title,circle
#         #     return value1,title,
        
#         def updatev(i):
#             value1.set_array(spkrate1[:,:,i])
#             #value2.set_array(spk2[:,:,i])
#             title.set_text(u"time: {} ms".format(time_title[i]))
#             # if show_pattern_size:
#             #     circle.set_center([pattern_ctr[i,1],pattern_ctr[i,0]])
#             #     circle.set_radius(pattern_size[i])
#             #     print('size')
#             #     return value1, title,circle#, value2
#             if show_pattern_size:
#                 circle.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
#                 circle.radius = pattern_size[i]
#             #print('size')
#                 return value1, title,circle,#, value2
#             #return title,circle,
#             else:
#                 return value1, title,
        
#         #value1.set_clim(vmin=0, vmax=6)
#         # ax1.axis('off')
#         #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
#         fig.suptitle(anititle)
#         ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
#         return ani
#     else:
#         fig, [ax1,ax2]= plt.subplots(1,2)
#         #fig.suptitle('sensory to association strength: %.2f\nassociation to sensory strength: %.2f'
#         #             %(scale_e_12[i],scale_e_21[j]))
#         #ax1.set_title('sensory')
#         #ax2.set_title('association')
#         cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
#         cmap_c = np.array([1.,0.,0.,1.])
#         cmap_stimulus = np.array([88/255,150/255.,0.,1.])
#         cmap = np.vstack((cmap_stimulus,cmap_c,cmap_spk(range(7))))
#         cmap = mpl.colors.ListedColormap(cmap)
#         #cmap.set_under('grey')
#         bounds = [-2,-1, 0, 1, 2, 3, 4, 5, 6, 7]
        
#         norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#         cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 
        
#         cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
#                                         norm=norm,
#                                         boundaries=bounds,
#                                         ticks=np.array([-2,-1, 0, 1, 2, 3, 4, 5, 6])+0.5,
#                                         spacing='proportional',
#                                         orientation='horizontal') #horizontal vertical
#         cb.ax.set_xticklabels(['stimulus','ctr', 0, 1, 2, 3, 4, 5, 6])
#         cb.set_label('number of spikes')
        
#         titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
#         titleaxes.axis('off')
#         title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
#         time_title = np.arange(spkrate1.shape[2]) + start_time
        
#         value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
#         value2=ax2.matshow(spkrate2[:,:,0], cmap=cb.cmap, norm=cb.norm)
#         def updatev(i):
#             value1.set_array(spkrate1[:,:,i])
#             value2.set_array(spkrate2[:,:,i])
#             title.set_text(u"time: {} ms".format(time_title[i]))
#             return value1, value2, title,
        
#         #cbaxes1 = fig.add_axes([0.1, 0.1, 0.35, 0.03]) 
#         #cbaxes2 = fig.add_axes([0.55, 0.1, 0.35, 0.03]) 
#         #cb1 = fig.colorbar(value1, cax = cbaxes1, orientation='horizontal', ticks=[0,1,2,3,4]) 
#         #cb2 = fig.colorbar(value2, cax = cbaxes2, orientation='horizontal', ticks=[0,1,2,3,4]) 
#         #value1.set_clim(vmin=0, vmax=6)
#         #value2.set_clim(vmin=0, vmax=6)
#         ax1.axis('off')
#         ax2.axis('off')
#         #ax1.set_title('sensory'); ax2.set_title('association')
# #        if stimu_onset >= 0: 
# #            default_title = 'bottom-up: %.1f*default top-down: %.1f*default\nonset of stimulus:%dms'%(bottom_up, top_down, stimu_onset)
# #            fig.suptitle(default_title + '\n' + anititle)
# #        else : 
# #            default_title = 'bottom-up: %.1f*default top-down: %.1f*default'%(bottom_up, top_down)
# #            fig.suptitle(default_title + '\n' + anititle)
#         fig.suptitle(anititle)
#         ani=animation.FuncAnimation(fig, updatev, frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
#         return ani

#%%
def show_pattern(spkrate1, spkrate2=None, frames = 1000, start_time = 0, interval_movie=10, ax1title='', ax2title='', anititle='', show_pattern_size=False, pattern_ctr=None, pattern_size=None,\
                 stim=None, adpt=None):
    '''
    create amination for the firing pattern of network
    input:
    spkrate1, spkrate2: (N*N*t array)
    frames: number of frames of amination
    start_time: the real-time in the simulation that the first frame of 'spkrate' cooresponds to
    anititle: set the title of amination
    
    stim: location(coordinate is expressed in matrix convention, not in cartesian), onset and offset time, and size of stimuli; for indicating the position of stimulus
    e.g. stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
          [area1:[[[loca1],  [loca2]],  [on_off_1(2D array), on_off_2], [[size for each on_off_1], [size for each on_off_2]]], area2:None]
    multiple stimuli are added at [31.5,31.5] and [63.5,-0.5]; on and off time of each stimulus appearred at [31.5,31.5] and [63.5,-0.5] are
    specified in stim_on_off, stim_on_off; size(radius) of each stimulus: [6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]
       
    adpt: location, onset and offset time, and size of adaptation modulation; for indicating the position of adaptation modulation
    e.g. adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[chg_adapt_range]]]]

    '''
    if spkrate2 is None:
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize = (6,6))
        ax1= fig.add_subplot(111, label="1",frame_on=False)
        ax1.axis('off')
        divider = make_axes_locatable(ax1)
        cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
            
        cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        camp_adapt = np.array([138/255,43/255,226/255,1.])
        cmap_c = np.array([1.,0.,0.,1.])
        cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        cmap = np.vstack((camp_adapt,cmap_stimulus,cmap_c,cmap_spk(range(7))))
        cmap = mpl.colors.ListedColormap(cmap)
        #cmap.set_under('red')
        bounds = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) - 0.5
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
        
        cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),
                                        spacing='proportional',
                                        orientation='horizontal') #horizontal vertical
        cb.ax.set_xticklabels(['adpt','stim','ctr', 0, 1, 2, 3, 4, 5, 6])
        cb.set_label('number of spikes')
        
        #titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
        titleaxes = divider.append_axes("top", size="5%", pad=0.2)
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(spkrate1.shape[2]) + start_time
        
        value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)

        if show_pattern_size:
            circle_pattern = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.,color='r',fill=False)
            ax1.add_patch(circle_pattern)
        # def init():
        #     value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        #     title.set_text(u"time: {} ms".format(time_title[0]))
        #     # if show_pattern_size:
        #     #     circle_pattern = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
        #     #     ax2.add_patch(circle_pattern)
        #     #     return value1,title,circle_pattern
        #     return value1,title,
        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]]
        # #adpt = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6.1,6.1],[6.1]]]]
        # adpt = None
        #ax_init_loc_stim = [None]*2
        if stim is not None:
            if len(stim) != 1: stim = [stim]
            loca_n_stim, max_trial_stim, current_state_stim, current_trial_stim, trial_state_stim, all_done_stim, circle_stim = \
                [None], [None], [None], [None], [None], [None], [None] 
            for ax_i in range(1):
                #*ax_init_loc_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_stim[ax_i], max_trial_stim[ax_i], current_state_stim[ax_i], current_trial_stim[ax_i], \
                trial_state_stim[ax_i], all_done_stim[ax_i], circle_stim[ax_i], ax1 = init_loca(ax1, stim[ax_i], 'g')
            circle_all = []
            for cir in circle_stim:
                if cir is not None:
                    circle_all += cir
        if adpt is not None:
            if len(adpt) != 1: adpt = [adpt]
            loca_n_adpt, max_trial_adpt, current_state_adpt, current_trial_adpt, trial_state_adpt, all_done_adpt, circle_adpt = \
                [None], [None], [None], [None], [None], [None], [None] 
            for ax_i in range(1):
                #*ax_init_loc_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_adpt[ax_i], max_trial_adpt[ax_i], current_state_adpt[ax_i], current_trial_adpt[ax_i], \
                trial_state_adpt[ax_i], all_done_adpt[ax_i], circle_adpt[ax_i], ax1 = init_loca(ax1, adpt[ax_i], 'tab:purple', lw=0, fill=True, alpha=0.2)
            if 'circle_all' not in locals():
                circle_all = []
            for cir in circle_adpt:
                if cir is not None:
                    circle_all += cir        
        def updatev(i):
            value1.set_array(spkrate1[:,:,i])
            #value2.set_array(spk2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            # if show_pattern_size:
            #     circle_pattern.set_center([pattern_ctr[i,1],pattern_ctr[i,0]])
            #     circle_pattern.set_radius(pattern_size[i])
            #     print('size')
            #     return value1, title,circle_pattern#, value2
            if show_pattern_size:
                circle_pattern.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                circle_pattern.radius = pattern_size[i]
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                    # for ax_i in range(1):
                    #     if stim[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_stim[ax_i]):
                    #             if not all_done_stim[ax_i][n]:
                    #                 if i == stim[ax_i][1][n][current_trial_stim[ax_i][n],trial_state_stim[ax_i][n]]:
                    #                     if current_state_stim[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_stim[ax_i][n].radius = stim[ax_i][2][n][current_trial_stim[ax_i][n]]#pattern_size[i]
                    #                         trial_state_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = True
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                     else:
                    #                         circle_stim[ax_i][n].radius = 0
                    #                         trial_state_stim[ax_i][n] -= 1
                    #                         current_trial_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = False
                    #                         if max_trial_stim[ax_i][n] == current_trial_stim[ax_i][n]:
                    #                             all_done_stim[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_stim[ax_i][n])
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                         #sleep(1)
                if adpt is not None:
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                    # for ax_i in range(1):
                    #     if adpt[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_adpt[ax_i]):
                    #             if not all_done_adpt[ax_i][n]:
                    #                 if i == adpt[ax_i][1][n][current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n]]:
                    #                     if current_state_adpt[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_adpt[ax_i][n].radius = adpt[ax_i][2][n][current_trial_adpt[ax_i][n]]#pattern_size[i]
                    #                         trial_state_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = True
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                    #                     else:
                    #                         circle_adpt[ax_i][n].radius = 0
                    #                         trial_state_adpt[ax_i][n] -= 1
                    #                         current_trial_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = False
                    #                         if max_trial_adpt[ax_i][n] == current_trial_adpt[ax_i][n]:
                    #                             all_done_adpt[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_adpt[ax_i][n])
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, circle_pattern, title) #, circle_adpt[ax_i][1]
                else: return value1, title, circle_pattern,
                
            else:
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                if adpt is not None:                    
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, title)
                else:
                    return value1, title,
        
        #value1.set_clim(vmin=0, vmax=6)
        # ax1.axis('off')
        #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
    else:
        fig, ax = plt.subplots(1,2)
        #fig.suptitle('sensory to association strength: %.2f\nassociation to sensory strength: %.2f'
        #             %(scale_e_12[i],scale_e_21[j]))
        #ax1.set_title('sensory')
        #ax2.set_title('association')
        cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        camp_adapt = np.array([138/255,43/255,226/255,1.])
        cmap_c = np.array([1.,0.,0.,1.])
        cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        cmap = np.vstack((camp_adapt,cmap_stimulus,cmap_c,cmap_spk(range(7))))
        cmap = mpl.colors.ListedColormap(cmap)
        #cmap.set_under('grey')
        bounds = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) - 0.5
        
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 
        
        cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),#+0.5,
                                        spacing='proportional',
                                        orientation='horizontal') #horizontal vertical
        cb.ax.set_xticklabels(['adpt','stim','ctr', 0, 1, 2, 3, 4, 5, 6])
        cb.set_label('number of spikes')
        
        titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(spkrate1.shape[2]) + start_time
        
        value1=ax[0].matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        value2=ax[1].matshow(spkrate2[:,:,0], cmap=cb.cmap, norm=cb.norm)
        

        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
        #         [np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]]
        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
        #         None]
        # adpt = [None,\
        #         [np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6,6],[6]]]]
        #ax_init_loc_stim = [None]*2
        if stim is not None:
            loca_n_stim, max_trial_stim, current_state_stim, current_trial_stim, trial_state_stim, all_done_stim, circle_stim = \
                [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
            for ax_i in range(2):
                #*ax_init_loc_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_stim[ax_i], max_trial_stim[ax_i], current_state_stim[ax_i], current_trial_stim[ax_i], \
                trial_state_stim[ax_i], all_done_stim[ax_i], circle_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
            circle_all = []
            for cir in circle_stim:
                if cir is not None:
                    circle_all += cir
        if adpt is not None:
            loca_n_adpt, max_trial_adpt, current_state_adpt, current_trial_adpt, trial_state_adpt, all_done_adpt, circle_adpt = \
                [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
            for ax_i in range(2):
                #*ax_init_loc_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_adpt[ax_i], max_trial_adpt[ax_i], current_state_adpt[ax_i], current_trial_adpt[ax_i], \
                trial_state_adpt[ax_i], all_done_adpt[ax_i], circle_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], adpt[ax_i], 'tab:purple', lw=0, fill=True, alpha=0.2)
            if 'circle_all' not in locals():
                circle_all = []
            for cir in circle_adpt:
                if cir is not None:
                    circle_all += cir
                            
        def updatev(i):
            value1.set_array(spkrate1[:,:,i])
            value2.set_array(spkrate2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            if stim is not None:
                detect_event(i, 2, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
            if adpt is not None:
                detect_event(i, 2, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                             trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                

            if stim is not None or adpt is not None:
                return (value1, value2, *circle_all, title) #, circle_adpt[ax_i][1]
            else:
                return value1, value2, title,
        
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].set_title(ax1title)
        ax[1].set_title(ax2title)
        
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev, frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
#%%
def init_loca(ax, stim, color='g', lw=1, fill=False, alpha=None):
        if stim is not None:  
            #print('True')
            stim[0] = np.array(stim[0])
            if len(np.shape(stim[0])) == 1:
                loca_n = 1
                stim[0] = np.array(stim[0]).reshape(-1,2)                
            else: 
                loca_n = np.shape(stim[0])[0]
                stim[0] = np.array(stim[0])
            
            for j in range(len(stim[1])):
                stim[1][j] = np.array(stim[1][j])
                #print(type(stim[1][j]))
                if len(stim[1][j].shape) == 1:
                    stim[1][j] = stim[1][j].reshape(-1,2)
                    #print(stim[1][j])
            for j in range(len(stim[2])):
                stim[2][j] = np.array(stim[2][j])  
                       
            max_trial = []
            for trial in stim[1]:
                max_trial.append(np.shape(trial)[0]) 
                #print(max_trial)
                
            current_state = np.array([False]*loca_n)
            current_trial = np.array([0]*loca_n)
            all_done = np.array([False]*loca_n)
            trial_state = np.array([0]*loca_n)
            circle = []            
            for n in range(loca_n):
#                circle.append(plt.Circle([stim[0][n,1],stim[0][n,0]],stim[2][n][0], lw=1.,color='r',fill=False))
                circle.append(plt.Circle([stim[0][n,1],stim[0][n,0]],0, lw=lw,color=color,fill=fill,alpha=alpha))               
                ax.add_patch(circle[n])
        else:
            loca_n = None
            max_trial = None
            current_state = None
            current_trial = None
            trial_state = None
            all_done = None
            circle = None
        
        return loca_n, max_trial, current_state, current_trial, trial_state, all_done, circle,  ax
#%%
def detect_event(frame, ax_num, event, loca_n, all_done, current_trial, trial_state, current_state, circle, max_trial):
    for ax_i in range(ax_num):
        if event[ax_i] is not None:
            #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
            #if frame%100 == 0: print(current_trial)
            for n in range(loca_n[ax_i]):
                if not all_done[ax_i][n]:
                    if frame == event[ax_i][1][n][current_trial[ax_i][n],trial_state[ax_i][n]]:
                        if current_state[ax_i][n] == False:
                            #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                            circle[ax_i][n].radius = event[ax_i][2][n][current_trial[ax_i][n]]#pattern_size[i]
                            trial_state[ax_i][n] += 1
                            current_state[ax_i][n] = True
                            #print(ax_i,current_trial[ax_i][n],trial_state[ax_i][n],current_state[ax_i][n],all_done[ax_i][n])
                        else:
                            circle[ax_i][n].radius = 0
                            trial_state[ax_i][n] -= 1
                            current_trial[ax_i][n] += 1
                            current_state[ax_i][n] = False
                            #print(current_trial)
                            if max_trial[ax_i][n] == current_trial[ax_i][n]:
                                all_done[ax_i][n] = True
                                #print(ax_i,n,'all done')
                                #print(ax_i,all_done[ax_i][n])
                            #print(ax_i,current_trial[ax_i][n],trial_state[ax_i][n],current_state[ax_i][n],all_done[ax_i][n])
#%%
#%%
def show_pattern_ms(spkrate1, spkrate2=None, frames = 1000, start_time = 0, interval_movie=10, ax1title='', ax2title='', anititle='', show_pattern_size=False, pattern_ctr=None, pattern_size=None,\
                 stim=None, adpt=None):
    '''
    create amination for the firing pattern of network
    input:
    spkrate1, spkrate2: (N*N*t array)
    frames: number of frames of amination
    start_time: the real-time in the simulation that the first frame of 'spkrate' cooresponds to
    anititle: set the title of amination
    
    stim: location(coordinate is expressed in matrix convention, not in cartesian), onset and offset time, and size of stimuli; for indicating the position of stimulus
    e.g. stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
          [area1:[[[loca1],  [loca2]],  [on_off_1(2D array), on_off_2], [[size for each on_off_1], [size for each on_off_2]]], area2:None]
    multiple stimuli are added at [31.5,31.5] and [63.5,-0.5]; on and off time of each stimulus appearred at [31.5,31.5] and [63.5,-0.5] are
    specified in stim_on_off, stim_on_off; size(radius) of each stimulus: [6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]
       
    adpt: location, onset and offset time, and size of adaptation modulation; for indicating the position of adaptation modulation
    e.g. adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[chg_adapt_range]]]]

    '''
    if spkrate2 is None:
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize = (6,6))
        ax1= fig.add_subplot(111, label="1",frame_on=False)
        ax1.axis('off')
        divider = make_axes_locatable(ax1)
        cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
            
        cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        camp_adapt = np.array([138/255,43/255,226/255,1.])
        cmap_c = np.array([1.,0.,0.,1.])
        cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        cmap = np.vstack((camp_adapt,cmap_stimulus,cmap_c,cmap_spk(range(7))))
        cmap = mpl.colors.ListedColormap(cmap)
        #cmap.set_under('red')
        bounds = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) - 0.5
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
        
        cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),
                                        spacing='proportional',
                                        orientation='horizontal') #horizontal vertical
        cb.ax.set_xticklabels(['adpt','stim','ctr', 0, 1, 2, 3, 4, 5, 6])
        cb.set_label('number of spikes')
        
        #titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
        titleaxes = divider.append_axes("top", size="5%", pad=0.2)
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(spkrate1.shape[2]) + start_time
        
        value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)

        if show_pattern_size:
            circle_pattern = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.,color='r',fill=False)
            ax1.add_patch(circle_pattern)
        # def init():
        #     value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        #     title.set_text(u"time: {} ms".format(time_title[0]))
        #     # if show_pattern_size:
        #     #     circle_pattern = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
        #     #     ax2.add_patch(circle_pattern)
        #     #     return value1,title,circle_pattern
        #     return value1,title,
        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]]
        # #adpt = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6.1,6.1],[6.1]]]]
        # adpt = None
        #ax_init_loc_stim = [None]*2
        if stim is not None:
            if len(stim) != 1: stim = [stim]
            loca_n_stim, max_trial_stim, current_state_stim, current_trial_stim, trial_state_stim, all_done_stim, circle_stim = \
                [None], [None], [None], [None], [None], [None], [None] 
            for ax_i in range(1):
                #*ax_init_loc_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_stim[ax_i], max_trial_stim[ax_i], current_state_stim[ax_i], current_trial_stim[ax_i], \
                trial_state_stim[ax_i], all_done_stim[ax_i], circle_stim[ax_i], ax1 = init_loca(ax1, stim[ax_i], 'g')
            circle_all = []
            for cir in circle_stim:
                if cir is not None:
                    circle_all += cir
        if adpt is not None:
            if len(adpt) != 1: adpt = [adpt]
            loca_n_adpt, max_trial_adpt, current_state_adpt, current_trial_adpt, trial_state_adpt, all_done_adpt, circle_adpt = \
                [None], [None], [None], [None], [None], [None], [None] 
            for ax_i in range(1):
                #*ax_init_loc_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_adpt[ax_i], max_trial_adpt[ax_i], current_state_adpt[ax_i], current_trial_adpt[ax_i], \
                trial_state_adpt[ax_i], all_done_adpt[ax_i], circle_adpt[ax_i], ax1 = init_loca(ax1, adpt[ax_i], 'C1', lw=1.5, fill=False, alpha=1)
            if 'circle_all' not in locals():
                circle_all = []
            for cir in circle_adpt:
                if cir is not None:
                    circle_all += cir        
        def updatev(i):
            value1.set_array(spkrate1[:,:,i])
            #value2.set_array(spk2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            # if show_pattern_size:
            #     circle_pattern.set_center([pattern_ctr[i,1],pattern_ctr[i,0]])
            #     circle_pattern.set_radius(pattern_size[i])
            #     print('size')
            #     return value1, title,circle_pattern#, value2
            if show_pattern_size:
                circle_pattern.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                circle_pattern.radius = pattern_size[i]
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                    # for ax_i in range(1):
                    #     if stim[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_stim[ax_i]):
                    #             if not all_done_stim[ax_i][n]:
                    #                 if i == stim[ax_i][1][n][current_trial_stim[ax_i][n],trial_state_stim[ax_i][n]]:
                    #                     if current_state_stim[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_stim[ax_i][n].radius = stim[ax_i][2][n][current_trial_stim[ax_i][n]]#pattern_size[i]
                    #                         trial_state_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = True
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                     else:
                    #                         circle_stim[ax_i][n].radius = 0
                    #                         trial_state_stim[ax_i][n] -= 1
                    #                         current_trial_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = False
                    #                         if max_trial_stim[ax_i][n] == current_trial_stim[ax_i][n]:
                    #                             all_done_stim[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_stim[ax_i][n])
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                         #sleep(1)
                if adpt is not None:
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                    # for ax_i in range(1):
                    #     if adpt[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_adpt[ax_i]):
                    #             if not all_done_adpt[ax_i][n]:
                    #                 if i == adpt[ax_i][1][n][current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n]]:
                    #                     if current_state_adpt[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_adpt[ax_i][n].radius = adpt[ax_i][2][n][current_trial_adpt[ax_i][n]]#pattern_size[i]
                    #                         trial_state_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = True
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                    #                     else:
                    #                         circle_adpt[ax_i][n].radius = 0
                    #                         trial_state_adpt[ax_i][n] -= 1
                    #                         current_trial_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = False
                    #                         if max_trial_adpt[ax_i][n] == current_trial_adpt[ax_i][n]:
                    #                             all_done_adpt[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_adpt[ax_i][n])
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, circle_pattern, title) #, circle_adpt[ax_i][1]
                else: return value1, title, circle_pattern,
                
            else:
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                if adpt is not None:                    
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, title)
                else:
                    return value1, title,
        
        #value1.set_clim(vmin=0, vmax=6)
        # ax1.axis('off')
        #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
    else:
        fig, ax = plt.subplots(1,2)
        #fig.suptitle('sensory to association strength: %.2f\nassociation to sensory strength: %.2f'
        #             %(scale_e_12[i],scale_e_21[j]))
        #ax1.set_title('sensory')
        #ax2.set_title('association')
        cmap_spk=plt.cm.get_cmap('Greys', 3) # viridis Blues
        camp_adapt = np.array([255/255,127/255,14/255,1.])
        cmap_c = np.array([1.,0.,0.,1.])
        cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        cmap = np.vstack((camp_adapt,cmap_stimulus,cmap_c,cmap_spk(range(3))))
        cmap = mpl.colors.ListedColormap(cmap)
        #cmap.set_under('grey')
        bounds = np.array([-3, -2, -1, 0, 1, 2, 3 ]) - 0.5
        
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 
        
        cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=np.array([-3, -2, -1, 0, 1, 2, ]),#+0.5,
                                        spacing='proportional',
                                        orientation='horizontal') #horizontal vertical
        cb.ax.set_xticklabels(['Cue','Input','CoM', 0, 1, 2, ])
        cb.set_label('                                                                   Number of spikes')
        
        titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", fontsize=10, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(spkrate1.shape[2]) + start_time
        
        value1=ax[0].matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        value2=ax[1].matshow(spkrate2[:,:,0], cmap=cb.cmap, norm=cb.norm)
        

        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
        #         [np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]]
        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
        #         None]
        # adpt = [None,\
        #         [np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6,6],[6]]]]
        #ax_init_loc_stim = [None]*2
        if stim is not None:
            loca_n_stim, max_trial_stim, current_state_stim, current_trial_stim, trial_state_stim, all_done_stim, circle_stim = \
                [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
            for ax_i in range(2):
                #*ax_init_loc_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_stim[ax_i], max_trial_stim[ax_i], current_state_stim[ax_i], current_trial_stim[ax_i], \
                trial_state_stim[ax_i], all_done_stim[ax_i], circle_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
            circle_all = []
            for cir in circle_stim:
                if cir is not None:
                    circle_all += cir
        if adpt is not None:
            loca_n_adpt, max_trial_adpt, current_state_adpt, current_trial_adpt, trial_state_adpt, all_done_adpt, circle_adpt = \
                [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
            for ax_i in range(2):
                #*ax_init_loc_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_adpt[ax_i], max_trial_adpt[ax_i], current_state_adpt[ax_i], current_trial_adpt[ax_i], \
                trial_state_adpt[ax_i], all_done_adpt[ax_i], circle_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], adpt[ax_i], 'C1', lw=1.5, fill=False, alpha=1)
            if 'circle_all' not in locals():
                circle_all = []
            for cir in circle_adpt:
                if cir is not None:
                    circle_all += cir
                            
        def updatev(i):
            value1.set_array(spkrate1[:,:,i])
            value2.set_array(spkrate2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            if stim is not None:
                detect_event(i, 2, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
            if adpt is not None:
                detect_event(i, 2, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                             trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                

            if stim is not None or adpt is not None:
                return (value1, value2, *circle_all, title) #, circle_adpt[ax_i][1]
            else:
                return value1, value2, title,
        
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].set_title(ax1title, fontsize=10)
        ax[1].set_title(ax2title, fontsize=10)
        
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev, frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
#%%
def show_timev(v1, v2=None, vrange=[[-80,-50]], frames = 1000, start_time = 0, interval_movie=10, anititle=''):#, show_pattern_size=False, pattern_ctr=None, pattern_size=None):
    '''
    create amination for the firing pattern of network
    input:
    spkrate1, spkrate2: (N*N*t array)
    frames: number of frames of amination
    start_time: the real-time of the simulation that the first frame of 'spkrate' cooresponds to
    anititle: set the title of amination
    '''
    if v2 is None:
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize = (6,6))
        ax1= fig.add_subplot(111, label="1",frame_on=False)
        ax1.axis('off')
        divider = make_axes_locatable(ax1)
        cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
        # cb = plt.colorbar(cbaxes)
        #fig.colorbar(img1, cax=cax1)
        # if show_pattern_size:
        #     ax2=fig.add_subplot(111, label="2",frame_on=False)
        #     ax2.set_xlim([-0.5,spkrate1.shape[1]+0.5])
        #     ax2.set_ylim([-0.5,spkrate1.shape[0]+0.5])       
        #     ax2.axis('off')
            
        # cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        # cmap_c = np.array([1.,0.,0.,1.])
        # cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        # cmap = np.vstack((cmap_stimulus,cmap_c,cmap_spk(range(7))))
        # cmap = mpl.colors.ListedColormap(cmap)
        # #cmap.set_under('red')
        # bounds = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # #cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
        
        # cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
        #                                 norm=norm,
        #                                 boundaries=bounds,
        #                                 ticks=np.array([-2,-1, 0, 1, 2, 3, 4, 5, 6])+0.5,
        #                                 spacing='proportional',
        #                                 orientation='horizontal') #horizontal vertical
        # cb.ax.set_xticklabels(['stimulus','ctr', 0, 1, 2, 3, 4, 5, 6])
        # cb.set_label('number of spikes')
        
        #titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
        titleaxes = divider.append_axes("top", size="5%", pad=0.01)
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(v1.shape[2]) + start_time
        
        value1=ax1.matshow(v1[:,:,0], vmin = vrange[0][0], vmax = vrange[0][1])#,cmap=cb.cmap, norm=cb.norm)
        cb = plt.colorbar(value1, cax=cbaxes, orientation='horizontal')
        # if show_pattern_size: 
        #     ax2 = fig.add_axes([0, 0, 1, 1]) 
        #     #ax2= plt.subplot(111)
        #     circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
        #     ax2.add_patch(circle)
        # if show_pattern_size:
        #     circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.,color='r',fill=False)
        #     ax1.add_patch(circle)

        
        def updatev(i):
            value1.set_array(v1[:,:,i])
            #value2.set_array(spk2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            # if show_pattern_size:
            #     circle.set_center([pattern_ctr[i,1],pattern_ctr[i,0]])
            #     circle.set_radius(pattern_size[i])
            #     print('size')
            #     return value1, title,circle#, value2
            # if show_pattern_size:
            #     circle.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
            #     circle.radius = pattern_size[i]
            #print('size')
                # return value1, title,circle,#, value2
            #return title,circle,
            #else:
            return value1, title,
        
        #value1.set_clim(vmin=0, vmax=6)
        # ax1.axis('off')
        #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
    else:
        fig, [ax1,ax2]= plt.subplots(1,2)

        cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03])

        titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(v1.shape[2]) + start_time
        
        value1=ax1.matshow(v1[:,:,0], vmin = vrange[0][0], vmax = vrange[0][1])
        value2=ax2.matshow(v2[:,:,0], vmin = vrange[1][0], vmax = vrange[1][1])
        
        cb = plt.colorbar(value1, cax=cbaxes, orientation='horizontal')
        
        
        def updatev(i):
            value1.set_array(v1[:,:,i])
            value2.set_array(v2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            return value1, value2, title,
        
        ax1.axis('off')
        ax2.axis('off')
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev, frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani

#%%


def firing_rate_time_multi(data, neu, dura_onoff, window, sample_interval=1,\
                           n_neu_all=4096, dt=0.1):
    '''
    calculating population mean firing rate of sampled neurons during multiple period

    Parameters
    ----------
    data : class object, mydata()
        an object which has attributes regarding spkie time and neuron index 
    neu : 1-D array
        an array storing index of sampled neuron .
    dura_onoff : 2-D array, ms
        a 2-D array, each row specifying start and end of sampled period.
    window : scalar, ms
        size of window to counting spikes when calculating firing rate.
    n_neu_all:
        number of total neurons in data
    sample_interval: ms
        sample interval of firing rate
    dt: ms
        simulation time step

    Returns
    -------
    hz_t: 2-D array, hz
        population mean firing rate across sampled neurons for each sampled period

    '''
#stim_onoff, 
    #hz_t = np.zeros([dura_onoff.shape[0], int(round((dura_onoff[0,1]-dura_onoff[0,0])/sample_interval))])
    for i in range(dura_onoff.shape[0]):
        spk_rate = get_spike_rate(data, dura_onoff[i,0], dura_onoff[i,1], indiv_rate = True, popu_rate = False, \
                  sample_interval = sample_interval, n_neuron = n_neu_all, window = window, dt = dt, \
                      reshape_indiv_rate = False, save_results_to_input = False)
        if i == 0:
            hz_t = np.zeros([dura_onoff.shape[0], spk_rate.shape[-1]])

        #hz_t[i, :spk_rate.shape[-1]] = spk_rate[neu].mean(0)/(window/1000)
        hz_t[i, :] = spk_rate[neu].mean(0)/(window/1000)
    
    return hz_t
#%%
def tuning_curve(spk_matrix, dura_onoff, n_in_bin, dt=0.1):
    '''
    get tuning curve; 
    calculates population mean firing rate across neurons in each neuron group in n_in_bin  

    Parameters
    ----------
    spk_matrix : sparse matrix
        sparse matrix for spike time and neuron index.
    dura_onoff : 2-D array, ms
        a 2-D array, each row specifying start and end of sampling period.
    n_in_bin : list containing arrays
        a list containing arrays, 
        each array specifies index of group of neurons for firing rate calculation
    dt : ms
        simulation time step.

    Returns
    -------
    hz_loc : 2-D array
        firing rate of each group of neurons for each sampling period.

    '''
    hz_loc = np.zeros([dura_onoff.shape[0], len(n_in_bin)])
    dt_1 = int(round(1/dt)) # 1_dt: 1/dt
    for i in range(dura_onoff.shape[0]):
        for j in range(len(n_in_bin)):
            hz_loc[i, j] = spk_matrix[n_in_bin[j], dura_onoff[i,0]*dt_1:dura_onoff[i,1]*dt_1].sum()/n_in_bin[j].shape[0]/((dura_onoff[i,1]-dura_onoff[i,0])/1000)
    
    return hz_loc

def get_spkcount_noSubSample(spk_sparmat, dura, dt=0.1, get_firingRate=True):
    
    dt_1 = int(round(1/dt))
    spk_count = np.zeros([spk_sparmat.shape[0], len(dura)], dtype = float)
    for i, dura_t in enumerate(dura):
        spk_count[:,i] = spk_sparmat[:, dura_t[0]*dt_1:dura_t[1]*dt_1].sum(1).A.reshape(-1)
        if get_firingRate:
            spk_count[:,i] /= (dura_t[1] - dura_t[0])/1000
    
    return spk_count
    
    
    
    
class noise_corr:
    
    def __init__(self):
        
        self.spk_matrix1 = None ; # sparse matrix recording time and index of neurons that emit spikes.
        self.spk_matrix2 = None
        self.dura1 = None ; # start and end of period during which the number of spikes are counted.
        #self.dura2 = None
        self.dt = 0.1 ; # ms simulation time step
        self.pair = 'all'  # 'all' or 'random'; if calculate the noise correlation between all the possible pairs of neurons
        self.return_every_coef = False
        # below params are only for calculating the time varying noise correlation
        self.win = 100 # ms sliding window length to count spikes
        self.move_step = 20 # ms sliding window move step, (sampling interval for time varying noise correlation)
        self.t_bf = 100 # ms; time before stimulus onset to start to sample noise correlation
        self.t_aft = 100 # ms; time after stimulus off to finish sampling noise correlation
    
    def get_nc_withingroup_t(self):
        
        spk_matrix1 = self.spk_matrix1
        dura = np.round(self.dura1/self.dt).astype(int)
        t_bf = int(round(self.t_bf/self.dt)) # ms; time before stimulus onset
        t_aft = int(round(self.t_aft/self.dt)) # ms; time after stimulus off
        #dt = 0.1 # ms
        win = int(round(self.win/self.dt)) # ms sliding window
        hfwin = int(round(win/2))
        move_step = int(round(self.move_step/self.dt)) # ms sliding window moving step
        
        samp_onoff = np.copy(dura)
        samp_onoff[:,0] -= t_bf
        samp_onoff[:,1] += t_aft
        
        samp_stp = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
        spk_count = np.zeros([spk_matrix1.shape[0], samp_stp.shape[0], len(dura)])

        for i in range(samp_onoff.shape[0]):
            samp_stp = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
            samp_stp += samp_onoff[i,0]
            for stp in range(samp_stp.shape[0]):
                spk_count[:,stp,i] = spk_matrix1[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
        
        corr = np.zeros([2, spk_count.shape[1]]) # 1st row: mean; 2nd row: sem
      
        if self.return_every_coef:
            coef_samp_stp = np.arange(int(np.ceil((t_bf + round(200/self.dt) + hfwin)/move_step)), spk_count.shape[1] - int(np.ceil((t_aft+hfwin)/move_step)), int(np.ceil(win/move_step))) # neglect 200ms onset transient
            corr_all = np.zeros([round(spk_count.shape[0]*(spk_count.shape[0]-1)/2), coef_samp_stp.shape[0]])
            samp_t = 0 
          
        for stp in range(spk_count.shape[1]):
            corr_ = np.corrcoef(spk_count[:,stp,:])[np.triu_indices(spk_count.shape[0], 1)]
            corr[0,stp] = np.nanmean(corr_) #corr_.mean()
            corr[1,stp] = sem(corr_, nan_policy='omit')
            
            if self.return_every_coef:
                if stp in coef_samp_stp:
                    corr_all[:,samp_t] = corr_
                    samp_t += 1
        
        if self.return_every_coef:
            return corr, corr_all
        else:
            return corr

    def get_nc_betweengroups_t(self):
        
        spk_matrix1 = self.spk_matrix1
        spk_matrix2 = self.spk_matrix2
        dura1 = np.round(self.dura1/self.dt).astype(int)
        #dura2 = np.round(self.dura2/self.dt).astype(int)
        if isinstance(spk_matrix1, csr_matrix):
            spk_matrix1 = spk_matrix1.tocsc()
        if isinstance(spk_matrix2, csr_matrix):
            spk_matrix2 = spk_matrix2.tocsc()
            
        t_bf = int(round(self.t_bf/self.dt)) # ms; time before stimulus onset
        t_aft = int(round(self.t_aft/self.dt)) # ms; time after stimulus off
        #dt = 0.1 # ms
        win = int(round(self.win/self.dt)) # ms sliding window
        hfwin = int(round(win/2))
        move_step = int(round(self.move_step/self.dt)) # ms sliding window moving step
        
        samp_onoff = np.copy(dura1)
        samp_onoff[:,0] -= t_bf
        samp_onoff[:,1] += t_aft
        
        samp_stp = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
        spk_count1 = np.zeros([spk_matrix1.shape[0], samp_stp.shape[0], len(dura1)])
        spk_count2 = np.zeros([spk_matrix2.shape[0], samp_stp.shape[0], len(dura1)])

        for i in range(samp_onoff.shape[0]):
            samp_stp = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
            samp_stp += samp_onoff[i,0]
            for stp in range(samp_stp.shape[0]):
                spk_count1[:,stp,i] = spk_matrix1[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
                spk_count2[:,stp,i] = spk_matrix2[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
        
        corr = np.zeros([2, spk_count1.shape[1]]) # 1st row: mean; 2nd row: sem

        if self.return_every_coef:
            coef_samp_stp = np.arange(int(np.ceil((t_bf + round(200/self.dt) + hfwin)/move_step)), spk_count1.shape[1] - int(np.ceil((t_aft+hfwin)/move_step)), int(np.ceil(win/move_step))) # neglect 200ms onset transient
            corr_all = np.zeros([spk_count1.shape[0]*spk_count2.shape[0], coef_samp_stp.shape[0]])
            samp_t = 0 
        
        for stp in range(spk_count1.shape[1]):
            corr_ = np.corrcoef(spk_count1[:,stp,:], spk_count2[:,stp,:])[:spk_count1.shape[0], spk_count1.shape[0]:].reshape(-1) #[np.triu_indices(spk_count.shape[0], 1)]
            corr[0,stp] = np.nanmean(corr_) #corr_.mean()
            corr[1,stp] = sem(corr_, nan_policy='omit')

            if self.return_every_coef:
                if stp in coef_samp_stp:
                    corr_all[:,samp_t] = corr_
                    samp_t += 1
    
        if self.return_every_coef:
            return corr, corr_all
        else:
            return corr
        
    
    def get_nc_withingroup(self):#, self.spk_matrix, self.dura, self.dt=0.1, self.pair='all'):
        
        spk_matrix1 = self.spk_matrix1
        dura = np.round(self.dura1/self.dt).astype(int)
        
        if isinstance(spk_matrix1, csr_matrix):
            spk_matrix1 = spk_matrix1.tocsc()
        spk_count = np.zeros([spk_matrix1.shape[0], len(dura)])
        for t in range(len(dura)):
            spk_count[:, t] = spk_matrix1[:, dura[t][0]:dura[t][1]].sum(1).A.reshape(-1)
        spk_count = spk_count.astype(float)
        if self.pair == 'all':
            spk_count -= spk_count.mean(1).reshape(-1,1)
            spk_count /= np.sqrt(np.sum(spk_count**2, 1)).reshape(-1,1)
            corr = np.dot(spk_count, spk_count.T)
            corr = corr[np.triu_indices(corr.shape[0], 1)]
            corr_mean = np.nanmean(corr) #corr.mean()
            corr_sem = sem(corr, nan_policy='omit')
        if self.return_every_coef:
            return corr, corr_mean, corr_sem
        else:
            return corr_mean, corr_sem
    
    def get_nc_betweengroups(self):
        
        spk_matrix1 = self.spk_matrix1
        spk_matrix2 = self.spk_matrix2
        dura1 = np.round(self.dura1/self.dt).astype(int)
        #dura2 = np.round(self.dura2/self.dt).astype(int)
        if isinstance(spk_matrix1, csr_matrix):
            spk_matrix1 = spk_matrix1.tocsc()
        if isinstance(spk_matrix2, csr_matrix):
            spk_matrix2 = spk_matrix2.tocsc()
        
        spk_count1 = np.zeros([spk_matrix1.shape[0], len(dura1)])
        spk_count2 = np.zeros([spk_matrix2.shape[0], len(dura1)])
        
        for t in range(len(dura1)):
            spk_count1[:, t] = spk_matrix1[:, dura1[t][0]:dura1[t][1]].sum(1).A.reshape(-1)
        #for t in range(len(dura2)): 
            spk_count2[:, t] = spk_matrix2[:, dura1[t][0]:dura1[t][1]].sum(1).A.reshape(-1)

        spk_count1 = spk_count1.astype(float)
        spk_count2 = spk_count2.astype(float)
        
        if self.pair == 'all':
            spk_count1 -= spk_count1.mean(1).reshape(-1,1)
            spk_count2 -= spk_count2.mean(1).reshape(-1,1)
            spk_count1 /= np.sqrt(np.sum(spk_count1**2, 1)).reshape(-1,1)
            spk_count2 /= np.sqrt(np.sum(spk_count2**2, 1)).reshape(-1,1)

            corr = np.dot(spk_count1, spk_count2.T)
            #corr = corr[np.triu_indices(corr.shape[0], 1)]m
            corr = corr.reshape(-1)
            corr_mean = np.nanmean(corr) #corr.mean()
            corr_sem = sem(corr, nan_policy='omit')
        if self.return_every_coef:
            return corr, corr_mean, corr_sem
        else:
            return corr_mean, corr_sem
        

def get_mean_var(spk_mat, dura, dt=0.1):
    '''
    

    Parameters
    ----------
    spk_mat : sparse matrix
        spike time and neuron index.
    dura : 2-D array
        start and end time of analysis period of each response.
    dt : scalar (ms)
        simulation time step. The default is 0.1 ms.

    Returns
    -------
    mean_var : 2-D array
        first colum: mean; second colum: variance.

    '''
    spk_mat = spk_mat.tocsc()
    
    mean_var = np.zeros([spk_mat.shape[0], 2])
    resp = np.zeros([spk_mat.shape[0] , dura.shape[0]])
    dt_ = int(round(1/dt))
    
    for i in range(dura.shape[0]):
        
        resp[:,i] = spk_mat[:, dura[i,0]*dt_:dura[i,1]*dt_].sum(1).A.reshape(-1)
    
    mean_var[:,0] = resp.mean(1)
    mean_var[:,1] = resp.var(1)
    
    return mean_var

def get_mean_var_singletrial(spk_mat, dura, win, sample_interval, dt=0.1):
    '''
    
    Parameters
    ----------
    spk_mat : sparse matrix
        spike time and neuron index.
    dura : 2-D array
        start and end time of analysis period of each response.
    win : (ms) 
        length of window to count spike
    sample_interval : (ms)
        sampling interval
    dt : scalar (ms)
        simulation time step. The default is 0.1 ms.

    Returns
    -------
    mean_var : 3-D array
        mean and var for each trial; first colum: mean; second colum: variance.
    fano : 2-D array
        fano for each trial.

    '''
    
    fano = np.zeros([spk_mat.shape[0],len(dura)])
    mean_var = np.zeros([spk_mat.shape[0],2,len(dura)])
    for i in range(dura.shape[0]):
        
        
        spk_count = get_spkcount_sparmat(spk_mat, dura[i,0], dura[i,1],\
                   sample_interval,  win, dt)    
        
        mean_var[:,0,i] = spk_count.mean(1)
        mean_var[:,1,i] = spk_count.var(1)
        fano[:,i] = mean_var[:,1,i]/mean_var[:,0,i]
    
    return fano, mean_var



def get_spkcount_sparmat(spk_sparmat, start_time, end_time,\
                   sample_interval = 1,  window = 10, dt = 0.1):
    
    sample_interval = int(np.round(sample_interval/dt))
    start_time = int(np.round(start_time/dt))
    end_time = int(np.round(end_time/dt))
    window_step = int(np.round(window/dt))

#################   
    if not isinstance(spk_sparmat, csc_matrix):        
        spk_sparmat = spk_sparmat.tocsc()
    
    sample_t = np.arange(start_time, end_time-window_step+1, sample_interval)
    spk_count = np.zeros([spk_sparmat.shape[0], sample_t.shape[0]], dtype=int)
    
    for i in range(sample_t.shape[0]):
        
        neu, counts = np.unique(spk_sparmat.indices[spk_sparmat.indptr[sample_t[i]]:spk_sparmat.indptr[sample_t[i]+window_step]],return_counts=True)
        spk_count[:, i][neu] += counts
    
    return spk_count

def get_spkcount_sum_sparmat(spk_sparmat, start_time, end_time,\
                   sample_interval = 1,  window = 10, dt = 0.1):
    
    sample_interval = int(np.round(sample_interval/dt))
    start_time = int(np.round(start_time/dt))
    end_time = int(np.round(end_time/dt))
    window_step = int(np.round(window/dt))

#################   
    if not isinstance(spk_sparmat, csc_matrix):        
        spk_sparmat = spk_sparmat.tocsc()
    
    sample_t = np.arange(start_time, end_time-window_step+1, sample_interval)
    spk_count = np.zeros(sample_t.shape[0], dtype=int)
    
    for i in range(sample_t.shape[0]):
        
        # neu, counts = np.unique(spk_sparmat.indices[spk_sparmat.indptr[sample_t[i]]:spk_sparmat.indptr[sample_t[i]+window_step]],return_counts=True)
        # spk_count[:, i][neu] += counts
        
        spk_count[i] = spk_sparmat.indptr[sample_t[i]+window_step] - spk_sparmat.indptr[sample_t[i]]
        
    return spk_count

#%%
def get_spkcount_sparmat_multi(spk_sparmat, dura, sum_activity=True, \
                   sample_interval = 1,  window = 10, dt = 0.1):
    
    for dura_i, dura_t in enumerate(dura):
        
        if sum_activity:
            spk_count_i = get_spkcount_sum_sparmat(spk_sparmat, dura_t[0], dura_t[1],\
                   sample_interval = sample_interval,  window = window, dt = dt)
            if dura_i == 0:
                spk_count = np.zeros([len(dura), spk_count_i.shape[-1]])
            spk_count[dura_i] = spk_count_i
        else:
            spk_count_i = get_spkcount_sparmat(spk_sparmat, dura_t[0], dura_t[1],\
                   sample_interval = sample_interval,  window = window, dt = dt)
            if dura_i == 0:
                spk_count = np.zeros([len(dura), spk_count_i.shape[0], spk_count_i.shape[1]])
                # print(spk_count.shape)
            spk_count[dura_i] = spk_count_i
        
    return spk_count

#%%
def get_spkcount_sparmat_multi_unequalDura(spk_sparmat, dura, sum_activity=True, \
                   sample_interval = 1,  window = 10, dt = 0.1):
    '''
    unequalDura: length of duration of each period in dura may not be equal.
    output spk_count is in format of 'list' instead of numpy array
    '''
    spk_count = []
    for dura_i, dura_t in enumerate(dura):
        
        
        if sum_activity:
            spk_count_i = get_spkcount_sum_sparmat(spk_sparmat, dura_t[0], dura_t[1],\
                   sample_interval = sample_interval,  window = window, dt = dt)
            # if dura_i == 0:
            #     spk_count = np.zeros([len(dura), spk_count_i.shape[-1]])
            # spk_count[dura_i] = spk_count_i
            spk_count.append(spk_count_i)
        else:
            spk_count_i = get_spkcount_sparmat(spk_sparmat, dura_t[0], dura_t[1],\
                   sample_interval = sample_interval,  window = window, dt = dt)
            # if dura_i == 0:
            #     spk_count = np.zeros([len(dura), spk_count_i.shape[0], spk_count_i.shape[1]])
            #     print(spk_count.shape)
            # spk_count[dura_i] = spk_count_i
            spk_count.append(spk_count_i)

        
    return spk_count


