"""
Created on Thu Aug  5 15:09:36 2021

@author: Shencong Ni
"""

import numpy as np
# onoff_bool = np.ones(10, dtype=bool)
# onoff_bool[-1] = False
# onoff_bool = onoff_bool[::-1]
# onoff_bool[-1] = False
# d = np.random.randint(1,10, 10)
#%%
def get_onoff_cpts(onoff_bool):
    
    onoff_bool_tmp = np.concatenate(([0], onoff_bool, [0]))
    c = np.where(np.diff(onoff_bool_tmp) != 0)[0]
    onset_t = c[0::2]
    offset_t = c[1::2]
    on_t = c[1::2] - c[0::2]
    #on_amp = []
     
    # for pts_str, pts_end in zip(c[0::2],c[1::2]):
    #     on_amp.append(np.mean(d[pts_str:pts_end]))
    # on_amp = np.array(on_amp)
    
    if onoff_bool[0] == 1:
        on_t = np.delete(on_t,0) # discard first points if they are on states
        # on_amp = np.delete(on_amp,0)
        onset_t = np.delete(onset_t,0)
    
    if onoff_bool[-1] == 1:
        if len(on_t) != 0:
            on_t = np.delete(on_t,-1) # discard end points if they are on states
            # on_amp = np.delete(on_amp,-1) 
        offset_t = np.delete(offset_t,-1)
            
    onoff_bool_tmp = np.concatenate(([1], onoff_bool, [1]))
    c = np.where(np.diff(onoff_bool_tmp) != 0)[0]
    off_t = c[1::2] - c[0::2]
    #off_amp = []
    
    # for pts_str, pts_end in zip(c[0::2],c[1::2]):
    #     off_amp.append(np.mean(d[pts_str:pts_end]))
    # off_amp = np.array(off_amp)
    
    if onoff_bool[0] == 0:
        off_t = np.delete(off_t,0) # discard first points if they are off states
        #off_amp = np.delete(off_amp,0)
    if onoff_bool[-1] == 0:
        if len(off_t) != 0:    
            off_t = np.delete(off_t,-1)  # discard end points if they are off states
            #off_amp = np.delete(off_amp,-1)
    
    del onoff_bool_tmp        

    return on_t, off_t, onset_t, offset_t
#%%


