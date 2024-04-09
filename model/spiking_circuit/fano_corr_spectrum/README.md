- `two_inputs_simu.py` is the script for simulating the spiking circuit model.
  Two external inputs are added to the network. The 1-input condition can be simulated by changing the input number (see the comments at the beginning of this script).  
  Run `two_inputs_simu.py` with one input argument, e.g.,
  
  ```
  python two_inputs_simu.py 0
  ```
  
  Here, `0` is the input argument, which specifies the index of the random network realization.
  By default, this index is the seed of random number generator.
  The default maximum value of this index can be changed by changing the 'repeat' variable in 'two_inputs_simu.py'.
  
  The output of `two_inputs_simu.py` is a data file and/or movies.
  The data file (`data0.file` in this case) contains the spiking information of neurons and/or the recorded local field potential (LFP).

- `analy_fano_corr_spect.py` is for the analyses of power spectrum, wavelet spectrogram, and top-down attention's effect on Fano factor and noise correlation.
  Run it by
  ```
  python analy_fano_corr_spect.py 0
  ```
  Here, `0` is the index of the network realization on which the analysis will be performed.
  
- `run.pbs` is a PBS script for running simulations and analysis for multiple networks simultaneously on cluster.
  
