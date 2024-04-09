`two_inputs_simu.py` is the script for simulating the spiking circuit model.
Two external inputs are added to the network. The 1-input condition can be simulated by changing the input number.
Run `two_inputs_simu.py` with one input argument, e.g.,

```
python two_inputs_simu.py 0
```

Here `0` is the input argument, which specify the index of the random network realization.
By default, this index is the seed of random number generator.
The default maximum value of this index can be changed by changing the 'repeat' variable in 'two_inputs_simu.py'.

The output of `two_inputs_simu.py` is a data file and/or movies.
The data file (`data0.file` in this case) contains the spiking information of neurons and/or the recorded local field potential (LFP).

`onoff_detection.py` is for detecting the On and Off states.
Run it by
```
python onoff_detection.py 0
```
Here `0` is the index of the network realization on which the analysis will be performed.

`phaselockingvalue.py` is for calculating the phase locking value betweem MUA, during the On and Off states, respectively.
Similarly, run it by
```
python analy_fano_corr_spect.py 0
```

`transfer_entropy.py` is for calculating the transfer entropy betweem MUA, during the On and Off states, respectively.
Run it by
```
python transfer_entropy.py 0
```

`extractDataforRRRforEachNet.py` is for extracting spiking data for reduced rank regression (RRR) analysis for each network realization.
`combineDataforRRR.py` combines together the extracted spiking data of each network, which would be analysed by `RRR.m`.
Run them by
```
python extractDataforRRRforEachNet.py 0
python extractDataforRRRforEachNet.py 1
python extractDataforRRRforEachNet.py 2
...
python extractDataforRRRforEachNet.py 29

python combineDataforRRR.py 

matlab -r "RRR"
```
Here 30 networks (0-29) are analyzed. 


`run.pbs` is a PBS script for running simulations and analysis for multiple networks simultaneously on cluster.
Note that `combineDataforRRR.py` and `RRR.m` still need to be run separately after running `run.pbs` to complete the RRR analysis. 
