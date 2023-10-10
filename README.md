# Distributed_and_dynamical_communication
Code associated with manuscript "Distributed and dynamical communication: A mechanism for flexible cortico-cortical interactions and its functional roles in visual attention"

Required packages:
Numpy Brian2 Matplotlib Scipy Scikit-learn 

Run 'twoInputs.py' with one command line argument. The argument is an non-negative integer number (e.g., 0) which defines the seed of random number generator. Each seed corresponds to one random network realization. The default maximum value of the argument is set to 1 (i.e., two random network realizations can be generated). This default maximum value can be changed by changing the 'repeat' variable in 'twoInputs.py'.

The output of 'twoInputs.py' are 4 files:
1. 'data.file' (saved in ./raw_data), containing the LFP and spikes data.
2. 'spontaneous.mp4'. The movie for spontaneous activity.
3. 'twoInputs_uncued.mp4'. The movie for the two-input condition without cue.
4. 'twoInputs_cued.mp4'. The movie for the two-input condition with one input being cued.
