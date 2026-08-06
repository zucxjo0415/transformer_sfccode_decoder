# transformer_sfccode_decoder
Using an autoregressive transformer as a high-level decoder for a rotated surface code.

## Overview

Consider a rotated surface code circuit of distance $d$. It has $d^2$ data qubits and $d^2-1$ syndrome qubits, of which half are X syndromes and half are Z syndromes.

We use stim to simulate a noisy* memory experiment on this circuit for $r$ rounds and record syndrome measurements ($r(d^2-1) bits per shot). We use a MWPM decoder on the syndrome data to predict when there is a logical X error, and record the difference between this prediction and a logical X measurement of the data qubits (1 bit per shot). 

*depolarization noise after measurements, between rounds, Z errors before and after resets, all independently with probability p.

This gives us a joint distribution on $(F_2)^{(d^2-1)r+1}$.

We then use a transformer decoder to predict the last bit (whether the MWPM decoder correctly predicts a logical X error) given the $r(d^2-1)$ syndrome measurements), as a high-level decoder / decoding correction. 

## Some details

Model hyperparameters: 6 heads, 6 hidden input dimension per head, 0.1 dropout. Training hyperparameters: learning rate 1E-5, batch size 128, using Adam.
Since the classes were highly imbalanced (the MWPM decoder made incorrect predictions in <10% of instances), we used a weighted cross-entropy loss (with weights inversely proportional to the class proportions; this should be equivalent to oversampling the minority class / undersampling the majority class).

$d=11$, $r=9$, $p=0.007$, 300,000,000 shots: MWPM decoder incorrectly predicts outcome in about 2.8% of cases. Due to memory issues, the transformer was trained only on 2,000,000 shots. This did not achieve good performance (likely due to insufficient data).

$d=5$, $r=5$, $p=0.1$, 15,000,000 shots: MWPM decoder incorrectly predicts outcome in about 8.4% of cases. This did slightly better, but still not great. 

$d=3$, $r=3$, $p=0.1$, 75,000,000 shots: MWPM decoder incorrectly predicts outcome in about 6% of cases. 
Best F1: about 0.28; at present, the decoder is unable to correct more logical errors than it introduces. 

## Files in repo
- `stimcode.py`: Python code using `stim` to simulate data
- `art sfccode decoder.ipynb`: Jupyter notebook containing Transformer code (PyTorch).
- `jax art sfccode decoder.ipynb`: Jupyter notebook containing Transformer code (Jax).

## Some things to try 
- Clean up the two notebooks, build out the Jax notebook more
- A more sophisticated weighted loss function such as focal loss
- Potentially a binary transformer (since we are using all binary inputs), with an appropriate positional encoding 
