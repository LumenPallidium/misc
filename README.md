# Miscellaneous 
Just files that are short and didn't have another home;

## baby_diffusion
I was somewhat confused with how diffusion models *really* worked (a few ELBOs tell you nothing about the netwokr) even after reading a number of papers, so I thought it best to implement one.

This is a pretty minimal model and well-annotated, as I shared it with a few people. Here are each of the CIFAR10 classes after training for 100 epochs with a model with 820K parameters (not sure if this is good or bad amount!)

[!diffusion](samples/diffusion_04500.png)

## koopman_learning
(WIP)
Implementation of [Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition](https://arxiv.org/pdf/1710.04340.pdf), a neural network approach for learning the measurement operators in Koopman theory / Dynamic Mode Decomposition.

## neural least action
(WIP)
Jax implementation of [A Neuronal Least-Action Principle for Real-Time Learning in Cortical Circuits](https://www.biorxiv.org/content/10.1101/2023.03.25.534198v1), a multi-compartmental neural ODE with local learning rules modeling cortical circuits.