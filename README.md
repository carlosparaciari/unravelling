# Unravelling CQ dynamics

[![Build Status](https://travis-ci.com/carlosparaciari/unravelling.svg?token=qysu8rvspZL66s8hKeeJ&branch=master)](https://travis-ci.com/carlosparaciari/unravelling)

Code for simulating classical-quantum master equations for finite-dimensional systems, using the unravelling technique. This code simulates the stochastic evolution of the classical and quantum degrees of freedom of a system by generating multiple trajectories, which can later be averaged to obtain the solution of the master equation. Further details on the unravelling method for classical-quantum systems can be found in *name of paper*, see ArXiv *number*.

The repository contains two example files which solve the dynamics of a two-level quantum system whose internal degrees of freedom interact with its classical position and momentum. Specifically,

- `main.py` : the main file uses the unrvelling tools for creating multiple trajectories which describe the evolution of the system under a specific dynamics.

- `analysis.ipynb` : the notebook is used to average the trajectories created with the main file, and to visualise the evolution of the system in phase space.
