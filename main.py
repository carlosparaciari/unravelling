import lib
import numpy as np
import random as rn
from datetime import datetime
from math import sqrt
from copy import copy

""" Stern Gerlach with diagonal Lindblad operators.

	This main file, together with the ipython notebook in this repository,
	is an example created for helping the user familiarise with the code.
	In this main, the trajectories of a CQ system representing a qubit
	particle interacting with its own position degrees of freedom are created.
	In the python notebook, these trajectories are processed and visualised.
"""

""" The initial CQ state is:
	- quantum state : \ket{+}, eigenstate of Pauli X with eigenvalue 1
	- position : 0, start with zero position 
	- momentum : 0, start with zero momentum 
"""
CQstate = lib.CQState(np.matrix([[1./sqrt(2.) + 0.j],[1./sqrt(2.) + 0.j]]),0.,0.)

""" The time scale:
	- tau : rate of jump
	- delta_time : time interval for single evolution
	- final time : final time we are interested in
"""
tau = 1e-2
delta_time = 2.5e-5
final_time = 5e-2

# Random seed
seed = 1424
rn.seed(seed)

# Number of iterations
iterations = 200

# Classical-quantum interaction strength
B = 1.

# Fully quantum Hamiltonian coefficient
gamma = 1e-4

""" The dynamical configuration is

	- Classical-quantum interaction Hamiltonian:

		- H = h0 L0^dagger L0 + h1 L1^dagger L1

	- Lindblad operators:

		- L0 = \sqrt(|w0|) \ket{0}\bra{0}
		- L1 = \sqrt(|w1|) \ket{1}\bra{1}

	- Coupling factors:

		- h0 = sign(w0) q B
		- h1 = sign(w1) q B

	- Coupling constants:

		- w0 = +1
		- w1 = -1

	- Quantum Hamiltonian:

		- Hq = gamma X

	- Classical Hamiltonian:

		- Hc = p^2/2
"""
def quantum_and_interaction_Hamiltonian(B_field, gamma):
	return lambda q,p : q * B_field * np.matrix([[1.,0.],[0.,-1.]]) + gamma * np.matrix([[0.,1.],[1.,0.]])

QHamlitonian = quantum_and_interaction_Hamiltonian(B,gamma)

L0 = np.matrix([[1.,0.],[0.,0.]])
L1 = np.matrix([[0.,0.],[0.,1.]])

qu_pos_derivs = [lambda q,p : B,
				 lambda q,p : - B
				]
qu_mom_derivs = [lambda q,p : 0.,
				 lambda q,p : 0.
				]

clas_pos_derivs = lambda q,p : 0.
clas_mom_derivs = lambda q,p : p

# Measure time before simulation
tstart = datetime.now()

for iteration in range(iterations):
	# Filename for output
	filename = './output/trajectory_{0}.dat'.format( str(iteration) )

	# Create unravelling and evolve
	stern_gerlach = lib.Unravelling( copy(CQstate),
									 [L0,L1],
									 qu_pos_derivs,
									 qu_mom_derivs,
									 QHamlitonian,
									 clas_pos_derivs,
									 clas_mom_derivs,
									 tau,
									 delta_time,
									 final_time,
									 seed,
									 filename
								   )
	stern_gerlach.evolution()
	del stern_gerlach

# Measure time after simulation
tfinal = datetime.now()

# Print time for loading and for averaging
tsim = tfinal - tstart
print('Simulation time is {}\n'.format(tsim))