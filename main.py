import lib
import numpy as np
import random as rn
from datetime import datetime
from math import sqrt
from copy import copy

""" The initial CQ state is:
	- state : \ket{1}, eigenstate of Pauli Z with eigenvalue 1
	- position : 0.1, start a bit displaced from the centre
	- momentum : 0, start with zero momentum 
"""
CQstate = lib.CQState(np.matrix([[1.],[0.]]),1e-1,0.)

""" The time scale:
	- tau : rate of jump
	- delta_time : time interval for single evolution
	- final time : final time we are interested in
"""
tau = 1e-2
delta_time = 1e-4
final_time = 1e-1

# Random seed
seed = 365
rn.seed(seed)

# Number of iterations
iterations = int(10)

# Classical-quantum interaction strength
B = 5e3

""" The dynamical configuration is
	- Lindblad :
		- L0 send \ket{0} in \ket{0}, and |w0| = 1
		- L1 send \ket{1} in \ket{1}, and |w1| = 1
	- Interaction Hamiltonian
		- H = h0 L0^dagger L0 + h1 L1^dagger L1
		- h0 = +q^2 B
		- h1 = -q^2 B
	- Classical Hamiltonian is
		- Hc = p^2/2
		- dHcdq = 0
		- dHcdp = p
"""
def interaction_Hamiltoian(B_field):
	return lambda q,p : q**2 * B_field * np.matrix([[1.,0.],[0.,-1.]])

L0 = np.matrix([[1.,0.],[0.,0.]])
L1 = np.matrix([[0.,0.],[0.,1.]])
QHamlitonian = interaction_Hamiltoian(B)
clas_pos_derivs = lambda q,p : 0
clas_mom_derivs = lambda q,p : p

""" The derivative of h0 and h1 is
	- dh0/dq = 2qB , dh0/dp = 0
	- dh1/dq = -2qB , dh1/dp = 0
"""
def derivatives_quantum_h_position(B_field):
	dh0dq = lambda q,p : 2 * q * B_field
	dh1dq = lambda q,p : -2 * q * B_field
	return [dh0dq, dh1dq]

def derivatives_quantum_h_momentum():
	dh0dp = lambda q,p : 0
	dh1dp = lambda q,p : 0
	return [dh0dp, dh1dp]

qu_pos_derivs = derivatives_quantum_h_position(B)
qu_mom_derivs = derivatives_quantum_h_momentum()

# Measure time before simulation
tstart = datetime.now()

for iteration in range(iterations):
# Filename for output
	filename = './output/output{0}.dat'.format( str(iteration) )

	# Create unravelling and evolve
	stern_gerlach = lib.Unravelling(copy(CQstate), [L0,L1], qu_pos_derivs, qu_mom_derivs, QHamlitonian, clas_pos_derivs, clas_mom_derivs, tau, delta_time, final_time, seed, filename)
	stern_gerlach.evolution()
	del stern_gerlach

# Measure time after simulation
tfinal = datetime.now()

# Print time for loading and for averaging
tsim = tfinal - tstart
print('Simulation time is {}\n'.format(tsim))