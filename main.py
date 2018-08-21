import lib
import numpy as np
import random as rn
from math import sqrt
from copy import copy

""" The initial CQ state is:
	- state : \ket{+}, eigenstate of Pauli X with eigenvalue 1
	- position : 0, start at the centre
	- momentum : 0, start with zero momentum 
"""
CQstate = lib.CQState(np.matrix([[1/sqrt(2)],[1/sqrt(2)]]),0.,0.)

""" The dynamical configuration is
	- Lindblad :
		- L0 send \ket{0} in \ket{1}, and |w0| = 1
		- L1 send \ket{1} in \ket{0}, and |w1| = 1
	- Interaction Hamiltonian
		- H = h0 L0^dagger L0 + h1 L1^dagger L1
		- h0 = +q
		- h1 = -q
	- Classical Hamiltonian is
		- Hc = p^2/2
		- dHcdq = 0
		- dHcdp = p
"""
L0 = np.matrix([[0.,0.],[1.,0.]])
L1 = np.matrix([[0.,1.],[0.,0.]])
QHamlitonian = lambda q,p : q * np.matrix([[1.,0.],[0.,-1.]])
clas_pos_derivs = lambda q,p : 0
clas_mom_derivs = lambda q,p : p

""" The derivative of h0 and h1 is
	- dh0/dq = 1 , dh0/dp = 0
	- dh1/dq = -1 , dh1/dp = 0
"""
pos_derivs = [1., -1.]
mom_derivs = [0., 0.]

""" The time scale:
	- tau : rate of jump
	- delta_time : time interval for single evolution
	- final time : final time we are interested in
"""
tau = 1e-1
delta_time = 1e-4
final_time = 1.

# Random seed
seed = 365
rn.seed(seed)

# Number of iterations
iterations = int(1e4)

for iteration in range(iterations):
# Filename for output
	filename = './output/output{0}.dat'.format( str(iteration) )

	# Create unravelling and evolve
	stern_gerlach = lib.Unravelling(copy(CQstate), [L0,L1], pos_derivs, mom_derivs, QHamlitonian, clas_pos_derivs, clas_mom_derivs, tau, delta_time, final_time, seed, filename)
	stern_gerlach.evolution()
	del stern_gerlach
