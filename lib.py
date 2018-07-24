import numpy as np

""" The CQ state of the theory.
	Composed by
	- pure quantum state : dx1 np.matrix with complex values
	- position : real value
	- momentum : real value
	- time : real value
"""
class CQState:

	def __init__(self, state, position, momentum):
		
		self.state = state
		self.position = position
		self.momentum = momentum
		self.time = 0

""" The algorithm for the unravelled CQ dynamics.

	Variables

	- CQstate : the quantum-classical state
	- lindblad_ops : the list of Lindblad operators L_{\alpha}
	- pos_derivs : the list of dh^{\alpha}/dq
	- mom_derivs : the list of dh^{\alpha}/dp
	- Qhamiltonian : the coupling Hamiltonian (this is a function in q and p)
	- tau : the rate of jumping
	- delta_time : the unit time of evolution
	- final_time : final time of the evolution
	- filename : the name of the file where we save the data
"""
class Unravelling:

	def __init__(self, CQstate, lindblad_ops, pos_derivs, mom_derivs, Qhamiltonian, tau, delta_time, final_time, filename):
		
		self.CQstate = CQstate
		self.lindblad_ops = lindblad_ops
		self.pos_derivs = pos_derivs
		self.mom_derivs = mom_derivs
		self.Qhamiltonian = Qhamiltonian		# This is a function
		self.tau = tau
		self.delta_time = delta_time
		self.final_time = final_time
		self.filename = filename

		# Compute the list of L_{\alpha}^{\dagger} L_{\alpha}
		self.double_lindblad_ops = [ (L.H)*L for L in self.lindblad_ops ]

		# Compute the sum of L_{\alpha}^{\dagger} L_{\alpha}
		self.sum_lindblad_ops = sum(self.double_lindblad_ops)

	# The method evolves the CQ state until final time
	def evolution(self):
		pass

	# The method evolves the state by one time step delta_time
	def _evolution_one_step(self):
		pass
	
	# The method chooses the evolution (continuous or jump, which jump)
	def _chose_evolution(self):
		pass

	# The method normalises the state obtained through continuous evolution
	def _normalisation_continuous(self):
		
		sandwich = self.CQstate.state.H * self.sum_lindblad_ops * self.CQstate.state

		if sandwich.size != 1:
			raise ValueError("The sandwich should return a scalar.")

		sandwich = np.asscalar(sandwich)
		normalisation = 1 - (self.delta_time/self.tau) * sandwich

		return normalisation

	# The method computes the probability of continuous evolution
	def _probability_continuous(self):
		pass

	# The method normalises the state obtained through jump evolution
	def _normalisation_jump(self, alpha):
		pass

	# The method computes the probability of jump evolution
	def _probability_jump(self, alpha):
		pass

	# The method stores the randomness in a list (to check good randomness was used)
	def _record_randomness(self):
		pass

	# The method stores the trajectory in a list
	def _record_trajectory(self):
		pass

	# The method saves the data in a file
	def _save_to_file(self):
		pass