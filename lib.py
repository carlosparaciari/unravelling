import numpy as np
import random as rn

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
	- random_seed : the seed for the random number generator
	- filename : the name of the file where we save the data
"""
class Unravelling:

	def __init__(self, CQstate, lindblad_ops, pos_derivs, mom_derivs, Qhamiltonian, tau, delta_time, final_time, random_seed, filename):
		
		self.CQstate = CQstate
		self.lindblad_ops = lindblad_ops
		self.pos_derivs = pos_derivs
		self.mom_derivs = mom_derivs
		self.Qhamiltonian = Qhamiltonian		# This is a function
		self.tau = tau
		self.delta_time = delta_time
		self.final_time = final_time
		self.filename = filename

		# Initialise randomness
		rn.seed(random_seed)					# CAREFULL if parallelise!
		self.random_list = []

		# Check consistency of passed operators/state
		self._check_shapes()

		# Number of Lindblad operators
		self.number_lindblad = len(self.lindblad_ops)

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
	
	""" The method chooses the evolution (continuous or jump? which jump?)
		It returns the evolution and the normalisation

		- evo_type : number ( -1 for continuous, [0, N-1] for jump )
		- norm : the normalisation
	"""
	def _choose_evolution(self):

		prob_cont = self._probability_continuous()

		random_outcome = rn.random()
		self.random_list.append(random_outcome)						# Save outcome to check if good randomness

		# If the outcome is less than the prob_cont, we evolve continuously
		if random_outcome < prob_cont:
			evo_type = -1
			norm = prob_cont
			return evo_type, norm
		
		# Otherwise we jump!
		prob_jumps = [ self._probability_jump(alpha) for alpha in range(self.number_lindblad) ]
		cumulative_prob_jumps = np.cumsum(prob_jumps)
		enumerate_prob_jumps = enumerate(cumulative_prob_jumps)
		tot_prob_jumps = cumulative_prob_jumps[-1]

		random_outcome = rn.uniform(0, tot_prob_jumps)
		self.random_list.append(random_outcome/tot_prob_jumps)		# Save outcome to check if good randomness

		alpha = next(x[0] for x in enumerate_prob_jumps if x[1] > random_outcome)

		evo_type = alpha
		norm = prob_jumps[alpha]
		
		return evo_type, norm

	# The method normalises the state obtained through continuous evolution
	def _probability_continuous(self):
		
		sandwich = self.CQstate.state.H * self.sum_lindblad_ops * self.CQstate.state
		sandwich = np.asscalar(sandwich)

		normalisation = 1 - (self.delta_time/self.tau) * sandwich

		return normalisation

	# The method normalises the state obtained through jump evolution
	def _probability_jump(self, alpha):
		
		normalisation = self.CQstate.state.H * self.double_lindblad_ops[alpha] * self.CQstate.state
		normalisation = np.asscalar(normalisation)

		return normalisation

	# The method stores the randomness in a list (to check good randomness was used)
	def _record_randomness(self):
		pass

	# The method stores the trajectory in a list
	def _record_trajectory(self):
		pass

	# The method saves the data in a file
	def _save_to_file(self):
		pass

	# The method checks for consistent state/operators
	def _check_shapes(self):

		state_row, state_col = (self.CQstate.state).shape

		if state_col != 1:
			raise TypeError("The state is not passed as a dx1 matrix.")
		
		lindblad_shapes = [ L.shape for L in self.lindblad_ops ]
		same_shape_lindblad = list_same_elements(lindblad_shapes)

		if not same_shape_lindblad:
			raise TypeError("The Lindblad operators do not have the same shape.")

		lindblad_row, lindblad_col = lindblad_shapes[0]

		if lindblad_row != lindblad_col:
			raise TypeError("The Lindblad operators are not squared.")

		if lindblad_col != state_row:
			raise RuntimeError("Lindblad and state are not consistent.")

# ----------------- ADDITIONAL FUNCTIONS -----------------

""" The function checks whether all element in the list are equal.
	Return True if they are, False if they are not.
"""
def list_same_elements(items):

	target = items[0]
	Qitems = map(lambda elem: elem == target, items)
	same_elements = all(Qitems)

	return same_elements