import numpy as np
import random as rn
from math import sqrt

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

	def __str__(self):

		flat_list = np.array(self.state).reshape(-1,).tolist()
		flat_list_str = map( str, flat_list)
		state_str = ' , '.join( flat_list_str )

		string = state_str + ' , ' + str(self.position) + ' , ' + str(self.momentum) + ' , ' + str(self.time)

		return string

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

	def __init__(self, CQstate, lindblad_ops, pos_derivs, mom_derivs, Qhamiltonian, clas_pos_derivs, clas_mom_derivs, tau, delta_time, final_time, random_seed, filename):
		
		self.CQstate = CQstate
		
		# Quantum part of evolution
		self.Qhamiltonian = Qhamiltonian		# This is a function (in q and p)
		self.lindblad_ops = lindblad_ops
		self.pos_derivs = pos_derivs			# NOTE : These might be functions in p and q! To be implemented!
		self.mom_derivs = mom_derivs

		# Classical part of evolution
		self.clas_pos_derivs = clas_pos_derivs
		self.clas_mom_derivs = clas_mom_derivs
		
		# Time scales
		self.tau = tau
		self.delta_time = delta_time
		self.final_time = final_time

		# Additional variables
		self.filename = filename
		self.random_seed = random_seed			# Pass this as information but nothing else (do the seeding before creating instance of Unravelling)

		# Initialise randomness
		self.random_list = []

		# Initialise trajectory record
		self.trajectory = [ str( self.CQstate ) ]

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
		
		while self.CQstate.time < self.final_time:

			evo_type, norm = self._choose_evolution()
			self._evolution_one_step(evo_type, norm)

		self._save_to_file()

	""" The method evolves the state by one time step delta_time
		Take as input
		- evo_type : the type of evolution (continuous or jump)
		- norm : the norm of the state
	"""
	def _evolution_one_step(self, evo_type, norm):

		if evo_type == -1:										# Continuous evolution
			Qham_matrix = self.Qhamiltonian( self.CQstate.position , self.CQstate.momentum )
			dHcdp = self.clas_mom_derivs( self.CQstate.position , self.CQstate.momentum )
			dHcdq = self.clas_pos_derivs( self.CQstate.position , self.CQstate.momentum )

			difference_state = self.delta_time * ( 1./(2 * self.tau) * self.sum_lindblad_ops + 1j * Qham_matrix ) * self.CQstate.state
			unnorm_state = self.CQstate.state - difference_state
			self.CQstate.state = unnorm_state/sqrt(norm)		# Here we are introducing an error prop to delta_time (due to normalisation)

			self.CQstate.position += dHcdp * self.delta_time
			self.CQstate.momentum -= dHcdq * self.delta_time			
		else:													# Jump evolution
			L = self.lindblad_ops[evo_type]
			dhdp = self.mom_derivs[evo_type]
			dhdq = self.pos_derivs[evo_type]

			self.CQstate.state = ( L * self.CQstate.state)/sqrt(norm)
			self.CQstate.position += dhdp * self.tau
			self.CQstate.momentum -= dhdq * self.tau

		self.CQstate.time += self.delta_time

		self.trajectory.append( str( self.CQstate ) )				# Save new point in trajectory evolution

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

		probability = 1 - (self.delta_time/self.tau) * sandwich

		return np.real(probability)

	# The method normalises the state obtained through jump evolution
	def _probability_jump(self, alpha):
		
		probability = self.CQstate.state.H * self.double_lindblad_ops[alpha] * self.CQstate.state
		probability = np.asscalar(probability)

		return np.real(probability)

	# The method saves the data in a file
	def _save_to_file(self):

		# Prepare the incipit of the file
		incipit = 'Lindblad operators = '

		for L in self.lindblad_ops:
			incipit += str(L.tolist()) + ' ; '

		incipit += 'dh/dq = ' + str(self.pos_derivs) + ' ; '
		incipit += 'dh/dp = ' + str(self.mom_derivs) + ' ; '
		
		incipit += ('Jump rate = {tau} ; '
					'Time unit = {del_time} ; '
					'Final_time = {fin_time} ; '
					'Random seed = {seed}\n\n').format(tau=self.tau,
												   	   del_time=self.delta_time,
												   	   fin_time=self.final_time,
												   	   seed=self.random_seed)

		# Prepare the trajectory file
		trajectory_record = 'Trajectory record\nstate e1 , state e2 , position , momentum , time\n'
		trajectory_record += '\n'.join(self.trajectory)

		# Merge incipit and trajectory
		text = incipit + trajectory_record

		# Write on file
		with open(self.filename, 'w') as output:
			output.write(text)

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

		Qham = self.Qhamiltonian(1,1)

		ham_row, ham_col = Qham.shape

		if ham_row != ham_col:
			raise TypeError("The interaction Hamiltonian is not squared.")

		if ham_col != state_row:
			raise RuntimeError("Interaction Hamiltonian and state are not consistent.")

		Qham_is_self_adjoint = np.all( Qham == Qham.H )

		if not Qham_is_self_adjoint:
			raise TypeError("Interaction Hamiltonian is not self-adjoint.")

# ----------------- ADDITIONAL FUNCTIONS -----------------

""" The function checks whether all element in the list are equal.
	Return True if they are, False if they are not.
"""
def list_same_elements(items):

	target = items[0]
	Qitems = map(lambda elem: elem == target, items)
	same_elements = all(Qitems)

	return same_elements