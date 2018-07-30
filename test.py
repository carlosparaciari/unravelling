import lib
from math import sqrt
import numpy as np
from nose.tools import assert_raises, assert_equal

# ---------------------------------- LINEAR ALGEBRA ----------------------------------

# Test matrix multiplication
def test_matrix_multiplication():

	v1 = np.matrix([[1.],[0.]])
	v2 = np.matrix([[0.],[1.]])
	v3 = np.matrix([[1/sqrt(2)],[1/sqrt(2)]])
	v4 = np.matrix([[1/sqrt(2)],[-1/sqrt(2)]])

	Hadamard = 1/sqrt(2) * np.matrix([[1,1],[1,-1]])

	w1 = Hadamard * v1
	w2 = Hadamard * v2
	w3 = Hadamard * v3
	w4 = Hadamard * v4

	np.testing.assert_allclose( np.asarray(w1), np.asarray(v3) )
	np.testing.assert_allclose( np.asarray(w2), np.asarray(v4) )
	np.testing.assert_allclose( np.asarray(w3), np.asarray(v1) )
	np.testing.assert_allclose( np.asarray(w4), np.asarray(v2) ) 

# Test inner product
def test_inner_product():

	v1 = np.matrix([[1.],[0.]])
	v2 = np.matrix([[1/sqrt(2)],[-1j/sqrt(2)]])

	# computed
	n1 = v1.H * v1					# return array of one element
	n1s = np.asscalar(n1)			# return scalar
	n2s = np.asscalar(v2.H * v2)
	ip = np.asscalar(v2.H * v1)

	# expected
	e1 = np.matrix([[1.]])
	e1s = 1.
	e2s = 1.
	eip = 1/sqrt(2)

	np.testing.assert_allclose( np.asarray(n1) , np.asarray(e1) )
	np.testing.assert_allclose( np.asarray(n1s), np.asarray(e1s) )
	np.testing.assert_allclose( np.asarray(n2s), np.asarray(e2s) )
	np.testing.assert_allclose( np.asarray(ip), np.asarray(eip) )

# ---------------------------------- CONSISTENCY CHECKS ----------------------------------

# Test wrong format state raises exception - should be dx1 numpy matrix
def test_state_wrong():

	CQstate = lib.CQState(np.matrix([[1.,2.],[3.,4.]]),0,0)

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.],[2.,0.]])

	with assert_raises(TypeError):
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0, 0)

# Test lindblad ops with different shape raise exception
def test_different_shape():

	CQstate = lib.CQState(np.matrix([[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.,0.],[2.,0.,1.]])

	with assert_raises(TypeError):
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0, 0)

# Test non squared lindblad ops raise exception
def test_non_squared():

	CQstate = lib.CQState(np.matrix([[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.,0.],[0.,0.,3.]])
	L1 = np.matrix([[0.,0.,0.],[2.,0.,1.]])

	with assert_raises(TypeError):
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0, 0)

# Test inconsistent state/operators raise exception
def test_inconsistent_state_ops():

	CQstate = lib.CQState(np.matrix([[1.],[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.],[0.,1.]])

	with assert_raises(RuntimeError):
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0, 0)

# ---------------------------------- LINDBLAD OPERATORS ----------------------------------

# Test sum the Lindblad operators
def test_sum_lindblad():

	CQstate = lib.CQState(np.matrix([[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.],[1j,0.]])

	test_lindblad = lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0, 0)

	obt1 = test_lindblad.double_lindblad_ops[0]
	obt2 = test_lindblad.double_lindblad_ops[1]
	obt3 = test_lindblad.sum_lindblad_ops

	exp1 = np.matrix([[0.,0.],[0.,1.]])
	exp2 = np.matrix([[1.,0.],[0.,0.]])
	exp3 = np.matrix([[1.,0.],[0.,1.]])

	np.testing.assert_allclose( np.asarray(obt1), np.asarray(exp1) )
	np.testing.assert_allclose( np.asarray(obt2), np.asarray(exp2) )
	np.testing.assert_allclose( np.asarray(obt3), np.asarray(exp3) )

# ---------------------------------- PROBABILITIES ----------------------------------

# Test continuous probability correct
def test_probability_continuous():

	CQstate = lib.CQState(np.matrix([[1/sqrt(2)],[1j/sqrt(2)]]),0,0)
	L = np.matrix([[0.,1.],[0.,0.]])
	delta_time = 4.
	tau = 3.

	test_prob = lib.Unravelling(CQstate, [L], 0, 0, 0, tau, delta_time, 0, 0, 0)

	obt_prob = test_prob._probability_continuous()
	exp_prob = 1./3.

	np.testing.assert_almost_equal(obt_prob,exp_prob)

# Test jump probability correct
def test_probability_jump():

	CQstate = lib.CQState(np.matrix([[1/sqrt(3)],[1j*sqrt(2)/sqrt(3)]]),0,0)
	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[1j,0.],[0.,0.]])
	delta_time = 4.
	tau = 3.

	test_prob = lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, tau, delta_time, 0, 0, 0)

	obt_prob1 = test_prob._probability_jump(0)
	exp_prob1 = 2./3.
	obt_prob2 = test_prob._probability_jump(1)
	exp_prob2 = 1./3.

	np.testing.assert_almost_equal(obt_prob1,exp_prob1)
	np.testing.assert_almost_equal(obt_prob2,exp_prob2)

# ---------------------------------- CHOOSE EVOLUTION ----------------------------------

# Test choice continuous evolution is correct
def test_choose_evo_continuous():

	CQstate = lib.CQState(np.matrix([[1./sqrt(2)],[1./sqrt(2)]]),0,0)
	L0 = np.matrix([[0.,0.],[1.,0.]])
	L1 = np.matrix([[0.,1.],[0.,0.]])
	delta_time = 1.
	tau = 2.
	seed = 1

	test_evo = lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, tau, delta_time, 0, seed, 0)
	obt_type, obt_norm = test_evo._choose_evolution()
	obt_randomness = test_evo.random_list[0]

	exp_type = -1
	exp_norm = 0.5
	exp_randomness = 0.13436424411240122

	assert_equal( obt_type, exp_type, "Different type returned." )
	np.testing.assert_almost_equal( obt_norm, exp_norm )
	np.testing.assert_almost_equal( obt_randomness, exp_randomness )

# Test choice first jump L0 is correct
def test_choose_evo_jump_0():

	CQstate = lib.CQState(np.matrix([[sqrt(2)/sqrt(3)],[1./sqrt(3)]]),0,0)
	L0 = np.matrix([[0.,0.],[1.,0.]])
	L1 = np.matrix([[0.,1.],[0.,0.]])
	delta_time = 1.
	tau = 2.
	seed = 10

	test_evo = lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, tau, delta_time, 0, seed, 0)
	obt_type, obt_norm = test_evo._choose_evolution()
	obt_randomness_first = test_evo.random_list[0]
	obt_randomness_second = test_evo.random_list[1]

	exp_type = 0
	exp_norm = 2./3.
	exp_randomness_first = 0.5714025946899135
	exp_randomness_second = 0.4288890546751146

	assert_equal( obt_type, exp_type, "Different type returned." )
	np.testing.assert_almost_equal( obt_norm, exp_norm )
	np.testing.assert_almost_equal( obt_randomness_first, exp_randomness_first )
	np.testing.assert_almost_equal( obt_randomness_second, exp_randomness_second )

# Test choice first jump L1 is correct
def test_choose_evo_jump_1():
	
	CQstate = lib.CQState(np.matrix([[sqrt(2)/sqrt(3)],[1./sqrt(3)]]),0,0)
	L0 = np.matrix([[0.,0.],[1.,0.]])
	L1 = np.matrix([[0.,1.],[0.,0.]])
	delta_time = 1.
	tau = 2.
	seed = 2

	test_evo = lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, tau, delta_time, 0, seed, 0)
	obt_type, obt_norm = test_evo._choose_evolution()
	obt_randomness_first = test_evo.random_list[0]
	obt_randomness_second = test_evo.random_list[1]

	exp_type = 1
	exp_norm = 1./3.
	exp_randomness_first = 0.9560342718892494
	exp_randomness_second = 0.9478274870593494

	assert_equal( obt_type, exp_type, "Different type returned." )
	np.testing.assert_almost_equal( obt_norm, exp_norm )
	np.testing.assert_almost_equal( obt_randomness_first, exp_randomness_first )
	np.testing.assert_almost_equal( obt_randomness_second, exp_randomness_second )

# --------------------------------- CONTINUOUS EVOLUTION --------------------------------

# Test continuous evolution with only diagonal Hamiltonian
def test_cont_evo_diagonal_Hamiltonian_only():
	
	CQstate = lib.CQState(np.matrix([[sqrt(2)/sqrt(3)],[1j*1./sqrt(3)]]),1./4.,0)
	L0 = np.matrix([[0.,0.],[0.,0.]])		# set L0 to be zero everywhere so we can check contribution of H only
	delta_time = 2.
	tau = 8.

	QHamlitonian = lambda q,p : q * np.matrix([[1.,0.],[0.,-1.]])

	test_cont = lib.Unravelling(CQstate, [L0], 0, 0, QHamlitonian, tau, delta_time, 0, 0, 0)
	test_cont._evolution_one_step(-1, 1.)

	obt_state = test_cont.CQstate.state
	obt_position = test_cont.CQstate.position
	obt_momentum = test_cont.CQstate.momentum
	obt_time = test_cont.CQstate.time

	exp_state = np.matrix([[sqrt(2)/sqrt(3) - 1j * 1./sqrt(6)]
						  ,[-1./sqrt(12) + 1j*1./sqrt(3)]])
	exp_position = 1./4.
	exp_momentum = 0
	exp_time = 2.

	np.testing.assert_almost_equal( obt_state, exp_state )
	np.testing.assert_almost_equal( obt_position, exp_position )
	np.testing.assert_almost_equal( obt_momentum, exp_momentum )
	np.testing.assert_almost_equal( obt_time, exp_time )

# Test continuous evolution with only non-diagonal Hamiltonian
def test_cont_evo_non_diagonal_Hamiltonian_only():
	
	CQstate = lib.CQState(np.matrix([[sqrt(2)/sqrt(3)],[1j*1./sqrt(3)]]),1./2.,0)
	L0 = np.matrix([[0.,0.],[0.,0.]])		# set L0 to be zero everywhere so we can check contribution of H only
	delta_time = 2.
	tau = 8.

	QHamlitonian = lambda q,p : q * np.matrix([[0.,1.],[1.,0.]])

	test_cont = lib.Unravelling(CQstate, [L0], 0, 0, QHamlitonian, tau, delta_time, 0, 0, 0)
	test_cont._evolution_one_step(-1, 1.)

	obt_state = test_cont.CQstate.state
	obt_position = test_cont.CQstate.position
	obt_momentum = test_cont.CQstate.momentum
	obt_time = test_cont.CQstate.time

	exp_state = np.matrix([[(1. + sqrt(2))/sqrt(3)]
						  ,[1j * (1. - sqrt(2))/sqrt(3)]])
	exp_position = 1./2.
	exp_momentum = 0
	exp_time = 2.

	np.testing.assert_almost_equal( obt_state, exp_state )
	np.testing.assert_almost_equal( obt_position, exp_position )
	np.testing.assert_almost_equal( obt_momentum, exp_momentum )
	np.testing.assert_almost_equal( obt_time, exp_time )

# Test continuous evolution with only Lindblad operators
def test_cont_evo_Lindblad_only():
	
	CQstate = lib.CQState(np.matrix([[sqrt(2)/sqrt(3)],[1j*1./sqrt(3)]]),0,0)
	L0 = np.matrix([[0.,0.],[1.,0.]])
	L1 = np.matrix([[0.,1.],[0.,0.]])
	delta_time = 2.
	tau = 8.

	QHamlitonian = lambda q,p : q * np.matrix([[0.,1.],[1.,0.]])

	test_cont = lib.Unravelling(CQstate, [L0,L1], 0, 0, QHamlitonian, tau, delta_time, 0, 0, 0)
	test_cont._evolution_one_step(-1, 1.)

	obt_state = test_cont.CQstate.state
	obt_position = test_cont.CQstate.position
	obt_momentum = test_cont.CQstate.momentum
	obt_time = test_cont.CQstate.time

	exp_state = 7./8. * np.matrix([[sqrt(2)/sqrt(3)],[1j*1./sqrt(3)]])
	exp_position = 0
	exp_momentum = 0
	exp_time = 2.

	np.testing.assert_almost_equal( obt_state, exp_state )
	np.testing.assert_almost_equal( obt_position, exp_position )
	np.testing.assert_almost_equal( obt_momentum, exp_momentum )
	np.testing.assert_almost_equal( obt_time, exp_time )


# Test continuous evolution complete
def test_cont_evo_complete():
	
	CQstate = lib.CQState(np.matrix([[sqrt(2)/sqrt(3)],[1j*1./sqrt(3)]]),1./4.,0)
	L0 = np.matrix([[0.,0.],[1.,0.]])
	L1 = np.matrix([[0.,1.],[0.,0.]])
	delta_time = 2.
	tau = 4.

	QHamlitonian = lambda q,p : q * np.matrix([[0.,1.],[1.,0.]])

	test_cont = lib.Unravelling(CQstate, [L0,L1], 0, 0, QHamlitonian, tau, delta_time, 0, 0, 0)
	test_cont._evolution_one_step(-1, 1.)

	obt_state = test_cont.CQstate.state
	obt_position = test_cont.CQstate.position
	obt_momentum = test_cont.CQstate.momentum
	obt_time = test_cont.CQstate.time

	exp_state = np.matrix([[(sqrt(3)/sqrt(2) + 1./sqrt(3)) / 2.]
						  ,[1j * (sqrt(3)/4. - 1/sqrt(6)) ]])
	exp_position = 1/4.
	exp_momentum = 0
	exp_time = 2.

	np.testing.assert_almost_equal( obt_state, exp_state )
	np.testing.assert_almost_equal( obt_position, exp_position )
	np.testing.assert_almost_equal( obt_momentum, exp_momentum )
	np.testing.assert_almost_equal( obt_time, exp_time )

# ---------------------------------- SAME ELEMENT LIST ----------------------------------

# Test return False with different elements and True with same
def test_same_element_check():

	same_list = [1,1,1,1]
	diff_list = [1,1,3,1]

	obt1 = lib.list_same_elements(same_list)
	obt2 = lib.list_same_elements(diff_list)


	assert_equal( obt1, True, "Returned False" )
	assert_equal( obt2, False, "Returned True" )