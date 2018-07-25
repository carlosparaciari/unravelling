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
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0)

# Test lindblad ops with different shape raise exception
def test_different_shape():

	CQstate = lib.CQState(np.matrix([[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.,0.],[2.,0.,1.]])

	with assert_raises(TypeError):
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0)

# Test non squared lindblad ops raise exception
def test_non_squared():

	CQstate = lib.CQState(np.matrix([[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.,0.],[0.,0.,3.]])
	L1 = np.matrix([[0.,0.,0.],[2.,0.,1.]])

	with assert_raises(TypeError):
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0)

# Test inconsistent state/operators raise exception
def test_inconsistent_state_ops():

	CQstate = lib.CQState(np.matrix([[1.],[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.],[0.,1.]])

	with assert_raises(RuntimeError):
		lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0)

# ---------------------------------- LINDBLAD OPERATORS ----------------------------------

# Test sum the Lindblad operators
def test_sum_lindblad():

	CQstate = lib.CQState(np.matrix([[1.],[1.]]),0,0)

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.],[1j,0.]])

	test_lindblad = lib.Unravelling(CQstate, [L0,L1], 0, 0, 0, 0, 0, 0, 0)

	obt1 = test_lindblad.double_lindblad_ops[0]
	obt2 = test_lindblad.double_lindblad_ops[1]
	obt3 = test_lindblad.sum_lindblad_ops

	exp1 = np.matrix([[0.,0.],[0.,1.]])
	exp2 = np.matrix([[1.,0.],[0.,0.]])
	exp3 = np.matrix([[1.,0.],[0.,1.]])

	np.testing.assert_allclose( np.asarray(obt1), np.asarray(exp1) )
	np.testing.assert_allclose( np.asarray(obt2), np.asarray(exp2) )
	np.testing.assert_allclose( np.asarray(obt3), np.asarray(exp3) )

# ---------------------------------- CONTINUOUS EVOLUTION ----------------------------------

# Test normalisation correct
def test_normalisation_continuous():

	CQstate = lib.CQState(np.matrix([[1/sqrt(2)],[1j/sqrt(2)]]),0,0)
	L = np.matrix([[0.,1.],[0.,0.]])
	delta_time = 4.
	tau = 3.

	test_norm = lib.Unravelling(CQstate, [L], 0, 0, 0, tau, delta_time, 0, 0)

	obt_norm = test_norm._normalisation_continuous()
	exp_norm = 1./3.

	np.testing.assert_almost_equal(obt_norm,exp_norm)

# ---------------------------------- SAME ELEMENT LIST ----------------------------------

# Test return False with different elements and True with same
def test_same_element_check():

	same_list = [1,1,1,1]
	diff_list = [1,1,3,1]

	obt1 = lib.list_same_elements(same_list)
	obt2 = lib.list_same_elements(diff_list)


	assert_equal( obt1, True, "Returned False" )
	assert_equal( obt2, False, "Returned True" )