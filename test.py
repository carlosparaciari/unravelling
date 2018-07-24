import lib
from math import sqrt
import numpy as np

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

# ---------------------------------- LINDBLAD OPERATORS ----------------------------------

# Test sum the Lindblad operators
def test_sum_lindblad():

	L0 = np.matrix([[0.,1.],[0.,0.]])
	L1 = np.matrix([[0.,0.],[2.,0.]])

	test_lindblad = lib.Unravelling(0, [L0,L1], 0, 0, 0, 0, 0, 0, 0)

	obt1 = test_lindblad.double_lindblad_ops[0]
	obt2 = test_lindblad.double_lindblad_ops[1]
	obt3 = test_lindblad.sum_lindblad_ops

	exp1 = np.matrix([[0.,0.],[0.,1.]])
	exp2 = np.matrix([[4.,0.],[0.,0.]])
	exp3 = np.matrix([[4.,0.],[0.,1.]])

	np.testing.assert_allclose( np.asarray(obt1), np.asarray(exp1) )
	np.testing.assert_allclose( np.asarray(obt2), np.asarray(exp2) )
	np.testing.assert_allclose( np.asarray(obt3), np.asarray(exp3) )

# ---------------------------------- CONTINUOUS EVOLUTION ----------------------------------

# Test exception raised if bad Lindblad ops or state

# Test normalisation correct