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

	np.testing.assert_allclose(w1,v3)
	np.testing.assert_allclose(w2,v4)
	np.testing.assert_allclose(w3,v1)
	np.testing.assert_allclose(w4,v2)

# Test inner product
def test_inner_product():

	v1 = np.matrix([[1.],[0.]])
	v2 = np.matrix([[1/sqrt(2)],[-1j/sqrt(2)]])

	# computed
	n1 = v1.H * v1					# return matrix of one element
	n1s = np.asscalar(n1)			# return scalar
	n2s = np.asscalar(v2.H * v2)
	ip = np.asscalar(v2.H * v1)

	# expected
	e1 = np.matrix([[1.]])
	e1s = 1.
	e2s = 1.
	eip = 1/sqrt(2)

	np.testing.assert_allclose(n1,e1)
	np.testing.assert_allclose(n1s,e1s)
	np.testing.assert_allclose(n2s,e2s)
	np.testing.assert_allclose(ip,eip)