import numpy as np
from datetime import datetime

# Measure initial time
tstart = datetime.now()

# Modify this before processing the raw data

number_of_outputs = int(1e5)	# Number of trajectories generated
max_q_range = int(1e3)			# Number of possible steps in position
q_interval = 1e-6				# Fundamental shift in position
max_p_range = 1					# Number of possible steps in momentum
p_interval = 1e-2				# Fundamental shift in momentum

precision = 1e-7				# Precision for binning data in phase space grid (CAREFUL WITH THIS!)

## Defining functions that will be useful for processing trajectories

# Map amplitude into population
def ampltopop(x):
	return np.real(np.conjugate(x)*x)

# Vectorise function ampltopop to be more efficient on long array
vampltopop = np.vectorize(ampltopop, otypes='f')

# Map amplitudes into coherence term (we take the abs value to reduce the size of this thing)
def ampltocoher(v):
	return np.absolute(v[0]*np.conjugate(v[1]))

# And function on two-value array (probably it exists already)
def andarray(v):
	return v[0] and v[1]

## Load the quantum trajectories and find populations and coherence

Qpopground = []
Qpopexcited = []
Qcoherence = []

for iterator in range(number_of_outputs):

	filename = './output/output{0}.dat'.format( str(iterator) )
	Qtrajectory = np.loadtxt(filename,
							 skiprows = 4,
							 delimiter = ' , ',
							 usecols = (0,1),
							 dtype = np.complex128    # NOTE: I prefer np.complex64, but I get weird error if I use that!
							 )

	Qtrajectory_populations = vampltopop(Qtrajectory)
	Qtrajectory_coherence = np.apply_along_axis(ampltocoher, 1, Qtrajectory)

	Qpopground.append(Qtrajectory_populations[:,0])
	Qpopexcited.append(Qtrajectory_populations[:,1])
	Qcoherence.append(Qtrajectory_coherence)
    
Qpopground_array = np.array(Qpopground)
Qpopexcited_array = np.array(Qpopexcited)
Qcoherence_array = np.array(Qcoherence)

# Clean up to free RAM
Qpopground = None
Qpopexcited = None
Qcoherence = None
Qtrajectory_populations = None
Qtrajectory_coherence = None
Qtrajectory = None

## Load the classical trajectories (we only store q and p, time is not necessary)

Ctrajectories = []

for iterator in range(number_of_outputs):
    
	filename = './output/output{0}.dat'.format( str(iterator) )
	Ctrajectory = np.loadtxt(filename,
							 skiprows = 4,
							 delimiter = ' , ',
							 usecols = (2,3),
							 dtype = np.float32
							 )
	Ctrajectories.append(Ctrajectory)

Ctrajectories_array = np.array(Ctrajectories)

# Clean up to free RAM
Ctrajectories = None
Ctrajectory = None

# Measure intermediate time
tinter = datetime.now()

## Get average trajectories in state-space

empty_array = np.zeros( Qpopground_array.shape, dtype=np.float )

evolution = []

q_grid = range(-max_q_range, max_q_range)
q_values = [q * q_interval for q in q_grid]
p_grid = range(-max_p_range, max_p_range)
p_values = [p * p_interval for p in p_grid]

for p in p_values:
	for q in q_values:

		statespace_point_both = np.absolute( Ctrajectories_array - [q,p] ) < precision
		statespace_point = np.apply_along_axis(andarray, 2, statespace_point_both)

		Qpopground_fixed_statespace = np.where(statespace_point, Qpopground_array, empty_array)
		Qpopexcited_fixed_statespace = np.where(statespace_point, Qpopexcited_array, empty_array)
		Qcoherence_fixed_statespace = np.where(statespace_point, Qcoherence_array, empty_array)

		Qpopground_fixed_statespace_average = np.sum(Qpopground_fixed_statespace,axis=0)/number_of_outputs
		Qpopexcited_fixed_statespace_average = np.sum(Qpopexcited_fixed_statespace,axis=0)/number_of_outputs
		Qcoherence_fixed_statespace_average = np.sum(Qcoherence_fixed_statespace,axis=0)/number_of_outputs

		# NOTICE: Float format needs to be modified according to variables q_interval and p_interval
		filename_output = './output_average/average_trajectory_pos_{0:.6f}_mom_{1:.2f}.dat'.format( q, p )

		np.savetxt(filename_output,
                   np.c_[Qpopground_fixed_statespace_average,
                         Qpopexcited_fixed_statespace_average,
                         Qcoherence_fixed_statespace_average],
                   fmt='%.12f',
                   delimiter=' , ',
                   header='population u0 , population u1 , coherence')
        
# Measure final time
tfinal = datetime.now()

## Print time for loading and for averaging
tloading = tinter - tstart
taveraging = tfinal - tinter

print('Loading time was {}\n'.format(tloading))
print('Averaging time was {}\n'.format(taveraging))
