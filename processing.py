import numpy as np

# Number of trajectories created
number_of_outputs = int(1e3)

## Defining functions that will be useful for processing trajectories

# Map amplitude into population
def ampltopop(x):
	return np.conjugate(x)*x

# Vectorise function ampltopop to be more efficient on long array
vampltopop = np.vectorize(ampltopop, otypes='f')

# Map amplitudes into coherence term
def ampltocoher(v):
	return v[0]*np.conjugate(v[1])

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
							 dtype = np.complex
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

print('Quantum trajectories loaded\n')

## Load the classical trajectories (we only store q and p, time is not necessary)

Ctrajectories = []

for iterator in range(number_of_outputs):
    
	filename = './output/output{0}.dat'.format( str(iterator) )
	Ctrajectory = np.loadtxt(filename,
							 skiprows = 4,
							 delimiter = ' , ',
							 usecols = (2,3),
							 dtype = np.float
							 )
	Ctrajectories.append(Ctrajectory)

Ctrajectories_array = np.array(Ctrajectories)

# Clean up to free RAM
Ctrajectories = None
Ctrajectory = None

print('Classical trajectories loaded\n')

## Get average trajectories in state-space

# NOTE : For this particular example we know all the possible values
q_values = [0.]
p_values = [0.1, 0., -0.1]

empty_array = np.zeros( Qpopground_array.shape, dtype=np.float )

evolution = []

for q in q_values:
	for p in p_values:
	    
		statespace_point_both = Ctrajectories_array == [q,p]
		statespace_point = np.apply_along_axis(andarray, 2, statespace_point_both)

		Qpopground_fixed_statespace = np.where(statespace_point, Qpopground_array, empty_array)
		Qpopexcited_fixed_statespace = np.where(statespace_point, Qpopexcited_array, empty_array)
		Qcoherence_fixed_statespace = np.where(statespace_point, Qcoherence_array, empty_array)

		Qpopground_fixed_statespace_average = np.sum(Qpopground_fixed_statespace,axis=0)/number_of_outputs
		Qpopexcited_fixed_statespace_average = np.sum(Qpopexcited_fixed_statespace,axis=0)/number_of_outputs
		Qcoherence_fixed_statespace_average = np.sum(Qcoherence_fixed_statespace,axis=0)/number_of_outputs

		evolution.append([Qpopground_fixed_statespace_average,
						  Qpopexcited_fixed_statespace_average,
						  Qcoherence_fixed_statespace_average,
						  q,
						  p]
						  )

# Clean up to free RAM
Ctrajectories_array = None
Qpopground_array = None
Qpopexcited_array = None
Qcoherence_array = None

print('Average trajectories computed\n')

## Save average trajectories on file
for Qpopu0, Qpopu1, Qcoher, q, p in evolution:
	# Prpare the output filename
	filename_output = './output/average_trajectory_pos_{0}_mom_{1}.dat'.format( str(q) , str(p))

	# Prepare the incipit of the datafile
	incipit = 'position , momentum\n'
	incipit += str(q) + ' , ' + str(p) + '\n\n'

	# Prepare the main body of the datafile
	body = 'population u0 , population u1 , coherence\n'
	for u0 , u1, coher in zip(Qpopu0, Qpopu1, Qcoher):
		body += str(u0) + ' , ' + str(u1) + ' , ' + str(coher) + '\n'

	text = incipit + body
	with open(filename_output, 'w') as output:
		output.write(text)

