import numpy as np
from datetime import datetime

# Modify this before processing the raw data

number_of_outputs = 100			# Number of trajectories generated
precision = 1e-7				# Precision for binning data in phase space grid (CAREFUL WITH THIS!)

# ------------------------------------------- USEFUL FUNCTIONS -------------------------------------------

# Map amplitude into population
def ampltopop(x):
	return np.real(np.conjugate(x)*x)

# Vectorise function ampltopop to be more efficient on long array
vampltopop = np.vectorize(ampltopop, otypes='f')

# Map amplitudes into coherence term (we take the abs value to reduce the size of this thing)
def ampltocoher(v):
	return np.absolute(v[0]*np.conjugate(v[1]))

# ------------------------------------------- LOADING THE DATA -------------------------------------------

Qpopground = []
Qpopexcited = []
Qcoherence = []

# Measure time before loading
t0 = datetime.now()

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

Qpopground_array = np.transpose(Qpopground)
Qpopexcited_array = np.transpose(Qpopexcited)
Qcoherence_array = np.transpose(Qcoherence)

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

Ctrajectories_array = np.transpose(Ctrajectories,axes=[1,0,2])

# Clean up to free RAM
Ctrajectories = None
Ctrajectory = None

# Measure time after loading
t1 = datetime.now()

# Print loading time
tloading = t1 - t0
print('Loading time was {}\n'.format(tloading))

# ------------------------------------------- BROADCASTING AND AVERAGING -------------------------------------------

# Measure time before broadcasting and averaging
t0 = datetime.now()

# Counts the number of timesteps and the number of trajectories we use
Ntimesteps , Ntrajectories , _ = Ctrajectories_array.shape

## Create phase space points of interest
q_values = np.unique(Ctrajectories_array[:,:,0])
q_value_num = len(q_values)
p_values = np.unique(Ctrajectories_array[:,:,1])
p_value_num = len(p_values)

# This vector contains all points we need to check on the phasespace
phasespace = np.stack((np.tile(q_values, p_value_num), np.repeat(p_values, q_value_num)), axis=-1)

# Clean up to free RAM
q_values = None
p_values = None

# Total number of different phase space points
Npspoints , _ = phasespace.shape

# Broadcast both phasespace and classical trajectories to check equality only once
broad_phasespace = np.transpose(np.broadcast_to(phasespace,(Ntimesteps,Ntrajectories,Npspoints, 2)),(2,0,1,3))
broad_Ctrajectories = np.broadcast_to(Ctrajectories_array,(Npspoints, Ntimesteps, Ntrajectories, 2))

# Find points that are the same
broad_phasespace_bool_both = np.absolute( broad_Ctrajectories - broad_phasespace ) < precision
broad_phasespace_bool = np.all(broad_phasespace_bool_both,axis=3)

# Clean up to free RAM
broad_Ctrajectories = None
broad_phasespace = None
broad_phasespace_bool_both = None

# Broadcast both populations and coherence
broad_Qpopground = np.broadcast_to(Qpopground_array,(Npspoints, Ntimesteps, Ntrajectories))
broad_Qpopexcited = np.broadcast_to(Qpopexcited_array,(Npspoints, Ntimesteps, Ntrajectories))
broad_Qcoherence = np.broadcast_to(Qcoherence_array,(Npspoints, Ntimesteps, Ntrajectories))

# Create zero vector that will be used later
empty_array = np.zeros( broad_Qpopground.shape, dtype=np.float32 )

# Single out the points with same position and momentum
Qpopground_fixed_phasespace = np.where(broad_phasespace_bool, broad_Qpopground, empty_array)
Qpopexcited_fixed_phasespace = np.where(broad_phasespace_bool, broad_Qpopexcited, empty_array)
Qcoherence_fixed_phasespace = np.where(broad_phasespace_bool, broad_Qcoherence, empty_array)

# Clean up to free RAM
broad_Qpopground = None
broad_Qpopexcited = None
broad_Qcoherence = None

# Now we average along the trajectories
Qpopground_fixed_phasespace_average = np.sum(Qpopground_fixed_phasespace,axis=2)/number_of_outputs
Qpopexcited_fixed_phasespace_average = np.sum(Qpopexcited_fixed_phasespace,axis=2)/number_of_outputs
Qcoherence_fixed_phasespace_average = np.sum(Qcoherence_fixed_phasespace,axis=2)/number_of_outputs

# Clean up to free RAM
Qpopground_fixed_phasespace = None
Qpopexcited_fixed_phasespace = None
Qcoherence_fixed_phasespace = None

# Measure time after broadcasting and averaging
t1 = datetime.now()

# Print broadcasting and averaging time
tbroad = t1 - t0
print('Broadcasting and averaging time was {}\n'.format(tbroad))

# ------------------------------------------- SAVING THE AVERAGE -------------------------------------------

# Measure time before saving data
t0 = datetime.now()

for iterator in range(Npspoints):

	# NOTICE: Float format needs to be modified according to variables q_interval and p_interval
	filename_output = './output_average/average_trajectory_pos_{0:.6f}_mom_{1:.2f}.dat'.format( *phasespace[iterator] )

	np.savetxt(filename_output,
			   np.c_[Qpopground_fixed_phasespace_average[iterator],
			   		 Qpopexcited_fixed_phasespace_average[iterator],
			   		 Qcoherence_fixed_phasespace_average[iterator]],
			   		 fmt='%.12f',
			   		 delimiter=' , ',
			   		 header='population u0 , population u1 , coherence')

# Measure time after saving data
t1 = datetime.now()

# Print saving time
tsave = t1 - t0
print('Saving time was {}\n'.format(tsave))