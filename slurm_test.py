import numpy as np
import os

import mylib as my
import mylib_server as my_server

#import lattice_gas_save as lattice_gas
import lattice_gas
move_modes = lattice_gas.get_move_modes()

#my_server.run_slurm('test.slurm', \
#	'which cmake', \
#	job_name='test_izing', \
#	time='00:02:00', \
#	nodes=1, \
#	ntasks_per_core=1, \
#	ntasks_per_node=1, \
#	cpus_per_task=1)


#my_server.run_slurm('test.slurm', \
#	'python run.py -mode BF_1 -Nt 1500000 -L 128 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.97 0.02 0.01 -OP_interfaces_set_IDs nvt10 -e -2.68010292 -1.34005146 -1.71526587 -OP_max_BF 0', \
#	job_name='test_izing', \
#	time='00:60:00', \
#	nodes=1, \
#	ntasks_per_core=1, \
#	ntasks_per_node=1, \
#	cpus_per_task=1)
	

script_name = 'run.py'

MC_mode = 'swap'
OP_set_name = 'nvt18'
OP_set_name = 'nvt16'
#n_24h_n30 = 80-100

#MC_mode = 'long_swap'
#OP_set_name = 'nvt16'
#n_24h_n15 = 3600

# for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
	# for n in [5, 10, 15]:
		# for L in [128, 160, 200]:
			# for phi in [0.013, 0.014, 0.015]:
# for seed in np.arange(1, 101):
	# for n in [250]:
		# for L in [128]:
			# for phi in [0.015]:
# for seed in np.arange(1, 2):
	# for n in [5]:
		# for L in [128]:
			# #for phi in [0.0161]:
			# for phi in np.linspace(0.0162, 0.02, 39):
#for seed in np.arange(220, 370):
for seed in np.arange(200, 400):
	for n in [80]:
		for L in [128]:
			for phi in [0.015]:
				n_init = n * 2
				n_FFS = n
				phi_s = my.f2s(phi, n=5)
				
				name = 'FFS_MC%d_%d_%d_L%d_ID%d_phi%s_OPs%s' % (move_modes[MC_mode], n_init, n_FFS, L, seed, phi_s, OP_set_name)
				#print(name)
				#input(name)
				
				my_server.run_slurm(os.path.join('/scratch/gpfs/yp1065/Izing/slurm', name + '.slurm'), \
					'python %s -mode FFS_AB -L %d -to_get_timeevol 0 -N_states_FFS %d -N_init_states_FFS %d -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode %s -init_composition %s %s 0.01 -OP_interfaces_set_IDs %s -to_show_on_screen 0 -my_seeds %d' \
						% (script_name, L, n_FFS, n_init, MC_mode, my.f2s(0.99 - phi, n=5), my.f2s(phi, n=5), OP_set_name, seed), \
					job_name=name, \
					time='23:59:59', \
					nodes=1, \
					ntasks_per_core=1, \
					ntasks_per_node=1, \
					cpus_per_task=1, \
					slurm_output_filepath = '/scratch/gpfs/yp1065/Izing/logs/' + name + '.out')

