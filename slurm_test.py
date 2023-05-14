import numpy as np

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
	

MC_mode = 'swap'
script_name = 'run.py'
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
for seed in np.arange(5):
	for n in [5, 10]:
		for L in [64]:
			for phi in [0.09, 0.08, 0.07]:
				n_init = n * 2
				n_FFS = n
				phi_s = my.f2s(phi, n=5)
				
				name = 'FFS_MC%d_%d_%d_L%d_ID%d_phi%s' % (move_modes[MC_mode], n_init, n_FFS, L, seed, phi_s)
				#print(name)
				input(name)
				
				my_server.run_slurm(name + '.slurm', \
					'python %s -mode FFS_AB -L %d -to_get_timeevol 0 -N_states_FFS %d -N_init_states_FFS %d -N_runs 2 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode %s -init_composition %s %s 0.01 -OP_interfaces_set_IDs nvt16 -to_show_on_screen 0 -my_seeds %d' \
						% (script_name, L, n_FFS, n_init, MC_mode, my.f2s(0.99 - phi, n=5), my.f2s(phi, n=5), seed), \
					job_name=name, \
					time='23:59:59', \
					nodes=1, \
					ntasks_per_core=1, \
					ntasks_per_node=1, \
					cpus_per_task=1)

