import mylib as my
import mylib_server as my_server

#my_server.run_slurm('test.slurm', \
#	'which cmake', \
#	job_name='test_izing', \
#	time='00:02:00', \
#	nodes=1, \
#	ntasks_per_core=1, \
#	ntasks_per_node=1, \
#	cpus_per_task=1)


my_server.run_slurm('test.slurm', \
	'python run.py -mode BF_1 -Nt 1500000 -L 128 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.97 0.02 0.01 -OP_interfaces_set_IDs nvt10 -e -2.68010292 -1.34005146 -1.71526587 -OP_max_BF 0', \
	job_name='test_izing', \
	time='00:60:00', \
	nodes=1, \
	ntasks_per_core=1, \
	ntasks_per_node=1, \
	cpus_per_task=1)


