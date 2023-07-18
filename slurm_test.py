import numpy as np
import os
import argparse

import mylib as my
import mylib_server as my_server
import table_data
import izing

#import lattice_gas_save as lattice_gas
import lattice_gas

def main():
	# python slurm_test.py --MC_mode long_swap --OP_set_name nvt --mode launch_run
	# python slurm_test.py --MC_mode swap --OP_set_name nvt --mode launch_run
	
	# python slurm_test.py --mode launch_run --MC_mode swap --OP_set_name nvt --seed_s -1 1135 1180 --n_s 33 --L_s 128 --phi1_s 0.015 --Temp_s 1.0 --slurm_time 72:00:00 --to_run 0
	# python slurm_test.py --mode launch_run --MC_mode swap --OP_set_name nvt --seed_s -1 1160 1175 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 0.015 0.0155 0.016 --Temp_s 0.8 0.9 1.0 --slurm_time 24:00:00 --to_run 0
	
	# python slurm_test.py --mode launch_run --MC_mode long_swap --OP_set_name nvt --seed_s -1 1010 1033 --n_s -1 --L_s 128 --phi1_s 0.0145 --Temp_s 1.0 --to_run 0
	# python slurm_test.py --mode launch_run --MC_mode long_swap --OP_set_name nvt --seed_s -1 1010 1020 --n_s -1 --L_s 128 --phi1_s 0.015 --Temp_s 1.0 --to_run 0
	
	# == running
	## == probably running, review that times are close to full times
	### == testing
	
	# python slurm_test.py --mode launch_run --MC_mode swap --OP_set_name nvt --seed_s -1 1000 1030 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 --Temp_s 0.95 --slurm_time 72:00:00 --to_run 0
	# python slurm_test.py --mode launch_run --MC_mode swap --OP_set_name nvt --seed_s -1 1020 1030 --n_s -1 --L_s 128 --phi1_s 0.015 0.0155 0.016 --Temp_s 0.95 --slurm_time 72:00:00 --to_run 0
	# python slurm_test.py --mode launch_run --MC_mode swap --OP_set_name nvt --seed_s -1 1010 1030 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 0.015 0.0155 0.016 --Temp_s 0.85 --slurm_time 72:00:00 --to_run 0
	# python slurm_test.py --mode launch_run --MC_mode swap --OP_set_name nvt --seed_s -1 1010 1030 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 --Temp_s 1.0 --slurm_time 144:00:00 --to_run 0
	
	# python slurm_test.py --mode launch_run --MC_mode long_swap --OP_set_name nvt --seed_s -1 1030 1060 --n_s -1 --L_s 128 --phi1_s 0.014 --Temp_s 1.0 --slurm_time 72:00:00 --RAM_per_CPU 6G --to_run 0
	# python slurm_test.py --mode launch_run --MC_mode long_swap --OP_set_name nvt --seed_s -1 1015 1030 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 0.015 0.0155 0.016 --Temp_s 0.95 0.85 --to_run 0
	
	# python slurm_test.py --mode launch_run --MC_mode swap --OP_set_name nvt --seed_s -1 1000 1006 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 0.015 0.0155 0.016 --Temp_s 1.0 0.95 0.9 0.85 --to_run 0 -RAM_per_CPU 30
	# python slurm_test.py --mode launch_run --MC_mode long_swap --OP_set_name nvt --seed_s -1 1000 1006 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 0.015 0.0155 0.016 --Temp_s 1.0 0.95 0.9 0.85 --to_run 0 -RAM_per_CPU 30
	
	# python slurm_test.py --mode search_logs --MC_mode swap --OP_set_name nvt --seed_s -1 1000 1200 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 0.015 0.0155 0.016 --Temp_s 0.8 0.9 1.0
	# python slurm_test.py --mode search_npzs --MC_mode swap --OP_set_name nvt --seed_s -1 1000 1200 --n_s -1 --L_s 128 --phi1_s 0.014 0.0145 0.015 0.0155 0.016 --Temp_s 0.8 0.9 1.0
	
	# python slurm_test.py --mode launch_run --MC_mode long_swap --OP_set_name nvt --seed_s -1 1000 1020 --n_s -1 --L_s 128 --phi1_s 0.014 --Temp_s 0.95 --to_run 0
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--mode', help='what to do', choices=['launch_run', 'search_logs', 'search_npzs'], required=True)
	
	parser.add_argument('--MC_mode', help='mode of MC moves', choices=['swap', 'long_swap', 'flip'], required=True)
	parser.add_argument('--OP_set_name', help='string-ID of the set of OP-intarfaces', choices=table_data.OP_interfaces_table.keys(), required=True)
	parser.add_argument('--slurm_time', help='mode of MC moves', default='24:00:00')
	parser.add_argument('--RAM_per_CPU', help='RAM for the run', default='4G')
	
	parser.add_argument('--phi_1', type=int, help='phi_1 value', default=0.015)
	parser.add_argument('--phi_2', type=int, help='phi_2 value', default=0.01)
	parser.add_argument('--Temp', type=float, help='scale e', default=1.0)
	parser.add_argument('--e11', type=float, help='e11', default=-2.68010292)
	parser.add_argument('--e12', type=float, help='e12', default=-1.34005146)
	parser.add_argument('--e22', type=float, help='e22', default=-1.71526587)
	
	parser.add_argument('--seed_s', type=int, help='seeds for cycle', nargs='+')
	parser.add_argument('--n_s', type=int, help='n-s for cycle', nargs='*', default=-1)
	parser.add_argument('--Dtop_Nruns_s', type=int, help='Nruns for Dtop estimation', nargs='*', default=-1)
	parser.add_argument('--L_s', type=int, help='L-s for cycle', nargs='+')
	parser.add_argument('--phi1_s', type=float, help='phi1-s for cycle', nargs='+')
	parser.add_argument('--Temp_s', type=float, help='Temp-s for cycle', nargs='*', default=1.0)
	
	parser.add_argument('--phi_dgt', type=int, help='system name', default=6)
	
	parser.add_argument('--to_run', type=int, help='whether to run or just print commands', default=0)
	
	clargs = parser.parse_args()
	
	move_modes = lattice_gas.get_move_modes()
	
	seed_s = np.arange(clargs.seed_s[1], clargs.seed_s[2]) if(clargs.seed_s[0] == -1) else clargs.seed_s
	mode = clargs.mode
	
	to_run = clargs.to_run
	script_name = 'run.py'
	script_name = 'run_tmp.py'
	
	phi_dgt = clargs.phi_dgt
	phi_2 = clargs.phi_2
	e_T1 = np.array([clargs.e11, clargs.e12, clargs.e22]) / clargs.Temp
	
	MC_mode = clargs.MC_mode
	OP_set_name = clargs.OP_set_name
	
	if(mode == 'launch_run'):
		low_RAW_choice = None
		low_Nuse_choice = None
		for seed in seed_s:
			for n in clargs.n_s:
				for Dtop_Nruns in clargs.Dtop_Nruns_s:
					for L in clargs.L_s:
						for phi in clargs.phi1_s:
							for Temp in clargs.Temp_s:
								if(n >= 0):
									n_use = n
								elif(n == -1):
									#n_use = int(table_data.n144h_FFS_dict[MC_mode][my.f2s(Temp, n=2)][my.f2s(phi, n=5)] * my_server.slurm_time_to_seconds(clargs.slurm_time)/(144*3600) + 0.5)
									n_use = table_data.nPlot_FFS_dict[MC_mode][my.f2s(Temp, n=2)][my.f2s(phi, n=5)]
								
								if(Dtop_Nruns >= 0):
									Dtop_Nruns_use = Dtop_Nruns
								else:
									Dtop_Nruns_use = int(table_data.n144h_Dtop_FFS_dict[MC_mode][my.f2s(Temp, n=2)][my.f2s(phi, n=5)] * my_server.slurm_time_to_seconds(clargs.slurm_time)/(144*3600) + 0.5)*10
								
								if(n_use > 0):
									if((n_use < 15) and (low_Nuse_choice != 'ALL')):
										low_Nuse_choice = input('n = %d < 15 - too small, poor statistics possible.\nProceed? [ALL/ok/CtrlC]\n' % n_use)
									RAM_in_GB = my_server.slurm_RAW_to_bytes(clargs.RAM_per_CPU)/2**30
									RAM_thr = 4 * n_use / 300   # for FFS simulations
									#RAM_thr = 4 * n_use / 300   # for FFS simulations
									to_post_proc = (RAM_in_GB >= RAM_thr)
									if((not to_post_proc) and (low_RAW_choice != 'ALL')):
										low_RAW_choice = input('RAM = %s G < 4G * (%d/300) - too small, post-proc will not be done.\nProceed? [ALL/ok/CtrlC]\n' % (my.f2s(RAM_in_GB), n_use))
									
									n_init = n_use * 2
									n_FFS = n_use
									phi_s = my.f2s(phi, n=phi_dgt)
									
									#name = 'FFS_MC%d_%d_%d_L%d_Temp%s_ID%d_phi%s_OPs%s' % (move_modes[MC_mode], n_init, n_FFS, L, my.f2s(Temp), seed, phi_s, OP_set_name)
									name = 'FFS_MC%d_%d_%d_L%d_Temp%s_ID%d_phi%s_OPs%s_DtopRuns%d' % (move_modes[MC_mode], n_init, n_FFS, L, my.f2s(Temp), seed, phi_s, OP_set_name, Dtop_Nruns_use)
									#print(name)
									#input(name)
									
									# cmd = 'python %s -mode FFS_AB -L %d -to_get_timeevol 0 -N_states_FFS %d -N_init_states_FFS %d -e %lf %lf %lf -MC_move_mode %s -init_composition %s %s %s -OP_interfaces_set_IDs %s -to_show_on_screen 0 -my_seeds %d -to_post_proc %s' \
											# % (script_name, L, n_FFS, n_init, e_T1[0]/Temp, e_T1[1]/Temp, e_T1[2]/Temp, MC_mode, my.f2s(1.0 - phi - phi_2, n=phi_dgt), my.f2s(phi, n=phi_dgt), my.f2s(phi_2, n=phi_dgt), OP_set_name, seed, '1' if(to_post_proc) else '0')
									cmd =  'python %s -mode FFS_AB -L %d -to_get_timeevol 0 -N_states_FFS FFS_auto -N_init_states_FFS FFS_auto -e %lf %lf %lf -MC_move_mode %s -init_composition %s %s %s -OP_interfaces_set_IDs %s -to_show_on_screen 0 -my_seeds %d -to_post_proc %s -Temp %s -to_recomp 0 -Dtop_Nruns %d -font_mode present' \
											% (script_name, L, e_T1[0], e_T1[1], e_T1[2], MC_mode, my.f2s(1.0 - phi - phi_2, n=phi_dgt), my.f2s(phi, n=phi_dgt), my.f2s(phi_2, n=phi_dgt), OP_set_name, seed, '1' if(to_post_proc) else '0', my.f2s(Temp, n=5), Dtop_Nruns_use)
									# cmd =  'python %s -mode FFS_AB -L %d -to_get_timeevol 0 -N_states_FFS FFS_auto -N_init_states_FFS FFS_auto -e %lf %lf %lf -MC_move_mode %s -init_composition %s %s %s -OP_interfaces_set_IDs %s -to_show_on_screen 0 -my_seeds %d -to_post_proc %s -Temp %s -to_recomp 0 -Dtop_Nruns 2000 -font_mode present' \
											# % (script_name, L, e_T1[0], e_T1[1], e_T1[2], MC_mode, my.f2s(1.0 - phi - phi_2, n=phi_dgt), my.f2s(phi, n=phi_dgt), my.f2s(phi_2, n=phi_dgt), OP_set_name, seed, '1' if(to_post_proc) else '0', my.f2s(Temp, n=5))
											
									
									
									if(to_run):
										my_server.run_slurm(os.path.join('/scratch/gpfs/yp1065/Izing/slurm', name + '.slurm'), \
											cmd, \
											job_name=name, \
											time=clargs.slurm_time, \
											nodes=1, \
											ntasks_per_core=1, \
											ntasks_per_node=1, \
											cpus_per_task=1, \
											RAM_per_CPU_usage=clargs.RAM_per_CPU, \
											slurm_output_filepath = '/scratch/gpfs/yp1065/Izing/logs/' + name + '.out')
									else:
										print(cmd)
	elif(mode in ['search_logs', 'search_npzs']):
		for L in clargs.L_s:
			for Temp in clargs.Temp_s:
				for phi in clargs.phi1_s:
					n_use = int(table_data.n144h_FFS_dict[MC_mode][my.f2s(Temp, n=2)][my.f2s(phi, n=5)] * my_server.slurm_time_to_seconds(clargs.slurm_time)/(144*3600) + 0.5) if(clargs.n_s[0] < 0) else clargs.n_s[0]
					
					if(n_use > 0):
						print('\nphi1 =', phi, '; Temp =', Temp, '; mode =', MC_mode, '; n =', n_use)
						
						if(mode == 'search_logs'):
							templ = '/scratch/gpfs/yp1065/Izing/logs/FFS_MC%d_%d_%d_L128_Temp%s_ID%s_phi%s_OPsnvt.out' % \
									(lattice_gas.get_move_modes()[MC_mode], 2*n_use, n_use, my.f2s(Temp), '%d', my.f2s(phi))
							
							izing.find_finished_runs_logs(templ, seed_s, verbose=True)
						elif(mode == 'search_npzs'):
							templ = '/scratch/gpfs/yp1065/Izing/npzs/MCmoves%d_L128_eT%s_%s_%s_phi%s_%s_NinitStates%d_%d_OPs12_125_22_stab16384_OPbf20_16_stride*_initGenMode-3_timeDataFalse_Nfourier5_ID%s_FFStraj.npz' % \
									(lattice_gas.get_move_modes()[MC_mode], my.f2s(e_T1[0] / Temp), my.f2s(e_T1[1] / Temp), my.f2s(e_T1[2] / Temp), my.f2s(phi), my.f2s(phi_2), 2*n_use, n_use, '%d')
							
							izing.find_finished_runs_npzs(templ, seed_s, verbose=True)


if(__name__ == "__main__"):
	main()

	#n_24h(p1=0.015, p2=0.01, T=1, nvt16, 'swap') = 80-100
	#n_24h(p1=0.015, p2=0.01, T=1, nvt16, 'long_swap') = 2500

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
	
