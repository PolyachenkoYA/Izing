import numpy as np
import os
import glob

import mylib as my
import table_data

import lattice_gas
move_modes, move_mode_names = lattice_gas.get_move_modes()

sgm_th_izing = lambda e: (e/2 + np.log(np.tanh(e/4)) + np.sqrt(2) * np.log(np.sinh(e/2))) / (2 * (1 - np.sinh(e/2)**(-4))**(1.0/16))   # sgm/T
# fig, ax, _ = my.get_fig('T/J = 4T/e = 1/K', r'$\sigma / J$')
# e_draw = np.linspace(2,8,100)
# ax.plot(4/e_draw, sgm_th(e_draw) / (e_draw/4))
# plt.show()

rho_c = 7.46e-3

binflags = {'all' : 2**(4*8+1) - 1, \
	'resimulate' : 2**1, \
	'Dtop_est' : 2**2, \
	'postproc_light' : 2**3, \
	'postproc_hard' : 2**4 + 2**3, \
	'0' : 0
	}

def find_finished_runs_logs(filename_mask, inds, verbose=False):
	done_list = []
	found_list = []
	for i in inds:
		filename = filename_mask % i
		if(os.path.isfile(filename)):
			found_list.append(i)
			lines = open(filename, 'r').readlines()
			if(lines[-1].startswith('-log10(k_AB * [1 step]) = (')):
				done_list.append(i)
	
	if(verbose):
		print('found:', len(found_list), '; done:', len(done_list))
		print(' '.join([str(d) for d in done_list]))
	
	return done_list

def find_finished_runs_npzs(filename_mask, inds, verbose=False):
	found_IDs = []
	found_filepaths = []
	found_mult_IDs = []
	found_mult_filepaths = []
	for i in inds:
		filename = filename_mask % i
		filepaths = glob.glob(filename)
		n_found = len(filepaths)
		if(n_found == 1):
			found_IDs.append(i)
			found_filepaths.append(filepaths[0])
		elif(n_found > 1):
			found_mult_IDs.append(i)
			found_mult_filepaths.append(filepaths)
	
	if(verbose):
		def report_found(l):
			print('Number of found files:', len(l))
			print(' '.join([str(d) for d in l]))
		
		print('mask:', filename_mask)
		report_found(found_IDs)
		if(len(found_mult_IDs) > 0):
			print('WARNING: multiple candidates found for')
			report_found(found_mult_IDs)
	
	return found_IDs, found_filepaths, found_mult_filepaths

def get_FFS_AB_npzTrajBasename(MC_move_mode, L, e, mu_str, OP_interfaces, N_init_states, \
						stab_step, OP_sample_BF_to, OP_match_BF_to, \
						timeevol_stride, init_gen_mode, to_get_timeevol, \
						N_fourier, seed, n_e_digits=6, mu=None, phi=None):
	
	#print(timeevol_stride_dict.keys())
	#print(e[1,1])
	swap_type_move = (MC_move_mode in [move_modes['swap'], move_modes['long_swap']])
	
	old_basenames = ['MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str + \
						'_NinitStates' + '_'.join([str(n) for n in N_init_states]) + \
						'_OPs' + '_'.join([str(ops) for ops in OP_interfaces]) + \
						'_stab' + str(stab_step) + \
						'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
						'_stride' + str(timeevol_stride) + \
						'_initGenMode' + str(init_gen_mode) + \
						'_timeData' + str(to_get_timeevol) + \
						'_ID' + str(seed), \
					'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str + \
						'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
						'_OPs' + '_'.join([str(ops) for ops in OP_interfaces]) + \
						'_stab' + str(stab_step) + \
						'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
						'_stride' + str(timeevol_stride) + \
						'_initGenMode' + str(init_gen_mode) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Nfourier' + str(N_fourier) + \
						'_ID' + str(seed), \
					'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str + \
						'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
						'_OPs' + '_'.join([str(ops) for ops in (np.append(OP_interfaces[[0, len(OP_interfaces)-1]], len(OP_interfaces)))]) + \
						'_stab' + str(stab_step) + \
						'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
						(('_stride' + str(timeevol_stride)) if(to_get_timeevol) else '') + \
						'_initGenMode' + str(init_gen_mode) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Nfourier' + str(N_fourier) + \
						'_ID' + str(seed)]
						#'_stride' + str(timeevol_stride) + \
	
	'''
					'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str + \
						'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
						'_OPs' + '_'.join([str(ops) for ops in (np.append(OP_interfaces[[0, len(OP_interfaces)-1]], len(OP_interfaces)))]) + \
						'_stab' + str(stab_step) + \
						'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
						'_stride' + str(table_data.timeevol_stride_dict[my.f2s(e[1,1], n=3)] if(my.f2s(e[1,1], n=3) in table_data.timeevol_stride_dict) else timeevol_stride) + \
						'_initGenMode' + str(init_gen_mode) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Nfourier' + str(N_fourier) + \
						'_ID' + str(seed), \
	'''
	
	if(((mu is not None) or (phi is not None))):
		mu_str_old = ('_phi' + my.join_lbls(phi[1:], '_')) \
						if(swap_type_move) \
						else ('_muT' + my.join_lbls(mu[1:], '_'))
		if(mu_str_old != mu_str):
			for on in ['MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
							'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str_old + \
							'_NinitStates' + '_'.join([str(n) for n in N_init_states]) + \
							'_OPs' + '_'.join([str(ops) for ops in OP_interfaces]) + \
							'_stab' + str(stab_step) + \
							'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
							'_stride' + str(timeevol_stride) + \
							'_initGenMode' + str(init_gen_mode) + \
							'_timeData' + str(to_get_timeevol) + \
							'_ID' + str(seed), \
						'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
							'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str_old + \
							'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
							'_OPs' + '_'.join([str(ops) for ops in OP_interfaces]) + \
							'_stab' + str(stab_step) + \
							'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
							'_stride' + str(timeevol_stride) + \
							'_initGenMode' + str(init_gen_mode) + \
							'_timeData' + str(to_get_timeevol) + \
							'_Nfourier' + str(N_fourier) + \
							'_ID' + str(seed), \
						'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
							'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str_old + \
							'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
							'_OPs' + '_'.join([str(ops) for ops in (np.append(OP_interfaces[[0, len(OP_interfaces)-1]], len(OP_interfaces)))]) + \
							'_stab' + str(stab_step) + \
							'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
							(('_stride' + str(timeevol_stride)) if(to_get_timeevol) else '') + \
							'_initGenMode' + str(init_gen_mode) + \
							'_timeData' + str(to_get_timeevol) + \
							'_Nfourier' + str(N_fourier) + \
							'_ID' + str(seed)]:
				old_basenames.append(on)
	'''
						'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
							'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str_old + \
							'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
							'_OPs' + '_'.join([str(ops) for ops in (np.append(OP_interfaces[[0, len(OP_interfaces)-1]], len(OP_interfaces)))]) + \
							'_stab' + str(stab_step) + \
							'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
							'_stride' + str(table_data.timeevol_stride_dict[my.f2s(e[1,1], n=3)] if(my.f2s(e[1,1], n=3) in table_data.timeevol_stride_dict) else timeevol_stride) + \
							'_initGenMode' + str(init_gen_mode) + \
							'_timeData' + str(to_get_timeevol) + \
							'_Nfourier' + str(N_fourier) + \
							'_ID' + str(seed), \
	'''
	# TODO: remove OPbf to timeevol-only
	# TODO: remove Nfourier (does not affect simulations)
	# TODO: remove timeData (because presence of _stride already tells it)
	basename = 'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_', n=n_e_digits) + mu_str + \
						'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
						'_OPs' + '_'.join([str(ops) for ops in (np.append(OP_interfaces[[0, len(OP_interfaces)-1]], len(OP_interfaces)))]) + \
						'_stab' + str(stab_step) + \
						'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
						(('_stride' + str(timeevol_stride)) if(to_get_timeevol) else '') + \
						'_initGenMode' + str(init_gen_mode) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Nfourier' + str(N_fourier) + \
						'_ID' + str(seed)
	
	return basename, old_basenames

def get_BF_npzTrajBasename(MC_move_mode, L, e, mu_str, Nt, N_saved_states_max, \
						R_clust_init, stab_step, OP_A, OP_B, OP_min, OP_max, \
						OP_min_save_state, OP_max_save_state, \
						timeevol_stride, to_equilibrate, to_get_timeevol, \
						seed, n_e_digits=6, mu=None, phi=None):
	
	old_basenames = ['MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str + \
						'_Nt' + str(Nt) + \
						'_NstatesMax' + str(N_saved_states_max) + \
						'_stride' + str(timeevol_stride) + \
						'_OPab' + str(OP_A) + '_' + str(OP_B) + \
						'_OPminx' + str(OP_min) + '_' + str(OP_max) + \
						'_OPminxSave' + str(OP_min_save_state) + '_' + str(OP_max_save_state) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Rinit' + str(R_clust_init) + \
						'_ID' + str(seed), \
					'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str + \
						'_Nt' + str(Nt) + '_NstatesMax' + str(N_saved_states_max) + \
						'_stabStep' + str(stab_step) + \
						(('_stride' + str(timeevol_stride)) if(to_get_timeevol) else '') + \
						'_OPab' + str(OP_A) + '_' + str(OP_B) + \
						'_OPminx' + str(OP_min) + '_' + str(OP_max) + \
						'_OPminxSave' + str(OP_min_save_state) + '_' + str(OP_max_save_state) + \
						('' if(R_clust_init is None) else ('_Rinit' + str(R_clust_init))) + \
						'_toEquil' + str(to_equilibrate) + \
						'_ID' + str(seed)]
						# '_ID' + str(lattice_gas.get_seed()
	
	if(((mu is not None) or (phi is not None))):
		mu_str_old = ('_phi' + my.join_lbls(phi[1:], '_')) \
						if(swap_type_move) \
						else ('_muT' + my.join_lbls(mu[1:], '_'))
		if(mu_str_old != mu_str):
			for on in ['MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str_old + \
						'_Nt' + str(Nt) + \
						'_NstatesMax' + str(N_saved_states_max) + \
						'_stride' + str(timeevol_stride) + \
						'_OPab' + str(OP_A) + '_' + str(OP_B) + \
						'_OPminx' + str(OP_min) + '_' + str(OP_max) + \
						'_OPminxSave' + str(OP_min_save_state) + '_' + str(OP_max_save_state) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Rinit' + str(R_clust_init) + \
						'_ID' + str(seed), \
					'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str_old + \
						'_Nt' + str(Nt) + '_NstatesMax' + str(N_saved_states_max) + \
						'_stabStep' + str(stab_step) + \
						(('_stride' + str(timeevol_stride)) if(to_get_timeevol) else '') + \
						'_OPab' + str(OP_A) + '_' + str(OP_B) + \
						'_OPminx' + str(OP_min) + '_' + str(OP_max) + \
						'_OPminxSave' + str(OP_min_save_state) + '_' + str(OP_max_save_state) + \
						('' if(R_clust_init is None) else ('_Rinit' + str(R_clust_init))) + \
						'_toEquil' + str(to_equilibrate) + \
						'_ID' + str(seed)]:
				old_basenames.append(on)
	
	basename = 'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
					'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_', n=n_e_digits) + mu_str + \
					'_Nt' + str(Nt) + '_NstatesMax' + str(N_saved_states_max) + \
					'_stabStep' + str(stab_step) + \
					(('_stride' + str(timeevol_stride)) if(to_get_timeevol) else '') + \
					'_OPab' + str(OP_A) + '_' + str(OP_B) + \
					'_OPminx' + str(OP_min) + '_' + str(OP_max) + \
					'_OPminxSave' + str(OP_min_save_state) + '_' + str(OP_max_save_state) + \
					('' if(R_clust_init is None) else ('_Rinit' + str(R_clust_init))) + \
					'_toEquil' + str(to_equilibrate) + \
					'_ID' + str(seed)
	
	return basename, old_basenames
