import numpy as np
import os
import glob

import mylib as my
import table_data

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
			print('WARNING: multiplt candidates found for')
			report_found(found_mult_IDs)
	
	return found_IDs, found_filepaths, found_mult_filepaths

def get_FFS_AB_npzTrajBasename(MC_move_mode, L, e, mu_str, OP_interfaces, N_init_states, \
						stab_step, OP_sample_BF_to, OP_match_BF_to, \
						timeevol_stride, init_gen_mode, to_get_timeevol, \
						N_fourier, seed):
	
	#print(timeevol_stride_dict.keys())
	#print(e[1,1])
	
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
						'_stride' + str(table_data.timeevol_stride_dict[my.f2s(e[1,1], n=3)]) + \
						'_initGenMode' + str(init_gen_mode) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Nfourier' + str(N_fourier) + \
						'_ID' + str(seed)]
						#'_stride' + str(timeevol_stride) + \
	
	# TODO: remove OPbf to timeevol-only
	basename = 'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
						'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + mu_str + \
						'_NinitStates' + '_'.join([str(n) for n in N_init_states[:2]]) + \
						'_OPs' + '_'.join([str(ops) for ops in (np.append(OP_interfaces[[0, len(OP_interfaces)-1]], len(OP_interfaces)))]) + \
						'_stab' + str(stab_step) + \
						'_OPbf' + str(OP_sample_BF_to) + '_' + str(OP_match_BF_to) + \
						(('_stride' + str(timeevol_stride)) if(timeevol_stride is not None) else '') + \
						'_initGenMode' + str(init_gen_mode) + \
						'_timeData' + str(to_get_timeevol) + \
						'_Nfourier' + str(N_fourier) + \
						'_ID' + str(seed)
	
	return basename, old_basenames


