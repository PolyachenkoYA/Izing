import numpy as np
import os
import glob

import mylib as my

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
	found_list = []
	found_mult_list = []
	for i in inds:
		filename = filename_mask % i
		n_found = len(glob.glob(filename))
		if(n_found == 1):
			found_list.append(i)
		elif(n_found > 1):
			found_mult_list.append(i)
	
	if(verbose):
		def report_found(l):
			print('Number of found files:', len(l))
			print(' '.join([str(d) for d in l]))
		
		report_found(found_list)
		if(len(found_mult_list) > 0):
			print('WARNING: multiplt candidates found for')
			report_found(found_mult_list)
	
	return found_list



