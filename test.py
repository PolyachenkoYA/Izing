import numpy as np
import matplotlib.pyplot as plt
import os

import mylib as my

def slurm_time_distr():
	times_list = [[0,2,9], [0,2,11], [0,2,13], [0,2,54], [0,2,59], [0,3,1], [0,3,42], [0,4,2], [0,5,4], [0,5,11], [0,5,13], [0,5,43], [0,6,3], [0,6,27], [0,7,49], [0,8,54], [0,9,42], [0,9,54], [0,9,57], [0,10,26], [0,10,56], [0,11,25], [0,11,35], [0,12,8], [0,12,17], [0,12,55], [0,12,57], [0,13,40], [0,14,58], [0,17,12], [0,18,7], [0,18,13], [0,18,28], [0,19,1], [0,19,13], [0,19,28], [0,19,44], [0,22,1], [0,24,21], [0,27,11], [0,28,25], [0,31,7], [0,31,27], [0,35,54], [0,40,3], [0,41,14], [0,45,12], [0,44,29], [0,43,40], [0,47,6], [0,47,29], [0,51,36], [0,58,4], [1,10,41], [1,22,1], [1,20,27], [1,29,7], [1,30,41], [1,35,30], [1,44,5], [1,50,8], [1,51,1], [1,52,5], [1,53,4], [2,10,10], [2,18,23], [2,19,37], [2,20,40], [2,32,29], [2,39,1], [2,40,57], [2,44,0], [2,45,56], [2,47,29], [3,28,55], [4,19,17], [4,25,37], [4,29,43], [5,34,41]]
	
	times = np.array([(t[2] + 60 * (t[1] + 60 * t[0])) for t in times_list])
	
	N = 10
	edges = np.exp(np.linspace(np.log(min(times) * 0.99), np.log(max(times) * 1.01), N))
	hist, _ = np.histogram(times, bins=edges)
	lens = edges[1:] - edges[:-1]
	rho = hist / np.sum(hist) / lens
	t_centers = np.sqrt(edges[1:] * edges[:-1])
	
	#fix, ax, _ = my.get_fig(r'$\log_{10}(t/[sec])$', r'$\sim \rho$', title=r'$\rho(t)$', yscl='log')
	fix, ax, _ = my.get_fig(r't (sec)', r'$\rho_t$', title=r'$\rho(t)$', xscl='log', yscl='log')
	
	ax.errorbar(t_centers, rho, fmt='.-', label='data')
	ax.plot(t_centers, 1e-3 * 170 / t_centers, label='A/t')
	for i, e in enumerate(edges):
		ax.plot([e]*2, [min(rho), max(rho)], '--', color=my.get_my_color(4), label='edges' if(i == 0) else None)
	
	ax.legend()
	
	plt.show()

def find_finished_runs(filename_mask, inds, verbose=False):
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

def main():
	#slurm_time_distr()
	
	#find_finished_runs('slurm-FFS_MC2_30_15_L64_ID%d_phi0.09_OPsnvt18.out', np.arange(100, 200), verbose=True)
	#find_finished_runs('slurm-FFS_MC2_500_250_L64_ID%d_phi0.07_OPsnvt18.out', np.arange(100, 200), verbose=True)
	#find_finished_runs('/scratch/gpfs/yp1065/Izing/logs/slurm-FFS_MC2_500_250_L128_ID%d_phi0.05_OPsnvt18.out', np.arange(100, 500), verbose=True)
	find_finished_runs('/scratch/gpfs/yp1065/Izing/logs/slurm-FFS_MC2_500_250_L64_ID%d_phi0.07_OPsnvt18.out', np.arange(100, 500), verbose=True)

if(__name__ == "__main__"):
	main()
