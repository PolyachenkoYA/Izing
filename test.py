import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

import mylib as my
import table_data
import lattice_gas
import izing

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

def avg_times(thr_std=3, slurm_time=24):
	Temp_s = np.array([0.8, 0.9, 1.0])
	phi_s = np.array([14, 14.5, 15, 15.5, 16]) / 1000
	
	# long_swap
	times_data = np.array([[[63, 62, 56, 55, 54, 54], [53, 52, 45, 49.7, 53.5, 53.3], [65, 43, 56, 53, 45, 41], [52.7, 46, 50, 42.75, 44.3, 44.3], [54, 45.5, 47.2, 52, 51, 43]], \
					  [[45.5, 49.2, 47.3, 45.5, 46.7, 46], [55, 51, 44, 37.25, 44, 43.7], [56, 48, 47, 48, 52, 40], [50.5, 39.75, 45, 41.5, 41, 36.6], [32.75, 39, 43, 38, 39.7, 38.3]], \
					  [[0,0,0,0,0,0], [60*3+47, 60*3+10, 60*3+28, 60*3+19, 60*3+15, 60*3+4], [125, 107, 129, 101, 114, 104], [86, 72.5, 78, 74, 67, 81], [65, 63, 61, 48, 55, 55]]])
	slurm_time_units = 60
	n_base = 50
	
	# swap
	times_data = np.array([[[15, 15, 15, 14, 10, 8], [18, 15, 14, 14, 14, 6], [17, 17, 15, 14, 13, 8], [14, 14, 14, 13, 13, 7], [15, 14, 12, 10, 8, 5]], \
					  [[15, 12, 12, 10, 8, 5], [13, 12, 12, 11, 10, 5], [13, 11, 10, 9, 8, 5], [12, 12, 11, 8, 7, 6], [11, 10, 10, 9, 8, 5]], \
					  [[0,0,0,0,0,0], [0,0,0,0,0,0], [22, 19, 20, 28, 30, 33], [18, 12, 17, 18, 19, 16], [16, 15, 15, 14, 13, 7]]]) + \
				np.array([[[46, 31, 15, 1, 52, 1], [14, 45, 13, 1, 0, 15], [58, 39, 0, 16, 4, 12], [42, 22, 10, 28, 20, 56], [45, 15, 17, 17, 26, 59]], \
					  [[15, 50, 9, 56, 45, 0], [59, 26, 4, 22, 4, 25], [17, 54, 55, 25, 41, 21], [28, 11, 53, 21, 12, 47], [29, 56, 23, 44, 39, 41]], \
					  [[0,0,0,0,0,0], [0,0,0,0,0,0], [40, 9, 49, 0,0,0], [58, 26, 48, 41, 25, 44], [39, 18, 15, 4, 41, 58]]]) / 60
	slurm_time_units = 1
	n_base = 20
	
	Temp_grid, phi_grid = np.meshgrid(phi_s, Temp_s)
	times, d_times = my.get_average(times_data, axis=2)
	print(times.shape, times_data.shape)
	print(Temp_grid.shape, phi_grid.shape)
	
	times_std = d_times * np.sqrt(times_data.shape[2] - 1)
	times_thr = times + times_std * thr_std
	print('times:')
	print(times)
	print(times_std)
	print('time_thr:')
	print(times_thr)
	print('Nffs for %s h:' % my.f2s(slurm_time))
	print(np.intc(slurm_time  * slurm_time_units / times_thr * n_base))
	
	fig, ax, _ = my.get_fig('$T$', '$\phi_1$', projection='3d')
	# surf = ax.plot_surface(phi_grid.T, Temp_grid.T, times.T,
						# linewidth=0, cmap=cm.coolwarm)
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	inds = times.flatten() > 0
	ax.errorbar(phi_grid.flatten()[inds], Temp_grid.flatten()[inds], times.flatten()[inds], zerr=times_std.flatten()[inds], fmt='.')
	
	plt.show()

def main():
	#slurm_time_distr()
	
	#find_finished_runs_logs('slurm-FFS_MC2_30_15_L64_ID%d_phi0.09_OPsnvt18.out', np.arange(100, 200), verbose=True)
	#find_finished_runs_logs('slurm-FFS_MC2_500_250_L64_ID%d_phi0.07_OPsnvt18.out', np.arange(100, 200), verbose=True)
	#find_finished_runs_logs('/scratch/gpfs/yp1065/Izing/logs/slurm-FFS_MC2_500_250_L128_ID%d_phi0.05_OPsnvt18.out', np.arange(100, 500), verbose=True)
	#find_finished_runs_logs('/scratch/gpfs/yp1065/Izing/logs/slurm-FFS_MC2_500_250_L64_ID%d_phi0.07_OPsnvt18.out', np.arange(100, 500), verbose=True)
	#find_finished_runs_logs('/scratch/gpfs/yp1065/Izing/logs/FFS_MC2_60_30_L128_ID%d_phi0.015_OPsnvt16.out', np.arange(100, 500), verbose=True)
	#find_finished_runs_logs('/scratch/gpfs/yp1065/Izing/logs/FFS_MC2_160_80_L128_ID%d_phi0.015_OPsnvt16.out', np.arange(100, 500), verbose=True)
	#find_finished_runs_logs('/scratch/gpfs/yp1065/Izing/logs/FFS_MC2_30_15_L128_ID%d_phi0.015_OPsnvt16.out', np.arange(500, 1000), verbose=True)
	
	# phi1 = 0.0145
	# Temp = 0.8
	# MC_mode = 'swap'
	# n = table_data.n_FFS_dict[MC_mode][Temp][phi1]
	# print('phi1 =', phi1, '; Temp =', Temp, '; mode =', MC_mode)
	# izing.find_finished_runs_logs('/scratch/gpfs/yp1065/Izing/logs/FFS_MC%d_%d_%d_L128_Temp%s_ID%s_phi%s_OPsnvt.out' % \
										# (lattice_gas.get_move_modes()[MC_mode], 2*n, n, my.f2s(Temp), '%d', my.f2s(phi1)), \
									# np.arange(1000, 1100), verbose=True)
	
	#find_finished_runs_npzs('/scratch/gpfs/yp1065/Izing/npzs/MCmoves2_L128_eT-2.68_-1.34_-1.72_phi0.015_0.01_NinitStates160_80_80_80_80_80_80_80_80_80_80_80_80_80_80_OPs19_23_27_31_35_39_43_48_52_56_60_65_70_75_80_stab16384_OPbf27_23_stride763_initGenMode-3_timeDataFalse_ID%d_FFStraj.npz', np.arange(100, 500), verbose=True)
	#find_finished_runs_npzs('/scratch/gpfs/yp1065/Izing/npzs/MCmoves2_L128_eT-2.68_-1.34_-1.72_phi0.015_0.01_NinitStates160_80_80_80_80_80_80_80_80_80_80_80_80_80_80_OPs19_23_27_31_35_39_43_48_52_56_60_65_70_75_80_stab16384_OPbf27_23_stride763_initGenMode-3_timeDataFalse_ID%d_FFStraj.npz', np.arange(500, 1000), verbose=True)
	
	avg_times()
	

if(__name__ == "__main__"):
	main()
