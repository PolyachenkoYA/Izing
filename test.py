import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import shutil

import mylib as my
import table_data
#import lattice_gas
#import izing

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

def avg_times_FFS(thr_std=6, slurm_time=144):
	Temp_s = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
	phi_s = np.array([14, 14.5, 15, 15.5, 16]) / 1000
	
	'''
		str substitutions
		
		{
			-
			 + (1/60) * (
		}
		and
		{
			\n
			)), 
		}
		and
		{
			.
			*24 + 
		}
		=======
		{
			, )), )), 
			\n
		}
	'''
	
	time_data = {\
			'swap'    : {\
					'0.85': {'0.014' : [29+ (1/60) * (6+ (1/60) * (36)), 25+ (1/60) * (49+ (1/60) * (47)), 24+ (1/60) * (23+ (1/60) * (43)), 21+ (1/60) * (46+ (1/60) * (20)), 20+ (1/60) * (37+ (1/60) * (5)), 20+ (1/60) * (2+ (1/60) * (22)), 18+ (1/60) * (19+ (1/60) * (35)), 17+ (1/60) * (42+ (1/60) * (57)), 17+ (1/60) * (31+ (1/60) * (35)), 16+ (1/60) * (32+ (1/60) * (26))], \
							 '0.0145': [23+ (1/60) * (47+ (1/60) * (26)), 22+ (1/60) * (5+ (1/60) * (31)), 20+ (1/60) * (13+ (1/60) * (29)), 20+ (1/60) * (8+ (1/60) * (9)), 18+ (1/60) * (51+ (1/60) * (51)), 18+ (1/60) * (28+ (1/60) * (13)), 18+ (1/60) * (24+ (1/60) * (5)), 18+ (1/60) * (17+ (1/60) * (37)), 16+ (1/60) * (44+ (1/60) * (42)), 15+ (1/60) * (29+ (1/60) * (3))], \
							 '0.015' : [22+ (1/60) * (5+ (1/60) * (57)), 21+ (1/60) * (25+ (1/60) * (10)), 21+ (1/60) * (9+ (1/60) * (10)), 19+ (1/60) * (36+ (1/60) * (33)), 19+ (1/60) * (12+ (1/60) * (29)), 17+ (1/60) * (59+ (1/60) * (59)), 17+ (1/60) * (28+ (1/60) * (57)), 16+ (1/60) * (34+ (1/60) * (25)), 15+ (1/60) * (38+ (1/60) * (25)), 15+ (1/60) * (23+ (1/60) * (6))], \
							 '0.0155': [20+ (1/60) * (42+ (1/60) * (27)), 20+ (1/60) * (24+ (1/60) * (24)), 19+ (1/60) * (40+ (1/60) * (40)), 18+ (1/60) * (27+ (1/60) * (14)), 17+ (1/60) * (48+ (1/60) * (51)), 16+ (1/60) * (54+ (1/60) * (52)), 16+ (1/60) * (23+ (1/60) * (38)), 16+ (1/60) * (2+ (1/60) * (8)), 15+ (1/60) * (46+ (1/60) * (5)), 14+ (1/60) * (48+ (1/60) * (44))], \
							 '0.016' : [19+ (1/60) * (22+ (1/60) * (29)), 19+ (1/60) * (19+ (1/60) * (19)), 19+ (1/60) * (18+ (1/60) * (32)), 18+ (1/60) * (23+ (1/60) * (17)), 17+ (1/60) * (29+ (1/60) * (24)), 17+ (1/60) * (7+ (1/60) * (58)), 16+ (1/60) * (39+ (1/60) * (9)), 16+ (1/60) * (12+ (1/60) * (53)), 15+ (1/60) * (14+ (1/60) * (53)), 13+ (1/60) * (54+ (1/60) * (51))]}, \
					'0.95': {'0.014' : [13 + 55/60, 12 + 27/60, 12 + 26/60, 12 + 16/60, 12 + 18/60, 11 + 26/60, 11 + 24/60, 11 + 16/60, 10 + 24/60, 10 + 18/60, 9 + 49/60, 9 + 45/60, 9 + 36/60, 9 + 14/60, 9 + 7/60, 8 + 51/60, 8 + 53/60, 8 + 37/60, 8 + 28/60, 7 + 27/60], \
							 '0.0145': [12 + (1/60) * (25 + (1/60) * (40)), 12 + (1/60) * (16 + (1/60) * (26)), 11 + (1/60) * (16 + (1/60) * (20)), 9 + (1/60) * (48 + (1/60) * (46)), 9 + (1/60) * (36 + (1/60) * (17)), 9 + (1/60) * (7 + (1/60) * (4)), 8 + (1/60) * (53 + (1/60) * (33)), 8 + (1/60) * (36 + (1/60) * (53)), 8 + (1/60) * (27 + (1/60) * (38)), 7 + (1/60) * (26 + (1/60) * (43))], \
							 '0.015' : [16+ (1/60) * (22+ (1/60) * (56)), 13+ (1/60) * (14+ (1/60) * (6)), 12+ (1/60) * (58+ (1/60) * (30)), 12+ (1/60) * (7+ (1/60) * (52)), 11+ (1/60) * (58+ (1/60) * (10)), 11+ (1/60) * (31+ (1/60) * (23)), 10+ (1/60) * (31+ (1/60) * (18)), 10+ (1/60) * (22+ (1/60) * (6)), 10+ (1/60) * (17+ (1/60) * (47)), 9+ (1/60) * (31+ (1/60) * (58))], \
							 '0.0155': [22+ (1/60) * (56+ (1/60) * (26)), 22+ (1/60) * (19+ (1/60) * (42)), 21+ (1/60) * (40+ (1/60) * (45)), 21+ (1/60) * (36+ (1/60) * (17)), 20+ (1/60) * (3+ (1/60) * (2)), 19+ (1/60) * (42+ (1/60) * (5)), 19+ (1/60) * (35+ (1/60) * (16)), 19+ (1/60) * (23+ (1/60) * (55)), 18+ (1/60) * (8+ (1/60) * (30)), 17+ (1/60) * (2+ (1/60) * (35))], \
							 '0.016' : [26+ (1/60) * (6+ (1/60) * (36)), 25+ (1/60) * (4+ (1/60) * (59)), 23+ (1/60) * (38+ (1/60) * (55)), 23+ (1/60) * (29+ (1/60) * (45)), 23+ (1/60) * (20+ (1/60) * (13)), 22+ (1/60) * (5+ (1/60) * (47)), 19+ (1/60) * (51+ (1/60) * (33)), 19+ (1/60) * (2+ (1/60) * (17)), 18+ (1/60) * (46+ (1/60) * (2)), 17+ (1/60) * (55+ (1/60) * (21))]}, \
					'1.0' : {'0.014' : [4*24 + 21 + (1/60) * (46 + (1/60) * (22)), 4*24 + 8 + (1/60) * (25 + (1/60) * (14)), 3*24 + 22 + (1/60) * (34 + (1/60) * (59)), 3*24 + 22 + (1/60) * (6 + (1/60) * (13)), 3*24 + 20 + (1/60) * (58 + (1/60) * (47)), 3*24 + 18 + (1/60) * (58 + (1/60) * (55)), 3*24 + 17 + (1/60) * (32 + (1/60) * (57)), 3*24 + 11 + (1/60) * (49 + (1/60) * (56)), 3*24 + 3 + (1/60) * (33 + (1/60) * (43)), 2*24 + 23 + (1/60) * (21 + (1/60) * (23))], \
							 '0.0145': [4*24 + 14 + (1/60) * (29 + (1/60) * (24)), 3*24 + 21 + (1/60) * (36 + (1/60) * (27)), 3*24 + 17 + (1/60) * (31 + (1/60) * (29)), 3*24 + 10 + (1/60) * (41 + (1/60) * (19)), 3*24 + 8 + (1/60) * (7 + (1/60) * (52)), 3*24 + 7 + (1/60) * (36 + (1/60) * (38)), 3*24 + 3 + (1/60) * (48 + (1/60) * (32)), 3*24 + 1 + (1/60) * (11 + (1/60) * (36)), 3*24 + 0 + (1/60) * (30 + (1/60) * (31)), 2*24 + 23 + (1/60) * (24 + (1/60) * (53))], \
							 '0.015' : [2 * 24 + 11 + (1/60) * (21 + (1/60) * (14)), 2 * 24 + 8 + (1/60) * (16 + (1/60) * (25)), 2 * 24 + 7 + (1/60) * (53 + (1/60) * (38)), 2 * 24 + 5 + (1/60) * (23 + (1/60) * (45)), 2 * 24 + 4 + (1/60) * (51 + (1/60) * (33)), 2 * 24 + 3 + (1/60) * (56 + (1/60) * (25)), 2 * 24 + 3 + (1/60) * (16 + (1/60) * (8)), 2 * 24 + 2 + (1/60) * (7 + (1/60) * (17)), 2 * 24 + 1 + (1/60) * (35 + (1/60) * (39)), 2 * 24 + 0 + (1/60) * (32 + (1/60) * (27)), 2 * 24 + 0 + (1/60) * (7 + (1/60) * (19)), 2 * 24 + 0 + (1/60) * (2 + (1/60) * (11)), 1 * 24 + 23 + (1/60) * (25 + (1/60) * (29)), 1 * 24 + 23 + (1/60) * (20 + (1/60) * (51)), 1 * 24 + 23 + (1/60) * (18 + (1/60) * (48)), 1 * 24 + 23 + (1/60) * (2 + (1/60) * (50)), 1 * 24 + 22 + (1/60) * (4 + (1/60) * (52)), 1 * 24 + 21 + (1/60) * (26 + (1/60) * (30)), 1 * 24 + 20 + (1/60) * (41 + (1/60) * (39)), 1 * 24 + 20 + (1/60) * (1 + (1/60) * (51)), 1 * 24 + 20 + (1/60) * (5 + (1/60) * (59)), 1 * 24 + 19 + (1/60) * (30 + (1/60) * (28)), 1 * 24 + 19 + (1/60) * (41 + (1/60) * (37)), 1 * 24 + 18 + (1/60) * (46 + (1/60) * (40)), 1 * 24 + 18 + (1/60) * (42 + (1/60) * (42)), 1 * 24 + 18 + (1/60) * (9 + (1/60) * (41)), 1 * 24 + 18 + (1/60) * (26 + (1/60) * (18)), 1 * 24 + 17 + (1/60) * (34 + (1/60) * (45)), 1 * 24 + 17 + (1/60) * (43 + (1/60) * (27)), 1 * 24 + 16 + (1/60) * (43 + (1/60) * (46)), 1 * 24 + 16 + (1/60) * (13 + (1/60) * (45)), 1 * 24 + 16 + (1/60) * (8 + (1/60) * (41)), 1 * 24 + 15 + (1/60) * (37 + (1/60) * (57)), 1 * 24 + 15 + (1/60) * (3 + (1/60) * (27)), 1 * 24 + 13 + (1/60) * (39 + (1/60) * (54)), 1 * 24 + 11 + (1/60) * (49 + (1/60) * (15)), 1 * 24 + 11 + (1/60) * (22 + (1/60) * (58)), 1 * 24 + 10 + (1/60) * (49 + (1/60) * (45)), 1 * 24 + 10 + (1/60) * (28 + (1/60) * (40)), 1 * 24 + 9 + (1/60) * (13 + (1/60) * (45)), 1 * 24 + 8 + (1/60) * (55 + (1/60) * (26)), 1 * 24 + 6 + (1/60) * (24 + (1/60) * (31))]}\
							} , \
#					'1.0' : {'0.014' : [53 + 53/60, 53 + 4/60, 47 + 12/60, 47 + 22/60, 46 + 19/60, 40 + 49/60, 37 + 16/60, 36 + 49/60, 30 + 47/60, 28 + 14/60], \
#							 '0.0145': [49 + 50/60, 37 + 49/60, 36 + 7/60, 32 + 56/60, 32 + 25/60, 31 + 44/60, 29 + 19/60, 28 + 24/60, 28 + 9/60, 26 + 48/60], \
#							 '0.015' : [2 * 24 + 11 + (1/60) * (21 + (1/60) * (14)), 2 * 24 + 8 + (1/60) * (16 + (1/60) * (25)), 2 * 24 + 7 + (1/60) * (53 + (1/60) * (38)), 2 * 24 + 5 + (1/60) * (23 + (1/60) * (45)), 2 * 24 + 4 + (1/60) * (51 + (1/60) * (33)), 2 * 24 + 3 + (1/60) * (56 + (1/60) * (25)), 2 * 24 + 3 + (1/60) * (16 + (1/60) * (8)), 2 * 24 + 2 + (1/60) * (7 + (1/60) * (17)), 2 * 24 + 1 + (1/60) * (35 + (1/60) * (39)), 2 * 24 + 0 + (1/60) * (32 + (1/60) * (27)), 2 * 24 + 0 + (1/60) * (7 + (1/60) * (19)), 2 * 24 + 0 + (1/60) * (2 + (1/60) * (11)), 1 * 24 + 23 + (1/60) * (25 + (1/60) * (29)), 1 * 24 + 23 + (1/60) * (20 + (1/60) * (51)), 1 * 24 + 23 + (1/60) * (18 + (1/60) * (48)), 1 * 24 + 23 + (1/60) * (2 + (1/60) * (50)), 1 * 24 + 22 + (1/60) * (4 + (1/60) * (52)), 1 * 24 + 21 + (1/60) * (26 + (1/60) * (30)), 1 * 24 + 20 + (1/60) * (41 + (1/60) * (39)), 1 * 24 + 20 + (1/60) * (1 + (1/60) * (51)), 1 * 24 + 20 + (1/60) * (5 + (1/60) * (59)), 1 * 24 + 19 + (1/60) * (30 + (1/60) * (28)), 1 * 24 + 19 + (1/60) * (41 + (1/60) * (37)), 1 * 24 + 18 + (1/60) * (46 + (1/60) * (40)), 1 * 24 + 18 + (1/60) * (42 + (1/60) * (42)), 1 * 24 + 18 + (1/60) * (9 + (1/60) * (41)), 1 * 24 + 18 + (1/60) * (26 + (1/60) * (18)), 1 * 24 + 17 + (1/60) * (34 + (1/60) * (45)), 1 * 24 + 17 + (1/60) * (43 + (1/60) * (27)), 1 * 24 + 16 + (1/60) * (43 + (1/60) * (46)), 1 * 24 + 16 + (1/60) * (13 + (1/60) * (45)), 1 * 24 + 16 + (1/60) * (8 + (1/60) * (41)), 1 * 24 + 15 + (1/60) * (37 + (1/60) * (57)), 1 * 24 + 15 + (1/60) * (3 + (1/60) * (27)), 1 * 24 + 13 + (1/60) * (39 + (1/60) * (54)), 1 * 24 + 11 + (1/60) * (49 + (1/60) * (15)), 1 * 24 + 11 + (1/60) * (22 + (1/60) * (58)), 1 * 24 + 10 + (1/60) * (49 + (1/60) * (45)), 1 * 24 + 10 + (1/60) * (28 + (1/60) * (40)), 1 * 24 + 9 + (1/60) * (13 + (1/60) * (45)), 1 * 24 + 8 + (1/60) * (55 + (1/60) * (26)), 1 * 24 + 6 + (1/60) * (24 + (1/60) * (31))]}\
		
		  'long_swap' : {\
					'0.85': {'0.014' : [8 + (1/60) * (43 + (1/60) * (18)), 8 + (1/60) * (43 + (1/60) * (0)), 8 + (1/60) * (32 + (1/60) * (13)), 8 + (1/60) * (37 + (1/60) * (13)), 8 + (1/60) * (15 + (1/60) * (45)), 8 + (1/60) * (8 + (1/60) * (47)), 8 + (1/60) * (0 + (1/60) * (54)), 7 + (1/60) * (40 + (1/60) * (39)), 7 + (1/60) * (40 + (1/60) * (12)), 7 + (1/60) * (26 + (1/60) * (59))], \
							 '0.0145': [8 + (1/60) * (57 + (1/60) * (1)), 8 + (1/60) * (26 + (1/60) * (4)), 8 + (1/60) * (19 + (1/60) * (30)), 8 + (1/60) * (17 + (1/60) * (29)), 8 + (1/60) * (4 + (1/60) * (11)), 7 + (1/60) * (59 + (1/60) * (35)), 8 + (1/60) * (0 + (1/60) * (31)), 7 + (1/60) * (47 + (1/60) * (52)), 7 + (1/60) * (12 + (1/60) * (55)), 7 + (1/60) * (0 + (1/60) * (23))], \
							 '0.015' : [8 + (1/60) * (16 + (1/60) * (20)), 7 + (1/60) * (59 + (1/60) * (3)), 7 + (1/60) * (55 + (1/60) * (57)), 7 + (1/60) * (54 + (1/60) * (45)), 7 + (1/60) * (43 + (1/60) * (50)), 7 + (1/60) * (39 + (1/60) * (21)), 7 + (1/60) * (42 + (1/60) * (54)), 7 + (1/60) * (22 + (1/60) * (25)), 6 + (1/60) * (57 + (1/60) * (39)), 6 + (1/60) * (55 + (1/60) * (31))], \
							 '0.0155': [7 + (1/60) * (54 + (1/60) * (38)), 7 + (1/60) * (32 + (1/60) * (35)), 7 + (1/60) * (35 + (1/60) * (15)), 7 + (1/60) * (10 + (1/60) * (1)), 7 + (1/60) * (12 + (1/60) * (17)), 7 + (1/60) * (7 + (1/60) * (11)), 7 + (1/60) * (1 + (1/60) * (13)), 6 + (1/60) * (55 + (1/60) * (28)), 6 + (1/60) * (51 + (1/60) * (39)), 6 + (1/60) * (38 + (1/60) * (33))], \
							 '0.016' : [7 + (1/60) * (11 + (1/60) * (56)), 7 + (1/60) * (13 + (1/60) * (10)), 7 + (1/60) * (3 + (1/60) * (31)), 6 + (1/60) * (51 + (1/60) * (2)), 6 + (1/60) * (44 + (1/60) * (51)), 6 + (1/60) * (31 + (1/60) * (24)), 6 + (1/60) * (34 + (1/60) * (15)), 6 + (1/60) * (28 + (1/60) * (0)), 6 + (1/60) * (24 + (1/60) * (32)), 5 + (1/60) * (48 + (1/60) * (55))]}, \
					'0.9' : {'0.014' : [20 + (1/60) * (9 + (1/60) * (51)), 19 + (1/60) * (16 + (1/60) * (56)), 19 + (1/60) * (8 + (1/60) * (38)), 18 + (1/60) * (43 + (1/60) * (2)), 18 + (1/60) * (18 + (1/60) * (15)), 18 + (1/60) * (13 + (1/60) * (50)), 17 + (1/60) * (59 + (1/60) * (36)), 17 + (1/60) * (55 + (1/60) * (46)), 17 + (1/60) * (54 + (1/60) * (52)), 17 + (1/60) * (29 + (1/60) * (29))], \
							 '0.0145': [19 + (1/60) * (29 + (1/60) * (26)), 19 + (1/60) * (24 + (1/60) * (15)), 19 + (1/60) * (18 + (1/60) * (58)), 19 + (1/60) * (9 + (1/60) * (21)), 18 + (1/60) * (19 + (1/60) * (20)), 18 + (1/60) * (8 + (1/60) * (54)), 18 + (1/60) * (7 + (1/60) * (12)), 17 + (1/60) * (59 + (1/60) * (41)), 17 + (1/60) * (26 + (1/60) * (13)), 16 + (1/60) * (57 + (1/60) * (38))], \
							 '0.015' : [20 + (1/60) * (16 + (1/60) * (17)), 19 + (1/60) * (20 + (1/60) * (56)), 18 + (1/60) * (59 + (1/60) * (48)), 18 + (1/60) * (44 + (1/60) * (22)), 18 + (1/60) * (42 + (1/60) * (10)), 18 + (1/60) * (9 + (1/60) * (31)), 18 + (1/60) * (0 + (1/60) * (56)), 17 + (1/60) * (50 + (1/60) * (0)), 17 + (1/60) * (47 + (1/60) * (6)), 17 + (1/60) * (44 + (1/60) * (24))], \
							 '0.0155': [18 + (1/60) * (28 + (1/60) * (12)), 18 + (1/60) * (16 + (1/60) * (59)), 18 + (1/60) * (8 + (1/60) * (7)), 17 + (1/60) * (51 + (1/60) * (39)), 17 + (1/60) * (22 + (1/60) * (28)), 17 + (1/60) * (8 + (1/60) * (1)), 16 + (1/60) * (57 + (1/60) * (47)), 16 + (1/60) * (57 + (1/60) * (5)), 16 + (1/60) * (19 + (1/60) * (24)), 16 + (1/60) * (17 + (1/60) * (31))], \
							 '0.016' : [18 + (1/60) * (0 + (1/60) * (11)), 17 + (1/60) * (47 + (1/60) * (6)), 17 + (1/60) * (37 + (1/60) * (51)), 17 + (1/60) * (32 + (1/60) * (47)), 17 + (1/60) * (24 + (1/60) * (34)), 16 + (1/60) * (48 + (1/60) * (22)), 16 + (1/60) * (43 + (1/60) * (30)), 16 + (1/60) * (22 + (1/60) * (39)), 15 + (1/60) * (55 + (1/60) * (54)), 15 + (1/60) * (35 + (1/60) * (31))]}, \
					'0.95': {'0.014' : [0 + (1/60) * (20 + (1/60) * (2)), 0 + (1/60) * (18 + (1/60) * (50)), 0 + (1/60) * (17 + (1/60) * (4)), 0 + (1/60) * (19 + (1/60) * (11)), 0 + (1/60) * (18 + (1/60) * (36)), 0 + (1/60) * (17 + (1/60) * (11)), 0 + (1/60) * (16 + (1/60) * (13)), 0 + (1/60) * (17 + (1/60) * (7)), 0 + (1/60) * (17 + (1/60) * (29)), 0 + (1/60) * (14 + (1/60) * (6))], \
							 '0.0145': [4 + (1/60) * (25 + (1/60) * (52)), 4 + (1/60) * (18 + (1/60) * (34)), 4 + (1/60) * (16 + (1/60) * (19)), 4 + (1/60) * (16 + (1/60) * (57)), 4 + (1/60) * (15 + (1/60) * (4)), 4 + (1/60) * (4 + (1/60) * (4)), 4 + (1/60) * (6 + (1/60) * (21)), 4 + (1/60) * (4 + (1/60) * (28)), 4 + (1/60) * (1 + (1/60) * (16)), 3 + (1/60) * (57 + (1/60) * (12))], \
							 '0.015' : [7 + (1/60) * (31 + (1/60) * (4)), 7 + (1/60) * (18 + (1/60) * (11)), 6 + (1/60) * (55 + (1/60) * (42)), 6 + (1/60) * (57 + (1/60) * (21)), 6 + (1/60) * (54 + (1/60) * (46)), 6 + (1/60) * (55 + (1/60) * (50)), 6 + (1/60) * (46 + (1/60) * (16)), 6 + (1/60) * (28 + (1/60) * (25)), 6 + (1/60) * (17 + (1/60) * (12)), 6 + (1/60) * (18 + (1/60) * (3))], \
							 '0.0155': [11 + 42/60, 10 + 54/60, 10 + 41/60, 10 + 29/60, 10 + 12/60, 10 + 7/60, 10 + 0/60, 9 + 48/60, 9 + 40/60, 9 + 27/60], \
							 '0.016' : [13 + 24/60, 13 + 14/60, 13 + 6/60, 12 + 50/60, 12 + 54/60, 12 + 50/60, 12 + 37/60, 12 + 32/60, 12 + 32/60, 12 + 19/60]}, \
					'1.0' : {'0.014' : [2*24 + 1 + (1/60) * (32 + (1/60) * (43)), 1*24 + 21 + (1/60) * (36 + (1/60) * (0)), 1*24 + 21 + (1/60) * (9 + (1/60) * (55)), 1*24 + 20 + (1/60) * (58 + (1/60) * (34)), 1*24 + 15 + (1/60) * (12 + (1/60) * (0)), 1*24 + 14 + (1/60) * (53 + (1/60) * (34)), 1*24 + 15 + (1/60) * (6 + (1/60) * (27)), 1*24 + 14 + (1/60) * (31 + (1/60) * (57)), 1*24 + 14 + (1/60) * (17 + (1/60) * (27)), 1*24 + 12 + (1/60) * (53 + (1/60) * (32))], \
							 '0.0145': [23 + 56/60, 23 + 40/60, 23 + 28/60, 23 + 23/60, 22 + 48/60, 22 + 40/60, 22 + 39/60, 22 + 18/60, 22 + 10/60, 21 + 34/60, 21 + 17/60, 21 + 5/60, 21 + 3/60, 21 + 3/60, 21 + 0/60, 20 + 53/60, 20 + 36/60, 19 + 57/60, 19 + 35/60, 23 + 13/60, 23 + 6/60, 22 + 41/60, 22 + 36/60, 22 + 18/60, 21 + 59/60, 21 + 21/60, 21 + 18/60, 23 + 1.15, 23 + 1.3, 23 + 1.5, 23 + 2, 23 + 2.7]}\
						}\
				}
	
	n_base = {\
		'swap'      : {
					'0.85': {'0.014' : 45, \
							 '0.0145': 45, \
							 '0.015' : 45, \
							 '0.0155': 45, \
							 '0.016' : 45}, \
					'0.95': {'0.014' : 20, \
							 '0.0145': 20, \
							 '0.015' : 33, \
							 '0.0155': 57, \
							 '0.016' : 63}, \
					'1.0' : {'0.014' : 29, \
							 '0.0145': 56, \
							 '0.015' : 33}\
						} , \
#					'1.0' : {'0.014' : 10, \
#							 '0.0145': 15, \
#							 '0.015' : 33}\
		  
		'long_swap' : {
					'0.85': {'0.014' : 500, \
							 '0.0145': 500, \
							 '0.015' : 500, \
							 '0.0155': 500, \
							 '0.016' : 500}, \
					'0.9': {'0.014' : 1200, \
							 '0.0145': 1200, \
							 '0.015' : 1200, \
							 '0.0155': 1200, \
							 '0.016' : 1200}, \
					'0.95': {'0.014' : 20, \
							 '0.0145': 300, \
							 '0.015' : 500, \
							 '0.0155': 750, \
							 '0.016' : 950}, \
					'1.0' : {'0.014' : 385, \
							 '0.0145': 300}\
					}\
			}
	slurm_time_units = 1
	
	Temp_grid, phi_grid = np.meshgrid(phi_s, Temp_s)
	times = {}
	times_thr = {}
	n_opt = {}
	for MCmode in time_data:
		times[MCmode] = {}
		times_thr[MCmode] = {}
		n_opt[MCmode] = {}
		for Temp in time_data[MCmode]:
			times[MCmode][Temp] = {}
			times_thr[MCmode][Temp] = {}
			n_opt[MCmode][Temp] = {}
			for phi1 in time_data[MCmode][Temp]:
				time_data[MCmode][Temp][phi1] = np.array(time_data[MCmode][Temp][phi1])
				times[MCmode][Temp][phi1] = my.get_average(time_data[MCmode][Temp][phi1])
				times_thr[MCmode][Temp][phi1] = times[MCmode][Temp][phi1][0] + times[MCmode][Temp][phi1][1] * thr_std * np.sqrt(len(time_data[MCmode][Temp][phi1]) - 1) * np.sqrt(times[MCmode][Temp][phi1][0] / slurm_time)
				n_opt[MCmode][Temp][phi1] = np.intc(slurm_time  * slurm_time_units / times_thr[MCmode][Temp][phi1] * n_base[MCmode][Temp][phi1])
	
	print('times:')
	print(times)
	print('time_thr:')
	print(times_thr)
	print('n_%d h:' % slurm_time)
	print(n_opt)
	
	# times, d_times = my.get_average(times_data, axis=2)
	# print(times.shape, times_data.shape)
	# print(Temp_grid.shape, phi_grid.shape)
	
	# times_std = d_times * np.sqrt(times_data.shape[2] - 1)
	# times_thr = times + times_std * thr_std
	# print('times:')
	# print(times)
	# print(times_std)
	# print('time_thr:')
	# print(times_thr)
	# print('Nffs for %s h:' % my.f2s(slurm_time))
	# print(np.intc(slurm_time  * slurm_time_units / times_thr * n_base))
	
	# fig, ax, _ = my.get_fig('$T$', '$\phi_1$', projection='3d')
	# # surf = ax.plot_surface(phi_grid.T, Temp_grid.T, times.T,
						# # linewidth=0, cmap=cm.coolwarm)
	# # fig.colorbar(surf, shrink=0.5, aspect=5)
	# inds = times.flatten() > 0
	# ax.errorbar(phi_grid.flatten()[inds], Temp_grid.flatten()[inds], times.flatten()[inds], zerr=times_std.flatten()[inds], fmt='.')
	
	# plt.show()

def avg_times_Dtop(thr_std=6, slurm_time=144):
	Temp_s = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
	phi_s = np.array([14, 14.5, 15, 15.5, 16]) / 1000
	
	'''
		str substitutions
		
		{
			-
			 + (1/60) * (
		}
		and
		{
			\n
			)), 
		}
		and
		{
			.
			*24 + 
		}
		=======
		{
			)), )), 
			\n
		}
		{
			\n
			))\n
		}
		{
			(0
			(
		}
	'''
	
	time_data = {\
			'swap'    : {\
					'0.85': {'0.014' : [0 + (1/60) * (39 + (1/60) * (42)), 0 + (1/60) * (32 + (1/60) * (46)), 0 + (1/60) * (28 + (1/60) * (10)), 0 + (1/60) * (15 + (1/60) * (59)), 0 + (1/60) * (17 + (1/60) * (53)), 0 + (1/60) * (21 + (1/60) * (13))], \
							 '0.0145': [0 + (1/60) * (50 + (1/60) * (41)), 0 + (1/60) * (45 + (1/60) * (5)), 0 + (1/60) * (42 + (1/60) * (11)), 0 + (1/60) * (39 + (1/60) * (41)), 0 + (1/60) * (33 + (1/60) * (53)), 0 + (1/60) * (36 + (1/60) * (8))], \
							 '0.015' : [0 + (1/60) * (51 + (1/60) * (51)), 0 + (1/60) * (37 + (1/60) * (22)), 0 + (1/60) * (45 + (1/60) * (31)), 0 + (1/60) * (34 + (1/60) * (48)), 0 + (1/60) * (29 + (1/60) * (52)), 0 + (1/60) * (24 + (1/60) * (37))], \
							 '0.0155': [0 + (1/60) * (48 + (1/60) * (31)), 0 + (1/60) * (47 + (1/60) * (7)), 0 + (1/60) * (47 + (1/60) * (44)), 0 + (1/60) * (37 + (1/60) * (37)), 0 + (1/60) * (40 + (1/60) * (5)), 0 + (1/60) * (14 + (1/60) * (46))], \
							 '0.016' : [0 + (1/60) * (59 + (1/60) * (10)), 0 + (1/60) * (54 + (1/60) * (54)), 0 + (1/60) * (52 + (1/60) * (5)), 0 + (1/60) * (46 + (1/60) * (59)), 0 + (1/60) * (51 + (1/60) * (54)), 0 + (1/60) * (45 + (1/60) * (40))]}, \
					'0.9': {'0.014' : [0 + (1/60) * (22 + (1/60) * (41)), 0 + (1/60) * (21 + (1/60) * (31)), 0 + (1/60) * (16 + (1/60) * (22)), 0 + (1/60) * (10 + (1/60) * (10)), 0 + (1/60) * (8 + (1/60) * (41)), 0 + (1/60) * (13 + (1/60) * (18))], \
							 '0.0145': [0 + (1/60) * (15 + (1/60) * (18)), 0 + (1/60) * (14 + (1/60) * (50)), 0 + (1/60) * (12 + (1/60) * (27)), 0 + (1/60) * (12 + (1/60) * (19)), 0 + (1/60) * (13 + (1/60) * (55)), 0 + (1/60) * (13 + (1/60) * (0))], \
							 '0.015' : [0 + (1/60) * (39 + (1/60) * (49)), 0 + (1/60) * (20 + (1/60) * (11)), 0 + (1/60) * (16 + (1/60) * (28)), 0 + (1/60) * (10 + (1/60) * (18)), 0 + (1/60) * (16 + (1/60) * (34)), 0 + (1/60) * (8 + (1/60) * (7))], \
							 '0.0155': [0 + (1/60) * (20 + (1/60) * (16)), 0 + (1/60) * (11 + (1/60) * (53)), 0 + (1/60) * (17 + (1/60) * (48)), 0 + (1/60) * (6 + (1/60) * (21)), 0 + (1/60) * (4 + (1/60) * (57)), 0 + (1/60) * (3 + (1/60) * (6))], \
							 '0.016' : [0 + (1/60) * (23 + (1/60) * (13)), 0 + (1/60) * (21 + (1/60) * (39)), 0 + (1/60) * (20 + (1/60) * (40)), 0 + (1/60) * (16 + (1/60) * (8)), 0 + (1/60) * (15 + (1/60) * (1)), 0 + (1/60) * (7 + (1/60) * (6))]}, \
					'0.95': {'0.014' : [0 + (1/60) * (31 + (1/60) * (57)), 0 + (1/60) * (25 + (1/60) * (2)), 0 + (1/60) * (23 + (1/60) * (45)), 0 + (1/60) * (19 + (1/60) * (51)), 0 + (1/60) * (15 + (1/60) * (5)), 0 + (1/60) * (18 + (1/60) * (17))], \
							 '0.0145': [0 + (1/60) * (25 + (1/60) * (23)), 0 + (1/60) * (21 + (1/60) * (10)), 0 + (1/60) * (28 + (1/60) * (26)), 0 + (1/60) * (18 + (1/60) * (47)), 0 + (1/60) * (15 + (1/60) * (25)), 0 + (1/60) * (17 + (1/60) * (16))], \
							 '0.015' : [0 + (1/60) * (59 + (1/60) * (12)), 0 + (1/60) * (24 + (1/60) * (11)), 0 + (1/60) * (25 + (1/60) * (37)), 0 + (1/60) * (23 + (1/60) * (36)), 0 + (1/60) * (20 + (1/60) * (34)), 0 + (1/60) * (26 + (1/60) * (14))], \
							 '0.0155': [0 + (1/60) * (23 + (1/60) * (43)), 0 + (1/60) * (21 + (1/60) * (42)), 0 + (1/60) * (19 + (1/60) * (53)), 0 + (1/60) * (19 + (1/60) * (55)), 0 + (1/60) * (15 + (1/60) * (34)), 0 + (1/60) * (16 + (1/60) * (42))], \
							 '0.016' : [0 + (1/60) * (59 + (1/60) * (57)), 0 + (1/60) * (36 + (1/60) * (50)), 0 + (1/60) * (24 + (1/60) * (14)), 0 + (1/60) * (27 + (1/60) * (14)), 0 + (1/60) * (20 + (1/60) * (19)), 0 + (1/60) * (18 + (1/60) * (59))]}, \
					'1.0' : {'0.014' : [0 + (1/60) * (29 + (1/60) * (31)), 0 + (1/60) * (16 + (1/60) * (58)), 0 + (1/60) * (3 + (1/60) * (44)), 0 + (1/60) * (2 + (1/60) * (33)), 0 + (1/60) * (3 + (1/60) * (17))], \
							 '0.0145' : [4 + (1/60) * (30 + (1/60) * (3)), 3 + (1/60) * (20 + (1/60) * (2)), 3 + (1/60) * (13 + (1/60) * (6)), 3 + (1/60) * (3 + (1/60) * (52)), 2 + (1/60) * (28 + (1/60) * (2)), 2 + (1/60) * (13 + (1/60) * (34))], \
							 '0.015' : [6 + (1/60) * (20 + (1/60) * (31)), 5 + (1/60) * (22 + (1/60) * (44)), 3 + (1/60) * (41 + (1/60) * (11)), 2 + (1/60) * (55 + (1/60) * (52)), 2 + (1/60) * (21 + (1/60) * (50)), 0 + (1/60) * (55 + (1/60) * (16))], \
							 '0.0155': [2 + (1/60) * (9 + (1/60) * (40)), 2 + (1/60) * (6 + (1/60) * (57)), 1 + (1/60) * (49 + (1/60) * (51)), 1 + (1/60) * (8 + (1/60) * (11)), 0 + (1/60) * (26 + (1/60) * (7)), 0 + (1/60) * (22 + (1/60) * (53))], \
							 '0.016' : [0 + (1/60) * (37 + (1/60) * (52)), 0 + (1/60) * (41 + (1/60) * (35)), 0 + (1/60) * (18 + (1/60) * (0)), 0 + (1/60) * (17 + (1/60) * (39)), 0 + (1/60) * (10 + (1/60) * (5)), 0 + (1/60) * (10 + (1/60) * (0))]}\
							} , \
		
		  'long_swap' : {\
					'0.85': {'0.014' : [0 + (1/60) * (44 + (1/60) * (13)), 0 + (1/60) * (44 + (1/60) * (3)), 0 + (1/60) * (45 + (1/60) * (33)), 0 + (1/60) * (40 + (1/60) * (9)), 0 + (1/60) * (41 + (1/60) * (18)), 0 + (1/60) * (48 + (1/60) * (20))], \
							 '0.0145': [0 + (1/60) * (36 + (1/60) * (49)), 0 + (1/60) * (35 + (1/60) * (32)), 0 + (1/60) * (36 + (1/60) * (45)), 0 + (1/60) * (34 + (1/60) * (16)), 0 + (1/60) * (36 + (1/60) * (35)), 0 + (1/60) * (36 + (1/60) * (42))], \
							 '0.015' : [0 + (1/60) * (36 + (1/60) * (33)), 0 + (1/60) * (39 + (1/60) * (52)), 0 + (1/60) * (36 + (1/60) * (54)), 0 + (1/60) * (39 + (1/60) * (45)), 0 + (1/60) * (39 + (1/60) * (24)), 0 + (1/60) * (23 + (1/60) * (27))], \
							 '0.0155': [0 + (1/60) * (20 + (1/60) * (38)), 0 + (1/60) * (42 + (1/60) * (19)), 0 + (1/60) * (39 + (1/60) * (49)), 0 + (1/60) * (38 + (1/60) * (12)), 0 + (1/60) * (40 + (1/60) * (15)), 0 + (1/60) * (38 + (1/60) * (38))], \
							 '0.016' : [0 + (1/60) * (41 + (1/60) * (30)), 0 + (1/60) * (39 + (1/60) * (28)), 0 + (1/60) * (36 + (1/60) * (13)), 0 + (1/60) * (39 + (1/60) * (8)), 0 + (1/60) * (37 + (1/60) * (26)), 0 + (1/60) * (39 + (1/60) * (11))]}, \
					'0.9': {'0.014' : [2 + (1/60) * (5 + (1/60) * (8)), 0 + (1/60) * (56 + (1/60) * (21)), 0 + (1/60) * (53 + (1/60) * (21)), 0 + (1/60) * (38 + (1/60) * (20)), 0 + (1/60) * (59 + (1/60) * (31)), 0 + (1/60) * (54 + (1/60) * (56))], \
							 '0.0145': [1 + (1/60) * (15 + (1/60) * (6)), 1 + (1/60) * (14 + (1/60) * (9)), 1 + (1/60) * (17 + (1/60) * (24)), 0 + (1/60) * (56 + (1/60) * (43)), 1 + (1/60) * (20 + (1/60) * (29)), 0 + (1/60) * (51 + (1/60) * (7))], \
							 '0.015' : [1 + (1/60) * (4 + (1/60) * (27)), 1 + (1/60) * (15 + (1/60) * (5)), 1 + (1/60) * (20 + (1/60) * (42)), 1 + (1/60) * (7 + (1/60) * (18)), 1 + (1/60) * (10 + (1/60) * (47)), 1 + (1/60) * (10 + (1/60) * (5))], \
							 '0.0155': [1 + (1/60) * (30 + (1/60) * (34)), 1 + (1/60) * (20 + (1/60) * (12)), 0 + (1/60) * (57 + (1/60) * (0)), 0 + (1/60) * (40 + (1/60) * (6)), 1 + (1/60) * (9 + (1/60) * (28)), 1 + (1/60) * (28 + (1/60) * (44))], \
							 '0.016' : [1 + (1/60) * (25 + (1/60) * (24)), 1 + (1/60) * (15 + (1/60) * (38)), 1 + (1/60) * (1 + (1/60) * (58)), 0 + (1/60) * (53 + (1/60) * (9)), 1 + (1/60) * (10 + (1/60) * (48)), 1 + (1/60) * (28 + (1/60) * (52))]}, \
					'0.95': {'0.014' : [1 + (1/60) * (7 + (1/60) * (1)), 1 + (1/60) * (11 + (1/60) * (13)), 1 + (1/60) * (8 + (1/60) * (39)), 1 + (1/60) * (6 + (1/60) * (19)), 1 + (1/60) * (8 + (1/60) * (6)), 0 + (1/60) * (40 + (1/60) * (30))], \
							 '0.0145': [1 + (1/60) * (4 + (1/60) * (19)), 0 + (1/60) * (40 + (1/60) * (21)), 1 + (1/60) * (4 + (1/60) * (29)), 1 + (1/60) * (2 + (1/60) * (10)), 0 + (1/60) * (43 + (1/60) * (3)), 0 + (1/60) * (43 + (1/60) * (27))], \
							 '0.015' : [1 + (1/60) * (0 + (1/60) * (57)), 1 + (1/60) * (2 + (1/60) * (48)), 0 + (1/60) * (59 + (1/60) * (24)), 0 + (1/60) * (55 + (1/60) * (45)), 0 + (1/60) * (59 + (1/60) * (52)), 1 + (1/60) * (5 + (1/60) * (42))], \
							 '0.0155': [1 + (1/60) * (16 + (1/60) * (51)), 0 + (1/60) * (35 + (1/60) * (43)), 0 + (1/60) * (55 + (1/60) * (53)), 1 + (1/60) * (12 + (1/60) * (8)), 1 + (1/60) * (15 + (1/60) * (3)), 1 + (1/60) * (12 + (1/60) * (20))], \
							 '0.016' : [1 + (1/60) * (23 + (1/60) * (17)), 1 + (1/60) * (26 + (1/60) * (28)), 1 + (1/60) * (31 + (1/60) * (59)), 1 + (1/60) * (29 + (1/60) * (38)), 1 + (1/60) * (24 + (1/60) * (6)), 1 + (1/60) * (25 + (1/60) * (23))]}, \
					'1.0' : {'0.014' : [0 + (1/60) * (18 + (1/60) * (29)), 0 + (1/60) * (17 + (1/60) * (46)), 0 + (1/60) * (18 + (1/60) * (44)), 0 + (1/60) * (17 + (1/60) * (25)), 0 + (1/60) * (18 + (1/60) * (21)), 0 + (1/60) * (18 + (1/60) * (9))], \
							 '0.015': [4 + (1/60) * (45 + (1/60) * (14)), 4 + (1/60) * (6 + (1/60) * (9)), 4 + (1/60) * (33 + (1/60) * (48)), 4 + (1/60) * (16 + (1/60) * (57)), 4 + (1/60) * (12 + (1/60) * (55)), 3 + (1/60) * (36 + (1/60) * (4))], \
							 '0.0155': [2 + (1/60) * (35 + (1/60) * (2)), 2 + (1/60) * (17 + (1/60) * (2)), 2 + (1/60) * (33 + (1/60) * (12)), 2 + (1/60) * (37 + (1/60) * (19)), 1 + (1/60) * (35 + (1/60) * (3)), 1 + (1/60) * (47 + (1/60) * (40))], \
							 '0.016' : [1 + (1/60) * (54 + (1/60) * (20)), 1 + (1/60) * (52 + (1/60) * (54)), 2 + (1/60) * (9 + (1/60) * (38)), 2 + (1/60) * (9 + (1/60) * (30)), 2 + (1/60) * (22 + (1/60) * (11)), 1 + (1/60) * (19 + (1/60) * (3))]}\
						}\
				}
	
	n_base = {\
		'swap'      : {
					'0.85': {'0.014' : 360, \
							 '0.0145': 440, \
							 '0.015' : 460, \
							 '0.0155': 500, \
							 '0.016' : 520}, \
					'0.9': {'0.014' : 230, \
							 '0.0145': 250, \
							 '0.015' : 270, \
							 '0.0155': 280, \
							 '0.016' : 310}, \
					'0.95': {'0.014' : 370, \
							 '0.0145': 390, \
							 '0.015' : 520, \
							 '0.0155': 560, \
							 '0.016' : 530}, \
					'1.0' : {'0.014' : 50, \
							 '0.0145': 90, \
							 '0.015' : 120, \
							 '0.0155' : 190, \
							 '0.016' : 210}\
						} , \
				
				
		'long_swap' : {
					'0.85': {'0.014' : 11060, \
							 '0.0145': 10660, \
							 '0.015' : 11830, \
							 '0.0155': 12800, \
							 '0.016' : 13160}, \
					'0.9': {'0.014' : 12000, \
							 '0.0145': 12000, \
							 '0.015' : 12000, \
							 '0.0155': 12000, \
							 '0.016' : 12000}, \
					'0.95': {'0.014' : 10580, \
							 '0.0145': 14270, \
							 '0.015' : 13160, \
							 '0.0155': 12770, \
							 '0.016' : 15430}, \
					'1.0' : {'0.014' : 2351, \
							 '0.015' : 5000, \
							 '0.0155' : 7500, \
							 '0.016' : 9500}\
					}\
			}
	slurm_time_units = 1
	
	Temp_grid, phi_grid = np.meshgrid(phi_s, Temp_s)
	times = {}
	times_thr = {}
	n_opt = {}
	for MCmode in time_data:
		times[MCmode] = {}
		times_thr[MCmode] = {}
		n_opt[MCmode] = {}
		for Temp in time_data[MCmode]:
			times[MCmode][Temp] = {}
			times_thr[MCmode][Temp] = {}
			n_opt[MCmode][Temp] = {}
			for phi1 in time_data[MCmode][Temp]:
				time_data[MCmode][Temp][phi1] = np.array(time_data[MCmode][Temp][phi1])
				times[MCmode][Temp][phi1] = my.get_average(time_data[MCmode][Temp][phi1])
				times_thr[MCmode][Temp][phi1] = times[MCmode][Temp][phi1][0] + times[MCmode][Temp][phi1][1] * thr_std * np.sqrt(len(time_data[MCmode][Temp][phi1]) - 1) * np.sqrt(times[MCmode][Temp][phi1][0] / slurm_time)
				n_opt[MCmode][Temp][phi1] = np.intc(slurm_time  * slurm_time_units / times_thr[MCmode][Temp][phi1] * n_base[MCmode][Temp][phi1])
	
	print('times:')
	print(times)
	print('time_thr:')
	print(times_thr)
	print('n_%d h:' % slurm_time)
	print(n_opt)

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
	
	#avg_times()
	
	#pubind11_bug_test(sys.argv[1])
	
	avg_times_Dtop()
	

if(__name__ == "__main__"):
	main()
