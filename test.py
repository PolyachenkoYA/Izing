import numpy as np
import os

import mylib as my

if(__name__ == "__main__"):
	to_recompile = True
	if(to_recompile):
		path_to_so = 'Init_state_gen/cmake-build-release'
		N_dirs_down = path_to_so.count('/') + 1
		path_back = '/'.join(['..'] * N_dirs_down)

		os.chdir(path_to_so)
		my.run_it('make izing.so')
		os.chdir(path_back)
		my.run_it('mv %s/izing.so.cpython-38-x86_64-linux-gnu.so ./izing.so' % (path_to_so))
		print('recompiled izing')

	import izing

state = np.array([[ 1,  1, -1, -1, -1, -1], \
				  [-1, -1,  1, -1, -1, -1], \
				  [-1, -1, -1, -1, -1, -1], \
				  [-1, -1, -1, -1, -1, -1], \
				  [-1, -1, -1, -1, -1, -1], \
				  [-1, -1, -1, -1, -1, -1]], \
				  dtype=int)

L = state.shape[0]
L2 = L*L
assert(L == state.shape[1])

verbose = 1
my_seed = 0

izing.init_rand(my_seed)
izing.set_verbose(verbose)

inds, sizes = izing.cluster_state(state.reshape((L2)))
N_inds = np.sum(sizes)
print(sizes)
print(inds[:N_inds])
