import numpy as np
import matplotlib.pyplot as plt
import os

import mylib as my

my.run_it('make izing.so')
my.run_it('mv izing.so.cpython-38-x86_64-linux-gnu.so izing.so')

import izing
#exit()

def proc_FFS(L, Temp, h, N_init_states, M_interfaces, verbose=None, max_flip_time=None, to_get_EM=False, to_plot=True):
	# py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> M_interfaces, int to_get_EM, std::optional<int> _verbose)
	# return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M);
	(_states, probs, d_probs, Nt, flux0, d_flux0, _E, _M) = izing.run_FFS(L, Temp, h, N_init_states, M_interfaces, verbose=verbose, to_get_EM=to_get_EM)
	
	L2 = L**2;
	N_M_interfaces = len(M_interfaces) - 2   # '-2' to be consistent with C++ implementation. 2 interfaces are '-L2-1' and '+L2'
	
	states = [_states[ : N_init_states[0] * L2].reshape((N_init_states[0], L, L))]
	for i in range(1, N_M_interfaces + 1):
		states.append(_states[np.sum(N_init_states[:i]) * L2 : np.sum(N_init_states[:(i+1)] * L2)].reshape((N_init_states[i], L, L)))
	del _states
	
	print('Nt:', Nt)
	print('flux0 = (%lf +- %lf) 1/step' % (flux0, d_flux0))
	print('-log(P):', -np.log(probs))
	print('d_P / P:', d_probs / probs)
	print('k_AB =', flux0 * np.prod(probs[1:-1]), ' 1/step')   # probs[0] == 1 because its the probability to go from A to M_0
	
	if(to_get_EM):
		E = [_E[ : Nt[0]]]
		for i in range(1, N_M_interfaces + 1):
			E.append(_E[np.sum(Nt[:i]) : np.sum(Nt[:(i+1)])])
		del _E
		M = [_M[ : Nt[0]]]
		for i in range(1, N_M_interfaces + 1):
			M.append(_M[np.sum(Nt[:i]) : np.sum(Nt[:(i+1)])])
		del _M
		
		M = M / L**2
		E = E / L**2
		Nt = len(E)
		
		steps = np.arange(Nt)
		dE_step = 8   # the maximum dE possible for 1 step = 'change in a spin' * 'max neighb spin' = 2*4 = 8
		dM_step = 2   # the magnitude of the spin flip, from -1 to 1
		if(max_flip_time is None):
			#max_flip_time = np.exp(dE_step / Temp)
			# The upper-bound estimate on the number of steps for 1 flip to happen
			# This would be true if I had all the attempts (even unsuccesful ones) to flip
			
			max_flip_time = 1.0
			# we save only succesful flips, so <flip time> = 1
		
		stab_step = int(min(L**2 * max_flip_time, Nt / 2)) * 5
		# Each spin has a good chance to flip during this time
		# This does not guarantee the global stable state, but it is sufficient to be sure-enough we are in the optimum of the local metastable-state.
		# The approximation for a global ebulibration would be '1/k_AB', where k_AB is the rate constant. But it's hard to estimate on that stage.
		stab_ind = (steps > stab_step)
		
		M_stab = M[stab_ind]
		Nt_stab = len(M_stab)
		M_mean = np.mean(M_stab)
		M_std = np.std(M_stab)
		memory_time = max(1, (M_std * L**2) / dM_step * max_flip_time)   
		# Time of statistical decorelation of the system state.
		# 'the number of flips necessary to cover the typical system states range' * 'the upper-bound estimate on the number of steps for 1 flip to happen'
		print('memory time =', my.f2s(memory_time))

		E_stab = E[stab_ind]
		E_mean = np.mean(E_stab)
		E_std = np.std(E_stab)
		
		M_hist_edges = L**2 + 1   # auto-build for edges
		M_hist_edges = (np.arange(L**2 + 2) * 2 - (L**2 + 1)) / L**2
		M_hist, M_hist_edges = np.histogram(M_stab, bins=M_hist_edges)
		# The number of bins cannot be arbitrary because M is descrete, thus it's a good idea if all the bins have 1 descrete value inside or all the bins have 2 or etc. 
		# It's not good if some bins have e.g. 1 possible M value, and others have 2 possible values. 
		# There are 'L^2 + 1' possible values of M, because '{M \in [-L^2; L^2]} and {dM = 2}'
		# Thus, we need edges which cover [-1-1/L^2; 1+1/L^2] with step=2/L^2
		M_hist_lens = (M_hist_edges[1:] - M_hist_edges[:-1])
		M_hist[M_hist == 0] = 1   # to not get errors for log(hist)
		M_hist = M_hist / memory_time
		M_hist_centers = (M_hist_edges[1:] + M_hist_edges[:-1]) / 2
		rho = M_hist / M_hist_lens / Nt_stab
		d_rho = np.sqrt(M_hist * (1 - M_hist / Nt_stab)) / M_hist_lens / Nt_stab
		F = -Temp * np.log(rho * M_hist_lens)
		d_F = Temp * d_rho / rho
		
		C = E_std**2 / Temp**2 * L**2

		d_E_mean = E_std / np.sqrt(Nt_stab / memory_time)
		d_M_mean = M_std / np.sqrt(Nt_stab / memory_time)
		
		if(to_plot):
			fig_E, ax_E = my.get_fig('step', '$E / L^2$', title='E(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_E.plot(steps, E, label='data')
			ax_E.plot([stab_step] * 2, [min(E), max(E)], label='equilibr')
			ax_E.plot([stab_step, Nt], [E_mean] * 2, label=('$<E> = ' + my.errorbar_str(E_mean, d_E_mean) + '$'))
			ax_E.legend()
			
			fig_M, ax_M = my.get_fig('step', '$M / L^2$', title='M(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_M.plot(steps, M, label='data')
			ax_M.plot([stab_step] * 2, [min(M), max(M)], label='equilibr')
			ax_M.plot([stab_step, Nt], [M_mean] * 2, label=('$<M> = ' + my.errorbar_str(M_mean, d_M_mean) + '$'))
			ax_M.legend()
			
			fig_Mhist, ax_Mhist = my.get_fig(r'$m = M / L^2$', r'$\rho(m)$', title=r'$\rho(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_Mhist.bar(M_hist_centers, rho, yerr=d_rho, width=M_hist_lens, align='center')
			#ax_Mhist.legend()
			
			fig_F, ax_F = my.get_fig(r'$m = M / L^2$', r'$F(m) = -T \ln(\rho(m))$', title=r'$F(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_F.errorbar(M_hist_centers, F, yerr=d_F)
			#ax_F.legend()

		return E_mean, d_E_mean, M_mean, d_M_mean, C

def proc_T(L, Temp, h, N0, M0, my_seed, verbose=None, max_flip_time=None, to_get_EM=False, to_plot=True):
	(init_states, E, M) = izing.get_init_states(L, Temp, h, N0, M0, verbose=verbose, to_get_EM=to_get_EM)
	
	if(to_get_EM):
		M = M / L**2
		E = E / L**2
		Nt = len(E)
		
		steps = np.arange(Nt)
		dE_step = 8   # the maximum dE possible for 1 step = 'change in a spin' * 'max neighb spin' = 2*4 = 8
		dM_step = 2   # the magnitude of the spin flip, from -1 to 1
		if(max_flip_time is None):
			#max_flip_time = np.exp(dE_step / Temp)
			# The upper-bound estimate on the number of steps for 1 flip to happen
			# This would be true if I had all the attempts (even unsuccesful ones) to flip
			
			max_flip_time = 1.0
			# we save only succesful flips, so <flip time> = 1
		
		stab_step = int(min(L**2 * max_flip_time, Nt / 2)) * 5
		# Each spin has a good chance to flip during this time
		# This does not guarantee the global stable state, but it is sufficient to be sure-enough we are in the optimum of the local metastable-state.
		# The approximation for a global ebulibration would be '1/k_AB', where k_AB is the rate constant. But it's hard to estimate on that stage.
		stab_ind = (steps > stab_step)
		
		M_stab = M[stab_ind]
		Nt_stab = len(M_stab)
		M_mean = np.mean(M_stab)
		M_std = np.std(M_stab)
		memory_time = max(1, (M_std * L**2) / dM_step * max_flip_time)   
		# Time of statistical decorelation of the system state.
		# 'the number of flips necessary to cover the typical system states range' * 'the upper-bound estimate on the number of steps for 1 flip to happen'
		print('memory time =', my.f2s(memory_time))

		E_stab = E[stab_ind]
		E_mean = np.mean(E_stab)
		E_std = np.std(E_stab)
		
		M_hist_edges = L**2 + 1   # auto-build for edges
		M_hist_edges = (np.arange(L**2 + 2) * 2 - (L**2 + 1)) / L**2
		M_hist, M_hist_edges = np.histogram(M_stab, bins=M_hist_edges)
		# The number of bins cannot be arbitrary because M is descrete, thus it's a good idea if all the bins have 1 descrete value inside or all the bins have 2 or etc. 
		# It's not good if some bins have e.g. 1 possible M value, and others have 2 possible values. 
		# There are 'L^2 + 1' possible values of M, because '{M \in [-L^2; L^2]} and {dM = 2}'
		# Thus, we need edges which cover [-1-1/L^2; 1+1/L^2] with step=2/L^2
		M_hist_lens = (M_hist_edges[1:] - M_hist_edges[:-1])
		M_hist[M_hist == 0] = 1   # to not get errors for log(hist)
		M_hist = M_hist / memory_time
		M_hist_centers = (M_hist_edges[1:] + M_hist_edges[:-1]) / 2
		rho = M_hist / M_hist_lens / Nt_stab
		d_rho = np.sqrt(M_hist * (1 - M_hist / Nt_stab)) / M_hist_lens / Nt_stab
		F = -Temp * np.log(rho * M_hist_lens)
		d_F = Temp * d_rho / rho
		
		C = E_std**2 / Temp**2 * L**2

		d_E_mean = E_std / np.sqrt(Nt_stab / memory_time)
		d_M_mean = M_std / np.sqrt(Nt_stab / memory_time)
		
		if(to_plot):
			fig_E, ax_E = my.get_fig('step', '$E / L^2$', title='E(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_E.plot(steps, E, label='data')
			ax_E.plot([stab_step] * 2, [min(E), max(E)], label='equilibr')
			ax_E.plot([stab_step, Nt], [E_mean] * 2, label=('$<E> = ' + my.errorbar_str(E_mean, d_E_mean) + '$'))
			ax_E.legend()
			
			fig_M, ax_M = my.get_fig('step', '$M / L^2$', title='M(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_M.plot(steps, M, label='data')
			ax_M.plot([stab_step] * 2, [min(M), max(M)], label='equilibr')
			ax_M.plot([stab_step, Nt], [M_mean] * 2, label=('$<M> = ' + my.errorbar_str(M_mean, d_M_mean) + '$'))
			ax_M.legend()
			
			fig_Mhist, ax_Mhist = my.get_fig(r'$m = M / L^2$', r'$\rho(m)$', title=r'$\rho(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_Mhist.bar(M_hist_centers, rho, yerr=d_rho, width=M_hist_lens, align='center')
			#ax_Mhist.legend()
			
			fig_F, ax_F = my.get_fig(r'$m = M / L^2$', r'$F(m) = -T \ln(\rho(m))$', title=r'$F(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_F.errorbar(M_hist_centers, F, yerr=d_F)
			#ax_F.legend()

		return E_mean, d_E_mean, M_mean, d_M_mean, C

# def get_E_T(box, Nt, my_seed, T_arr, time_verb=0, E_verb=0):
	# N_T = len(T_arr)
	# E_arr = np.empty(N_T)
	# d_E_arr = np.empty(N_T)
	# M_arr = np.empty(N_T)
	# d_M_arr = np.empty(N_T)
	# C_fluct = np.empty(N_T)
	# for i in range(N_T):
		# E_arr[i], d_E_arr[i], M_arr[i], d_M_arr[i], C_fluct[i] = proc_T(T_arr[i], box, Nt, my_seed, verbose=time_verb)
		# print((i + 1) / N_T)
	
	# return E_arr, d_E_arr, M_arr, d_M_arr, C_fluct

# def get_deriv(x, y, n_gap=2, degr=1):
	# N = len(x)
	
	# deriv = []
	# x_der = []
	# #for i in range(n_gap, N - n_gap - 2):
	# for i in range(N):
		# fit_ind = np.arange(max(0, i - n_gap), min(N - 1, i + n_gap))
		# fit = np.polyfit(x[fit_ind] - x[i], y[fit_ind], degr)
		# deriv.append(fit[degr - 1])
		# x_der.append(x[i])
	
	# return np.array(x_der), np.array(deriv)
	
# def proc_N(box, Nt, my_seed, ax_C=None, ax_E=None, ax_M=None, ax_M2=None, recomp=0, T_min_log10=-0.1, T_max_log10=1.5, N_T=100):
	# N_particles = box**2
		
	# T_arr = np.power(10, np.linspace(T_min_log10, T_max_log10, N_T))
	# #T_arr = [1, 10]
	# res_filename = r'N%d_Nstep%d_Tmin%lf_Tmax%lf_NT%d_ID%d.npz' % (box, Nt, T_min_log10, T_max_log10, N_T, my_seed)
	
	# if(os.path.isfile(res_filename) and (not recomp)):
		# print('loading ' + res_filename)
		# E_data = np.load(res_filename)
		# E_arr = E_data['E_arr']
		# d_E_arr = E_data['d_E_arr']
		# M_arr = E_data['M_arr']
		# d_M_arr = E_data['d_M_arr']
		# C_fluct = E_data['C_fluct']
	# else:
		# print('computing E & M')
		# E_arr, d_E_arr, M_arr, d_M_arr, C_fluct = get_E_T(box, Nt, my_seed, T_arr)
		# np.savez(res_filename, E_arr=E_arr, d_E_arr=d_E_arr, M_arr=M_arr, d_M_arr=d_M_arr, C_fluct=C_fluct)
		# print('saved ' + res_filename)
		
	# if(ax_E is not None):
		# #fig_E, ax_E = my.get_fig('T', 'E', xscl='log')
		# ax_E.errorbar(T_arr, E_arr, yerr=d_E_arr, label=('box = ' + str(box)))
	# if(ax_M is not None):
		# ax_M.errorbar(T_arr, M_arr, yerr=d_E_arr, label=('box = ' + str(box)))
	
	# if(ax_M2 is not None):
		# T_small_ind = T_arr < 3.3
		# ax_M2.errorbar(T_arr[T_small_ind], np.power(M_arr[T_small_ind], 2), yerr=d_E_arr[T_small_ind], label=('box = ' + str(box)))

	# T_C_1, C_1 = get_deriv(T_arr, E_arr, degr=1, n_gap=3)
	# #T_C_3, C_3 = get_deriv(T_arr, E_arr, degr=3)
	
	# if(ax_C is not None):
		# #fig_C, ax_C = my.get_fig('T', 'C', yscl='log')
		# ax_C.plot(T_arr, C_fluct * box**2, label=('box = ' + str(box)))
		# #ax_C.plot(T_C_1, C_1, label=('box = ' + str(box)))
		# #ax_C.plot(T_C_3, C_3, label='d=3')
		# #ax_C.legend()
		
	# return T_arr

#import Izing_data

#T_table = 1 / Izing_data.data[:, 0]
#E_table = Izing_data.data[:, 1]

my_seed = 2
recomp = 0
mode = 1
to_get_EM = 0
verbose = 2

L = 11
Temp = 2.0   # T_c = 2.27
h = -0.010   # arbitrary small number that gives nice pictures

izing.init_rand(my_seed)
izing.set_verbose(verbose)

if(mode == 0):
	# -------- T < Tc, transitions ---------
	M0 = L**2   # this insludes the whole range [-L^2; L^2] into a single simulation
	N0 = 10

	# -------- T > Tc, fluctuations around 0 ---------
	#N0 = 30
	#Temp = 3.0
	#h = 0.05
	
	proc_T(L, Temp, h, N0, M0, my_seed, to_get_EM=to_get_EM)
elif(mode == 1):
	N_M_interfaces = 10
	M_0 = -L**2 + 20
	M_max = -M_0
	N_init_states = np.ones(N_M_interfaces + 2, dtype=np.intc).T * 10
	M_interfaces = np.array([-L**2 - 1] + list(M_0 + np.round(np.arange(N_M_interfaces) * (M_max - M_0) / (N_M_interfaces - 1) / 2) * 2) + [L**2], dtype=np.intc).T
	
	proc_FFS(L, Temp, h, N_init_states, M_interfaces)
elif(mode == 99):
	fig_E, ax_E = my.get_fig('T', '$E/L^2$', xscl='log')
	fig_M, ax_M = my.get_fig('T', '$M/L^2$', xscl='log')
	fig_C, ax_C = my.get_fig('T', '$C/L^2$', xscl='log', yscl='log')
	fig_M2, ax_M2 = my.get_fig('T', '$(M/L^2)^2$')
	
	T_arr = np.power(10, np.linspace(T_min_log10, T_max_log10, N_T))
	
	for n in N_s:
		T_arr = proc_N(n, Nt, my_seed, ax_C=ax_C, ax_E=ax_E, ax_M=ax_M, ax_M2=ax_M2, recomp=recomp, T_min_log10=T_min_log10, T_max_log10=T_max_log10, N_T=N_T)
	
	ax_E.plot(T_arr, -np.tanh(1/T_arr)*2, '--', label='$-2 th(1/T)$')
	T_simulated_ind = (min(T_arr) < T_table) & (T_table < max(T_arr))
	ax_E.plot(T_table[T_simulated_ind], E_table[T_simulated_ind], '.', label='Onsager')

	ax_C.legend()
	ax_E.legend()
	ax_M.legend()

plt.show()
