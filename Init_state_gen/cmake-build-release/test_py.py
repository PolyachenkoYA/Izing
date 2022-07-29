import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

import mylib as my

if(__name__ == "__main__"):
	to_recompile = True
	if(to_recompile):
		my.run_it('make izing.so')
		my.run_it('mv izing.so.cpython-38-x86_64-linux-gnu.so izing.so')
		print('recompiled izing')

	import izing
	#exit()

def log_norm_err(l, dl):
	"""Returns the confidence interval of a log-norm distributed value
	
	Parameters
	----------
	l : float
		The <log(x)> value
	dl : float
		std(<log(x)>), assuming <log(x)> is normally distributed

	Returns
	-------
	1: e^<log> ~= <x>
	2: e^(<log> - d_<log>)
	3: e^(<log> + d_<log>)
	4: e^<log> * d_<log> ~= dx
	"""
	return np.exp(l), np.exp(l - dl), np.exp(l + dl), np.exp(l) * dl

def proc_FFS(L, Temp, h, N_init_states_AB, N_init_states_BA, M_interfaces_AB, M_interfaces_BA, verbose=None, max_flip_time=None, to_plot_hists=True, init_gen_mode=-2, Nt_bf=15000000):
	"""Runs FFS from both ends and obtains the F profile

	Parameters
	----------
	L : int
		The linear size of the lattice
	
	Temp : float
		Temperature of the system, in J units, so it's T/J
	
	h : float
		Magnetic field (the 'h' in E = -h*sum(s_i)), actually h/J
	
	N_init_states_AB : np.array(N_M_interfaces + 1, int)
		The number of states to save on each interface [M_0, M_1, ..., M_N-1, L2]. 
		See 'M_interfaces' for more info
	N_init_states_BA : np.array(N_M_interfaces + 1, int)
		same as N_init_states_AB, but for BA process
	
	M_interfaces_AB : np.array(N_M_interfaces + 2, float)
		All the interfaces for the run. +2 because theare is an interface at -L2-1 and at +L2 to formally complete the picture
		So, for L=11, M_0 = -101, N_M_interfaces=10, the interfaces might be:
		-122, -101, -79, -57, -33, -11, 11, 33, 57, 79, 101, 121
	
	M_interfaces_BA : 
		same as N_init_states_AB, but for BA process
	
	verbose : int, (None)
		How hoisy the computation is, {0, 1, 2, 3} are now possible
		None means the verbosity will be taken from the global value, which is a part of the namespace program state in the 'izing' module
		A number means it will be passed to all the deeper functions and used imstead of the default-state value
	
	max_flip_time : double, (None)
		Used to estimate the equlibration time (to a local optimum), t_eq ~ L2 * flip_time
		None means it will be estimated from some physical reasoning
		A number will be used as given
	
	to_plot_hists : bool (True)
		whether or not to plot F profile and other related plots
	
	init_gen_mode : int (-2)
		The way to generate initial states in A to be propagated towards M_0
		-2 - generate an ensemble and in A and sample from this ensemble
		-1 - generate each spin randomly 50/50 to be +-1
		>=0 - generate all spins -1 and then set |mode| random (uniformly distributed over all lattice points) spins to +1
	

	Returns
	-------
	1: e^<log> ~= <x>
	2: e^(<log> - d_<log>)
	3: e^(<log> + d_<log>)
	4: e^<log> * d_<log> ~= dx
	"""
	probs_AB, d_probs_AB, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, rho_AB, d_rho_AB, M_hist_centers_AB, M_hist_lens_AB = \
		proc_FFS_AB(L, Temp, h, N_init_states_AB, M_interfaces_AB, verbose=verbose, max_flip_time=max_flip_time, to_get_EM=True, to_plot_time_evol=False, to_plot_hists=False, init_gen_mode=init_gen_mode)

	probs_BA, d_probs_BA, ln_k_BA, d_ln_k_BA, flux0_BA, d_flux0_BA, rho_BA, d_rho_BA, M_hist_centers_BA, M_hist_lens_BA = \
		proc_FFS_AB(L, Temp, -h, N_init_states_BA, M_interfaces_BA, verbose=verbose, max_flip_time=max_flip_time, to_get_EM=True, to_plot_time_evol=False, to_plot_hists=False, init_gen_mode=init_gen_mode)
	probs_BA = np.flip(probs_BA)
	d_probs_BA = np.flip(d_probs_BA)
	rho_BA = np.flip(rho_BA)
	d_rho_BA = np.flip(d_rho_BA)
	#M_hist_centers_BA = np.flip(M_hist_centers_BA)
	#M_hist_lens_BA = np.flip(M_hist_lens_BA)
	
	assert(np.all(abs(M_hist_centers_AB - M_hist_centers_BA) < 1 / L**2 * 1e-3))
	assert(np.all(abs(M_hist_lens_AB - M_hist_lens_BA) < 1 / L**2 * 1e-3))
	
	steps_used = 1   # 1 step, because [k] = 1/step, and 1 step is used to ln(k)
	k_AB, k_AB_low, k_AB_up, d_k_AB = log_norm_err(ln_k_AB, d_ln_k_AB)
	k_BA, k_BA_low, k_BA_up, d_k_BA = log_norm_err(ln_k_BA, d_ln_k_BA)
	
	k_min = min(k_AB, k_BA)
	k_max = max(k_AB, k_BA)
	ln_p_min = -np.log(1 + k_max / k_min)
	d_ln_p_min = np.sqrt(d_ln_k_AB**2 + d_ln_k_BA**2) / (1 + k_min / k_max)
	
	confidence_thr = 0.3
	if(d_ln_p_min > confidence_thr):
		print('WARNING: poor sampling for p_min, dp/p = ' + my.f2s(d_ln_p_min) + ' > ' + str(confidence_thr))
	
	p_min, p_min_low, p_min_up, d_p_min = log_norm_err(ln_p_min, d_ln_p_min)
	p_max = 1 - p_min
	p_max_low = 1 - p_min_up
	p_max_up = 1 - p_min_low
	d_p_max = d_p_min
	
	if(k_AB > k_BA):
		p_AB, p_AB_low, p_AB_up, d_p_AB = p_min, p_min_low, p_min_up, d_p_min
		p_BA, p_BA_low, p_BA_up, d_p_BA = p_max, p_max_low, p_max_up, d_p_max
	else:
		p_AB, p_AB_low, p_AB_up, d_p_AB = p_max, p_max_low, p_max_up, d_p_max
		p_BA, p_BA_low, p_BA_up, d_p_BA = p_min, p_min_low, p_min_up, d_p_min
	
	rho = p_AB * flux0_AB * rho_AB + p_BA * flux0_BA * rho_BA
	d_rho = np.sqrt((p_AB * flux0_AB * rho_AB)**2 * ((d_p_AB / p_AB)**2 + (d_flux0_AB / flux0_AB)**2 + (d_rho_AB / rho_AB)**2) + \
			  (p_BA * flux0_BA * rho_BA)**2 * ((d_p_BA / p_BA)**2 + (d_flux0_BA / flux0_BA)**2 + (d_rho_BA / rho_BA)**2))
	
	F = -Temp * np.log(rho * M_hist_lens_AB)
	d_F = Temp * d_rho / rho
	F = F - min(F)
	
	F_bf, d_F_bf, E_bf, M_hist_centers_bf, k_bc_AB, E_mean_bf, d_E_mean_bf, M_mean_bf, d_M_mean_bf, C_bf = \
		proc_T(L, Temp, h, Nt_bf, to_plot_time_evol=False, to_plot_F=False)
	
	if(to_plot_hists):
		fig_F, ax_F = my.get_fig(r'$m = M / L^2$', r'$F(m) = -T \ln(\rho(m) \cdot dm(m))$', title=r'$F(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_F.errorbar(M_hist_centers_AB, F, yerr=d_F, fmt='.', label='FFS data')
		ax_F.errorbar(M_hist_centers_bf, F_bf - min(F_bf), yerr=d_F_bf, fmt='.', label='BF data')
		ax_F.legend()

def proc_FFS_AB(L, Temp, h, N_init_states, M_interfaces, verbose=None, max_flip_time=None, to_get_EM=False, to_plot_time_evol=True, to_plot_hists=True, EM_stride=-3000, init_gen_mode=-2, Ms_alpha=0.5):
	print('=============== running h = %s ==================' % (my.f2s(h)))
	
	# py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> M_interfaces, int to_get_EM, std::optional<int> _verbose)
	# return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M);
	(_states, probs, d_probs, Nt, flux0, d_flux0, _E, _M) = izing.run_FFS(L, Temp, h, N_init_states, M_interfaces, verbose=verbose, to_get_EM=to_get_EM, init_gen_mode=init_gen_mode)
	
	L2 = L**2;
	N_M_interfaces = len(M_interfaces) - 2   # '-2' to be consistent with C++ implementation. 2 interfaces are '-L2-1' and '+L2'
	
	states = [_states[ : N_init_states[0] * L2].reshape((N_init_states[0], L, L))]
	for i in range(1, N_M_interfaces + 1):
		states.append(_states[np.sum(N_init_states[:i]) * L2 : np.sum(N_init_states[:(i+1)] * L2)].reshape((N_init_states[i], L, L)))
	del _states
	
	ln_k_AB = np.log(flux0 * 1) + np.sum(np.log(probs[1:-1]))   # [flux0 * 1] = 1, because [flux] = 1/time = 1/step
	d_ln_k_AB = np.sqrt((d_flux0 / flux0)**2 + np.sum((d_probs[1:-1] / probs[1:-1])**2))
	k_AB, k_AB_low, k_AB_up, d_k_AB = log_norm_err(ln_k_AB, d_ln_k_AB)
	
	print('Nt:', Nt)
	print('N_init_guess: ', np.exp(np.mean(np.log(Nt))) / Nt)
	print('flux0 = (%lf +- %lf) 1/step' % (flux0, d_flux0))
	print('-log10(P):', -np.log(probs) / np.log(10))
	print('d_P / P:', d_probs / probs)
	print('<k_AB> = ', my.f2s(k_AB), ' 1/step')   # probs[0] == 1 because its the probability to go from A to M_0
	if(d_ln_k_AB > 0.3):
		print('k_AB \\in [', my.f2s(k_AB_low), ';', my.f2s(k_AB_up), '] 1/step with 68% prob')
	else:
		print('d_<k_AB> = ', my.f2s(k_AB * d_ln_k_AB), ' 1/step')
	
	dE_step = 8   # the maximum dE possible for 1 step = 'change in a spin' * 'max neighb spin' = 2*4 = 8
	dM_step = 2   # the magnitude of the spin flip, from -1 to 1
	if(max_flip_time is None):
		#max_flip_time = np.exp(dE_step / Temp)
		# The upper-bound estimate on the number of steps for 1 flip to happen
		# This would be true if I had all the attempts (even unsuccesful ones) to flip
			
		max_flip_time = 1.0
		# we save only succesful flips, so <flip time> = 1
	M_std = 2   # appromixation, overestimation
	memory_time = max(1, (M_std * L2) / dM_step * max_flip_time)
	
	P_B = np.empty(N_M_interfaces)
	d_P_B = np.empty(N_M_interfaces)
	P_B[N_M_interfaces - 1] = 1
	d_P_B[N_M_interfaces - 1] = 0
	for i in range(N_M_interfaces - 1):
		_p = probs[(i + 1):(N_M_interfaces)]
		d__p = d_probs[(i + 1):(N_M_interfaces)]
		P_B[i] = np.prod(_p)
		d_P_B[i] = P_B[i] * np.sqrt(np.sum((d__p / _p)**2))

	rho_s = np.zeros((L2 + 1, N_M_interfaces + 1))
	d_rho_s = np.zeros((L2 + 1, N_M_interfaces + 1))
	timesteps = []
	M_hist_centers = []
	rho = None
	if(to_get_EM):
		_M = _M / L2
		_E = _E / L2
		M_interfaces = M_interfaces / L2
		
		Nt_total = len(_E)
		Nt_totals = np.empty(N_M_interfaces + 1, dtype=np.intc)
		Nt_totals[0] = Nt[0]
		E = [_E[ : Nt_totals[0]]]
		for i in range(1, N_M_interfaces + 1):
			Nt_totals[i] = Nt_totals[i-1] + Nt[i]
			E.append(_E[Nt_totals[i-1] : Nt_totals[i]])
		del _E
		assert(Nt_total == Nt_totals[-1])
		M = [_M[ : Nt_totals[0]]]
		for i in range(1, N_M_interfaces + 1):
			M.append(_M[Nt_totals[i-1] : Nt_totals[i]])
		del _M
		
		
		M_hist_edges = L2 + 1   # auto-build for edges
		M_hist_edges = (np.arange(L2 + 2) * 2 - (L2 + 1)) / L2
		# The number of bins cannot be arbitrary because M is descrete, thus it's a good idea if all the bins have 1 descrete value inside or all the bins have 2 or etc. 
		# It's not good if some bins have e.g. 1 possible M value, and others have 2 possible values. 
		# There are 'L^2 + 1' possible values of M, because '{M \in [-L^2; L^2]} and {dM = 2}'
		# Thus, we need edges which cover [-1-1/L^2; 1+1/L^2] with step=2/L^2
		M_hist_lens = (M_hist_edges[1:] - M_hist_edges[:-1])
		M_hist_centers = (M_hist_edges[1:] + M_hist_edges[:-1]) / 2

		rho = np.empty((L2 + 1, N_M_interfaces + 1))
		d_rho = np.zeros((L2 + 1, N_M_interfaces + 1))
		for i in range(N_M_interfaces + 1):
			timesteps.append(np.arange(Nt[i]) + (0 if(i == 0) else Nt_totals[i-1]))
			M_hist, _ = np.histogram(M[i], bins=M_hist_edges)
			M_hist = M_hist / memory_time
			M_hist_ok_inds = (M_hist > 0)
			#rho_s[M_hist_ok_inds, i] = M_hist[M_hist_ok_inds] / M_hist_lens[M_hist_ok_inds] / Nt[i]   # \pi(q, l_i)
			#d_rho_s[M_hist_ok_inds, i] = np.sqrt(M_hist[M_hist_ok_inds] * (1 - M_hist[M_hist_ok_inds] / Nt[i])) / M_hist_lens[M_hist_ok_inds] / Nt[i]
			rho_s[M_hist_ok_inds, i] = M_hist[M_hist_ok_inds] / M_hist_lens[M_hist_ok_inds] / N_init_states[i+1]   # \pi(q, l_i)
			d_rho_s[M_hist_ok_inds, i] = np.sqrt(M_hist[M_hist_ok_inds] * (1 - M_hist[M_hist_ok_inds] / Nt[i])) / M_hist_lens[M_hist_ok_inds] / N_init_states[i+1]
			
			rho[:, i] = rho_s[:, i] * (1 if(i == 0) else np.prod(probs[:i]))
			d_rho[M_hist_ok_inds, i] = rho[M_hist_ok_inds, i] * np.sqrt((d_rho_s[M_hist_ok_inds, i] / rho_s[M_hist_ok_inds, i])**2 + (0 if(i == 0) else np.sum((d_probs[:i] / probs[:i])**2)))
		
		rho = np.sum(rho, axis=1)
		d_rho = np.sqrt(np.sum(d_rho**2, axis=1))
		F = -Temp * np.log(rho * M_hist_lens)
		d_F = Temp * d_rho / rho
		F = F - min(F)
		
		# stab_step = int(min(L**2 * max_flip_time, Nt / 2)) * 5
		# # Each spin has a good chance to flip during this time
		# # This does not guarantee the global stable state, but it is sufficient to be sure-enough we are in the optimum of the local metastable-state.
		# # The approximation for a global ebulibration would be '1/k_AB', where k_AB is the rate constant. But it's hard to estimate on that stage.
		# stab_ind = (steps > stab_step)
		
		if(to_plot_time_evol):
			if(EM_stride < 0):
				EM_stride = np.int_(- Nt_total / EM_stride)
			
			fig_E, ax_E = my.get_fig('step', '$E / L^2$', title='E(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			fig_M, ax_M = my.get_fig('step', '$M / L^2$', title='M(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			for i in range(N_M_interfaces + 1):
				ax_E.plot(timesteps[i][::EM_stride], E[i][::EM_stride], label='data, i=%d' % (i))
				# ax_E.plot([stab_step] * 2, [min(E), max(E)], label='equilibr')
				# ax_E.plot([stab_step, Nt], [E_mean] * 2, label=('$<E> = ' + my.errorbar_str(E_mean, d_E_mean) + '$'))
				
				ax_M.plot(timesteps[i][::EM_stride], M[i][::EM_stride], label='data, i=%d' % (i))
				# ax_M.plot([stab_step] * 2, [min(M), max(M)], label='equilibr')
				# ax_M.plot([stab_step, Nt], [M_mean] * 2, label=('$<M> = ' + my.errorbar_str(M_mean, d_M_mean) + '$'))
			
			ax_E.legend()
			ax_M.legend()
		
		if(to_plot_hists):
			fig_Mhists, ax_Mhists = my.get_fig(r'$m = M / L^2$', r'$\rho_i(m) \cdot P(i | A)$', title=r'$\rho(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			for i in range(N_M_interfaces + 1):
				p_factor = (1 if(i == 0) else np.prod(probs[:i]))
				ax_Mhists.bar(M_hist_centers, rho_s[:, i] * p_factor, yerr=d_rho_s[:, i] * p_factor, width=M_hist_lens, align='center', label=str(i), alpha=Ms_alpha)
			for i in range(N_M_interfaces + 2):
				ax_Mhists.plot([M_interfaces[i]] * 2, [min(rho), max(rho)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_Mhists.legend()
			
			fig_F, ax_F = my.get_fig(r'$m = M / L^2$', r'$F(m) = -T \ln(\rho(m) \cdot dm(m))$', title=r'$F(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_F.errorbar(M_hist_centers, F, yerr=d_F, fmt='.', label='data')
			for i in range(N_M_interfaces + 2):
				ax_F.plot([M_interfaces[i]] * 2, [min(F), max(F)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_F.legend()
			
			fig_PB_log, ax_PB_log = my.get_fig(r'$m = M / L^2$', r'$P_B(m) = P(i|0)$', title=r'$P_B(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			fig_PB, ax_PB = my.get_fig(r'$m = M / L^2$', r'$P_B(m) = P(i|0)$', title=r'$P_B(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			ax_PB_log.errorbar(M_interfaces[1:-1], P_B, yerr=d_P_B, fmt='.', label='data')
			ax_PB.errorbar(M_interfaces[1:-1], P_B, yerr=d_P_B, fmt='.', label='data')
			
			ax_PB_log.legend()
			ax_PB.legend()
		
	return probs, d_probs, ln_k_AB, d_ln_k_AB, flux0, d_flux0, rho, d_rho, M_hist_centers, M_hist_lens

def exp_integrate(x, f):
	# integrates \int_{exp(f(x))dx}
	#dx = x[1:] - x[:-1]
	#b = (f[1:] - f[:-1]) / dx
	#return np.sum(np.exp(f[:-1]) * (np.exp(b * dx) - 1) / b)
	# This converges to the right answer, but for some reason 
	# this gives worse approximation than the regular central sum
	
	dx = x[1:] - x[:-1]
	f = np.exp(f)
	fc = (f[1:] + f[:-1]) / 2
	return np.sum(fc * dx)

def exp2_integrate(fit2, x1):
	# Integrate \int{fit2(x)}_{x1}^{x0}, where:
	# fit2(x) - quadratic 'fit2[0] * x^2 + fit2[1] * x + fit2[2]'
	# It's used if the form 'c + a*(x-x0)^2'
	# x0 = peak of fit2(x)
	# x1 < x0
	# a > 0
	
	a = fit2[0]
	assert(a > 0)
	
	x0 = -fit2[1] / (2 * a)
	assert(x0 > x1)
	
	c = fit2[2] - a * x0**2
	
	return np.exp(c) / 2 * np.sqrt(np.pi / a) * scipy.special.erfi((x0 - x1) * np.sqrt(a))

def proc_T(L, Temp, h, Nt, verbose=None, max_flip_time=None, to_plot_time_evol=True, to_plot_F=True, M_peak_guess=0, M_fit2_width=0.5, EM_stride=-3000):
	(E, M) = izing.run_bruteforce(L, Temp, h, Nt, verbose=verbose)
	
	L2 = L**2
	M = M / L2
	E = E / L2
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
	
	M_hist_edges = (np.arange(L2 + 2) * 2 - (L2 + 1)) / L2
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
	
	F_m = F[M_hist_centers < 0]
	F_p = F[M_hist_centers > 0]
	F_min_m_ind = np.argmin(F_m)
	F_min_p_ind = np.argmin(F_p)
	M_min_m = M_hist_centers[M_hist_centers < 0][F_min_m_ind]
	M_min_p = M_hist_centers[M_hist_centers > 0][F_min_p_ind]
	F_min_m = F_m[F_min_m_ind]
	F_min_p = F_p[F_min_p_ind]
	sanity_k = abs((F_min_m - F_min_p) / (h*(M_min_p - M_min_m)*L2))
	
	E_avg = np.empty(L2 + 1)
	for i in range(len(M_hist_centers)):
		E_for_avg = E[(M_hist_edges[i] < M) & (M < M_hist_edges[i+1])] * L2
		E_avg[i] = np.average(E_for_avg, weights=np.exp(- E_for_avg / Temp))
	E_avg = E_avg - E_avg[0]   # we can choose the E constant
	S = (E_avg - F) / Temp
	S0 = S[0]   # S[m=+-1] = 0 because S = log(N), and N(m=+-1)=1
	S = S - S0
	F = F + S0 * Temp
	
	M_fit2_Mmin_ind = np.argmax(M_hist_centers > M_peak_guess - M_fit2_width)
	assert(M_fit2_Mmin_ind > 0), 'issues finding analytical border'
	M_fit2_inds = (M_fit2_Mmin_ind <= np.arange(len(M_hist_centers))) & (M_hist_centers < M_peak_guess + M_fit2_width)
	M_fit2 = np.polyfit(M_hist_centers[M_fit2_inds], F[M_fit2_inds], 2, w = 1/d_F[M_fit2_inds])
	M_peak = -M_fit2[1] / (2 * M_fit2[0])
	M_peak_ind = np.argmin(abs(M_hist_centers - M_peak))
	F_max = np.polyval(M_fit2, M_peak)
	#M_peak_ind = np.argmin(M_hist_centers - M_peak)
	bc_Z = exp_integrate(M_hist_centers[ : M_fit2_Mmin_ind + 1], -F[ : M_fit2_Mmin_ind + 1] / Temp) + exp2_integrate(- M_fit2 / Temp, M_hist_centers[M_fit2_Mmin_ind])
	k_bc_AB = (np.mean(M_hist_centers[M_peak_ind + 1] - M_hist_centers[M_peak_ind - 1])/2 / 2) * (np.exp(-F_max / Temp) / bc_Z)
	
	print('k_bc_AB =', my.f2s(k_bc_AB))
	
	C = E_std**2 / Temp**2 * L**2

	d_E_mean = E_std / np.sqrt(Nt_stab / memory_time)
	d_M_mean = M_std / np.sqrt(Nt_stab / memory_time)
	
	if(to_plot_time_evol):
		if(EM_stride < 0):
			EM_stride = np.int_(- Nt / EM_stride)
		
		fig_E, ax_E = my.get_fig('step', '$E / L^2$', title='E(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_E.plot(steps[::EM_stride], E[::EM_stride], label='data')
		ax_E.plot([stab_step] * 2, [min(E), max(E)], label='equilibr')
		ax_E.plot([stab_step, Nt], [E_mean] * 2, label=('$<E> = ' + my.errorbar_str(E_mean, d_E_mean) + '$'))
		ax_E.legend()
		
		fig_M, ax_M = my.get_fig('step', '$M / L^2$', title='M(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_M.plot(steps[::EM_stride], M[::EM_stride], label='data')
		ax_M.plot([stab_step] * 2, [min(M), max(M)], label='equilibr')
		ax_M.plot([stab_step, Nt], [M_mean] * 2, label=('$<M> = ' + my.errorbar_str(M_mean, d_M_mean) + '$'))
		ax_M.legend()
		
	if(to_plot_F):
		fig_Mhist, ax_Mhist = my.get_fig(r'$m = M / L^2$', r'$\rho(m)$', title=r'$\rho(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_Mhist.bar(M_hist_centers, rho, yerr=d_rho, width=M_hist_lens, align='center')
		#ax_Mhist.legend()
		
		fig_F, ax_F = my.get_fig(r'$m = M / L^2$', r'$E / J$', title=r'$F(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h) + '; $|\Delta F_{min, \pm}| / |\Delta M_{min, \pm} \cdot h| = ' + my.f2s(sanity_k) + '$')
		ax_F.errorbar(M_hist_centers, F, yerr=d_F, fmt='.', label='F(m)')
		ax_F.plot(M_hist_centers[M_fit2_inds], np.polyval(M_fit2, M_hist_centers[M_fit2_inds]), label='fit2; $M_0 = ' + my.f2s(M_peak) + '$')
		ax_F.plot(M_hist_centers, E_avg, '.', label='<E>(m)')
		ax_F.plot(M_hist_centers, S * Temp, '.', label='$T \cdot S(m)$')
		ax_F.legend()


	return F, d_F, E, M_hist_centers, k_bc_AB, E_mean, d_E_mean, M_mean, d_M_mean, C

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

def get_init_states(Ns, N_min, N0=1):
	# first value is '1' because it's the size of the states' set from which the initial states for simulations in A will be picked. 
	# Currently we start all the simulations from 'all spins -1', so there is really only 1 initial state, so values >1 will just mean storing copies of the same state
	return np.array([N0] + list(Ns / min(Ns) * N_min), dtype=np.intc)

def main():
	my_seed = 2
	recomp = 0
	mode = 'FFS'
	to_get_EM = 1
	verbose = 1

	L = 11
	h = -0.010   # arbitrary small number that gives nice pictures

	# -------- T < Tc, transitions ---------
	Temp = 2.0   # T_c = 2.27
	# -------- T > Tc, fluctuations around 0 ---------
	#Temp = 3.0
	#Temp = 2.5   # looks like Tc for L=16

	izing.init_rand(my_seed)
	izing.set_verbose(verbose)

	if(mode == 'BF'):
		Nt = 15000000   # enough sampling grows rapidly with L. 5e6 is enought for L=11, but not for >= 13
		proc_T(L, Temp, h, Nt, to_plot_time_evol=True)
		# ===== h=-0.01, T=2.0, L=11 ========
		# k_AB = 400 e-7 1/step
		# k_BA = 760 e-7 1/step
		
	elif((mode == 'FFS') or (mode == 'FFS_AB')):
		init_gen_mode = -2
		N_M_interfaces = 30
		M_0 = -L**2 + 4
		M_max = -M_0
		Nt_narrowest = 3000
		
		# ======== Ni=10, M_0=-101 ==========
		N_init_states_AB = np.array([5.86494782, 0.20780331, 0.64108675, 0.45093819, 0.59296037, 0.63323846, 0.82477221, 1.02855918, 1.90803348, 2.81926314, 1.65642474]) * \
						np.array([0.97378795, 1.06975615, 1.02141394, 0.97481339, 0.93258517, 1.05343075, 0.94876457, 0.96916528, 0.9942971, 0.98931741, 1.08498797]) * \
						np.array([34.44184113, 3.56362745, 0.21981169, 0.3781434, 0.3940806, 0.51355653, 0.62098097, 0.82222466, 0.85768272, 0.75359971, 1.46758935]) * \
						np.array([0.98588286, 1.04962581, 0.99399027, 1.00544771, 1.03247692, 0.99962003, 1.01109933, 1.02817106, 0.91720793, 0.96675774, 1.01633903])
			# for h=-0.01, T=2.0, L=11, Ni=10, M_0=-101
			
		N_init_states_BA = np.array([5.04242695, 0.24306255, 0.64430179, 0.46376104, 0.55123104, 0.67255364, 0.857448, 1.02944648, 1.75315376, 2.67029485, 1.78240891]) * \
						np.array([1.03900965, 1.02364505, 0.97349883, 0.94509095, 0.9860851, 0.95276988, 0.97621008, 1.03575653, 1.06265366, 1.02110462, 0.9914176]) * \
						np.array([31.83003551, 3.57493802, 0.23064763, 0.37330525, 0.42065408, 0.55995892, 0.63625423, 0.82914687, 0.7983409, 0.70740199, 1.45439774]) * \
						np.array([1.01404907, 0.95053083, 1.01731895, 1.01836105, 0.9710529, 0.99439959, 0.98608457, 1.00517813, 1.01599291, 1.01267056, 1.01694163])
						
			# for h=+0.01, T=2.0, L=11, Ni=10, M_0=-101
		
		# ======== Ni=10, M_0=-117 ==========
		N_init_states_AB = np.array([5.86494782, 0.20780331, 0.64108675, 0.45093819, 0.59296037, 0.63323846, 0.82477221, 1.02855918, 1.90803348, 2.81926314, 1.65642474]) * \
						np.array([0.97378795, 1.06975615, 1.02141394, 0.97481339, 0.93258517, 1.05343075, 0.94876457, 0.96916528, 0.9942971, 0.98931741, 1.08498797]) * \
						np.array([0.98588286, 1.04962581, 0.99399027, 1.00544771, 1.03247692, 0.99962003, 1.01109933, 1.02817106, 0.91720793, 0.96675774, 1.01633903])
			# for h=-0.01, T=2.0, L=11, Ni=10, M_0=-101
		
		N_init_states_BA = np.array([5.04242695, 0.24306255, 0.64430179, 0.46376104, 0.55123104, 0.67255364, 0.857448, 1.02944648, 1.75315376, 2.67029485, 1.78240891]) * \
						np.array([1.03900965, 1.02364505, 0.97349883, 0.94509095, 0.9860851, 0.95276988, 0.97621008, 1.03575653, 1.06265366, 1.02110462, 0.9914176]) * \
						np.array([1.01404907, 0.95053083, 1.01731895, 1.01836105, 0.9710529, 0.99439959, 0.98608457, 1.00517813, 1.01599291, 1.01267056, 1.01694163])
			# for h=+0.01, T=2.0, L=11, Ni=10, M_0=-101

		# ======== Ni=20, M_0=-101 ==========
		#N_init_states_AB = np.array([2.33868958, 0.51789641, 1.42604944, 1.2149675, 0.94619927, 0.5871483, 0.69429377, 0.66489963, 0.51122314, 0.69779038, 0.72645796, 0.77746991, 0.67691482, 0.9581531, 0.98233807, 1.0223547, 1.62734606, 1.98560952, 1.95467, 2.9860634, 0.75066852])
			# for h=-0.01, T=2.0, L=11, Ni=20, M_0=-101
		
		#N_init_states_BA = np.array([2.17348042, 0.48972788, 1.40017212, 1.20991913, 0.9091747, 0.66442213, 0.70013649, 0.62187844, 0.56822718, 0.63563581, 0.77455908, 0.80039118, 0.68343128, 0.86287252, 1.02853796, 0.99677704, 1.70917241, 2.13442205, 2.27402395, 2.51349462, 0.74694019])
			# for h=+0.01, T=2.0, L=11, Ni=20, M_0=-101
			
		# ======== Ni=20, M_0=-117 ==========
		N_init_states_AB = np.array([104.58137675, 4.52651586, 0.9389709, 0.43419338, 0.26126731, 0.32707436, 0.33881547, 0.35159559, 0.37967477, 0.43670435, 0.42026171, 0.581579, 0.61943924, 0.68033028, 0.879528, 1.3003374, 1.5022591, 1.99221445, 2.59343701, 2.24145835, 1.49785133])
			# for h=-0.01, T=2.0, L=11, Ni=20, M_0=-117
		
		N_init_states_BA = np.array([1.80245815, 5.05678955, 1.10000627, 0.4953472, 0.31701429, 0.38355781, 0.40640283, 0.42864111, 0.47713556, 0.5551159, 0.48965043, 0.61552494, 0.79758504, 0.93919693, 1.24596722, 1.70109637, 1.90688393, 2.65422979, 3.17230448, 2.68977524, 1.73665982])
			# for h=+0.01, T=2.0, L=11, Ni=20, M_0=-117
		
		# ======== Ni=30, M_0=-117 ==========
		N_init_states_AB = np.array([59.23722756, 9.19233736, 2.61266382, 1.21618065, 0.76656654, 0.61166114, 0.57317545, 0.51439522, 0.45957972, 0.45246327, 0.43193812, 0.43959542, 0.45809447, 0.44831959, 0.51318388, 0.38675889, 0.49759893, 0.52024278, 0.58332438, 0.65210667, 0.70142344, 0.78572439, 1.10507602, 1.35863037, 1.45500263, 1.50165668, 2.05613283, 2.13398826, 2.29724441, 1.77473994, 0.81564219]) * \
							np.array([1.01134707, 1.01668083, 0.99770083, 0.99693929, 1.02841981, 1.03022968, 0.9648083, 1.00411582, 1.00274245, 0.98991819, 1.00155996, 0.99328857, 0.93081579, 1.02280934, 0.94989452, 0.98469081, 1.06015326, 1.05398132, 0.99421305, 0.99792504, 1.01645643, 1.16205045, 0.84636068, 0.86532691, 0.96942996, 1.14203607, 0.92927976, 1.05460351, 0.99708638, 1.01788432, 1.02577104])
			# for h=-0.01, T=2.0, L=11, Ni=20, M_0=-117
		
		N_init_states_BA = np.array([55.55835609, 8.52465298, 2.38771005, 1.16204589, 0.78008274, 0.59651014, 0.54800459, 0.44661705, 0.44707403, 0.39778446, 0.41945624, 0.40592288, 0.42940058, 0.43707759, 0.45712177, 0.443849, 0.55194654, 0.55967198, 0.62966159, 0.69834416, 0.80396093, 0.96261902, 1.05299106, 1.39556854, 1.53041928, 1.53895228, 2.03392523, 2.28837613, 2.37871075, 1.73388344, 0.829685]) * \
							np.array([0.00470185, 1.20204266, 1.2353155, 1.1844567, 1.13462661, 1.21798712, 1.14197801, 1.30594286, 1.2326769, 1.2785938, 1.17933553, 1.32219787, 1.162985, 1.2492839, 1.19178171, 1.08671672, 1.1178604, 1.36132457, 1.21534674, 1.17862562, 1.07802381, 1.11321367, 1.17841559, 1.06123281, 1.17551723, 1.32906336, 1.20508691, 1.12206798, 1.20344888, 1.22687804, 1.24334004])
			# for h=+0.01, T=2.0, L=11, Ni=20, M_0=-117

		N0 = 1000000
		N0 = 1000
		N_init_states_AB = get_init_states(N_init_states_AB, Nt_narrowest, N0)
		N_init_states_BA = get_init_states(N_init_states_BA, Nt_narrowest, N0)
		
		#N_init_states_BA = np.ones(N_M_interfaces + 2, dtype=np.intc) * Nt_narrowest
		#N_init_states_AB = np.ones(N_M_interfaces + 2, dtype=np.intc) * Nt_narrowest
		
		print('Ns_AB:', N_init_states_AB)
		print('Ns_BA:', N_init_states_BA)
		M_interfaces = M_0 + np.round(np.arange(N_M_interfaces) * (M_max - M_0) / (N_M_interfaces - 1) / 2) * 2
		M_interfaces_AB = np.array([-L**2 - 1] + list(M_interfaces) + [L**2], dtype=np.intc)
		M_interfaces_BA = np.array([-L**2 - 1] + list(-np.flip(M_interfaces)) + [L**2], dtype=np.intc)
		# this gives Ms such that there are always pairs +-M[i], so flipping this does not move the intefraces, which is (is it?) good for backwords FFS (B->A)
		
		if(mode == 'FFS'):
			proc_FFS(L, Temp, h, N_init_states_AB, N_init_states_BA, M_interfaces_AB, M_interfaces_BA)
			# ===== h=-0.01, T=2.0, L=11 ========
			# k_FFS_AB = (3.45 +- 0.50) e-7 (1/step)
			# k_FFS_BA = (10.0 +- 2.1 ) e-7 (1/step)
		else:
			proc_FFS_AB(L, Temp, h, N_init_states_AB, M_interfaces_AB, to_get_EM=to_get_EM)

	elif(mode == 'XXX'):
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

if(__name__ == "__main__"):
	main()
