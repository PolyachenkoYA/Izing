import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

import mylib as my

to_recompile = True
if(to_recompile):
	my.run_it('make izing.so')
	my.run_it('mv izing.so.cpython-38-x86_64-linux-gnu.so izing.so')

import izing
#exit()

def log_norm_err(l, dl):
	return np.exp(l), np.exp(l - dl), np.exp(l + dl), np.exp(l) * dl

def proc_FFS(L, Temp, h, N_init_states_AB, N_init_states_BA, M_interfaces_AB, M_interfaces_BA, verbose=None, max_flip_time=None, to_plot_hists=True, init_gen_mode=-2):
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
	
	if(to_plot_hists):
		fig_F, ax_F = my.get_fig(r'$m = M / L^2$', r'$F(m) = -T \ln(\rho(m))$', title=r'$F(m)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_F.errorbar(M_hist_centers_AB, F, yerr=d_F, fmt='.', label='data')


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
	print('k_AB \\in [', my.f2s(k_AB_low), ';', my.f2s(k_AB_up), '] 1/step with 68% prob')
	
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
			rho_s[M_hist_ok_inds, i] = M_hist[M_hist_ok_inds] / M_hist_lens[M_hist_ok_inds] / Nt[i]   # \pi(q, l_i)
			d_rho_s[M_hist_ok_inds, i] = np.sqrt(M_hist[M_hist_ok_inds] * (1 - M_hist[M_hist_ok_inds] / Nt[i])) / M_hist_lens[M_hist_ok_inds] / Nt[i]
			
			rho[:, i] = rho_s[:, i] * (1 if(i == 0) else np.prod(probs[:i]))
			d_rho[M_hist_ok_inds, i] = rho[M_hist_ok_inds, i] * np.sqrt((d_rho_s[M_hist_ok_inds, i] / rho_s[M_hist_ok_inds, i])**2 + (0 if(i == 0) else np.sum((d_probs[:i] / probs[:i])**2)))
		
		rho = np.sum(rho, axis=1)
		d_rho = np.sqrt(np.sum(d_rho**2, axis=1))
		F = -Temp * np.log(rho * M_hist_lens)
		d_F = Temp * d_rho / rho
		F = F - F[0]
		
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

	return probs, d_probs, ln_k_AB, d_ln_k_AB, flux0, d_flux0, rho, d_rho, M_hist_centers, M_hist_lens

def exp_integrate(x, f):
	# integrates \int_{exp(f(x))dx}
	dx = x[1:] - x[:-1]
	b = (f[1:] - f[:-1]) / dx
	return np.sum(np.exp(f[:-1]) * (np.exp(b * dx) - 1) / b)

def exp2_integrate(fit2, x1):
	# integrate \int{fit2(x)}_{x1}^{x0}, where:
	# fit2(x) - quadratic 'c + a*(x-x0)^2'
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
	F_max = np.polyval(M_fit2, M_peak)
	#M_peak_ind = np.argmin(M_hist_centers - M_peak)
	bc_Z = exp_integrate(M_hist_centers[ : M_fit2_Mmin_ind + 1], -F[ : M_fit2_Mmin_ind + 1] / Temp) + exp2_integrate(- M_fit2 / Temp, M_hist_centers[M_fit2_Mmin_ind])
	k_bc_AB = np.exp(-F_max / Temp) / 2 / L**2 / bc_Z
	
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
		
	# elif(mode == 'FFS_AB'):
		# N_M_interfaces = 10
		# M_0 = -L**2 + 20
		# M_max = -M_0
		# Nt_narrowest = 20
		
		# N_init_states = np.ones(N_M_interfaces + 2, dtype=np.intc) * Nt_narrowest
		
		# # N_init_states = np.array([1.23969588e+02, 5.84450113e+00, 5.50214644e-01, 1.37587470e-01, 6.36924888e-02, 4.42375796e-02, 4.75439693e-02, 1.67227416e-01, 3.56319211e-01, 6.57449723e+01, 3.47409817e+01]) * \
						# # np.array([1.34723054, 1.37957324, 2.10078518, 1.95356022, 1.53971505, 1.74875067, 1.54874392, 1.22188151, 2.28112138, 0.03512974, 0.32107357]) * \
						# # np.array([0.94354313, 0.95265428, 0.94368868, 0.90844004, 0.84067221, 1.29158654, 0.23381741, 2.39335866, 0.73958596, 1.85306872, 1.55834244]) * \
						# # np.array([1.32609833, 1.33163682, 1.27471986, 1.16235763, 0.93947925, 0.70587731, 1.00300579, 0.76772614, 1.99785438, 0.59193796, 0.63287473]) * \
						# # np.array([0.95995858, 0.98238746, 0.98080595, 0.81573092, 1.48149265, 0.74719818, 6.31016512, 0.6011239, 0.48058931, 0.76629058, 0.85709126])
			# # for h=+0.01, T=2.0, L=11
		
		# N_init_states = np.array([1.00000000e+00, 5.38064219e-02, 8.14175284e-03, 2.37053429e-03, 1.48518289e-03, 9.50170981e-04, 2.18241023e-03, 5.31478411e-03, 2.19227675e-02, 1.64862731e-02, 3.85026277e-01]) * \
						# np.array([2.54034094,  2.62280884,  2.86598065,  2.19024415,  2.09899753, 2.51145122,  2.30368486,  2.09866929,  1.        , 46.29191244, 1.47942745]) * \
						# np.array([0.72173372, 0.72056242, 0.69804581, 0.74477665, 1.03137778, 0.70481042, 0.77929911, 0.99110226, 52.87379662, 0.1431188, 0.87054553]) * \
						# np.array([1.16575118, 1.14464192, 1.3104362, 1.50333849, 0.64985351, 1.5033924, 1.49060548, 0.37652677, 0.03124324, 13.08451377, 1.69703905]) * \
						# np.array([1.15945569, 1.13287169, 1.2031427, 0.96265214, 0.92756198, 2.13027455, 1.21681243, 1.54634793, 1.0057098, 0.1562718, 1.12490521]) * \
						# np.array([1.09514031, 1.16373809, 0.92639821, 1.2773923, 2.29584632, 0.64151696, 0.54354285, 1.18696861, 0.98060184, 0.8953073, 0.79481157])
			# # for h=-0.01, T=2.0, L=11
						
		# N_init_states = get_init_states(N_init_states, Nt_narrowest)
		# # first value is '1' because it's the size of the states' set from which the initial states for simulations in A will be picked. 
		# # Currently we start all the simulations from 'all spins -1', so there is really only 1 initial state, so values >1 will just mean storing copies of the same state
		
		# print(N_init_states)
		# M_interfaces = np.array([-L**2 - 1] + list(M_0 + np.round(np.arange(N_M_interfaces) * (M_max - M_0) / (N_M_interfaces - 1) / 2) * 2) + [L**2], dtype=np.intc)
		
		# proc_FFS_AB(L, Temp, h, N_init_states, M_interfaces, to_get_EM=to_get_EM)
		
	elif((mode == 'FFS') or (mode == 'FFS_AB')):
		init_gen_mode = -2
		N_M_interfaces = 10
		M_0 = -L**2 + 20
		M_max = -M_0
		Nt_narrowest = 20
		
		N_init_states_AB = np.array([1.00000000e+00, 5.38064219e-02, 8.14175284e-03, 2.37053429e-03, 1.48518289e-03, 9.50170981e-04, 2.18241023e-03, 5.31478411e-03, 2.19227675e-02, 1.64862731e-02, 3.85026277e-01]) * \
						np.array([2.54034094,  2.62280884,  2.86598065,  2.19024415,  2.09899753, 2.51145122,  2.30368486,  2.09866929,  1.        , 46.29191244, 1.47942745]) * \
						np.array([0.72173372, 0.72056242, 0.69804581, 0.74477665, 1.03137778, 0.70481042, 0.77929911, 0.99110226, 52.87379662, 0.1431188, 0.87054553]) * \
						np.array([1.16575118, 1.14464192, 1.3104362, 1.50333849, 0.64985351, 1.5033924, 1.49060548, 0.37652677, 0.03124324, 13.08451377, 1.69703905]) * \
						np.array([1.15945569, 1.13287169, 1.2031427, 0.96265214, 0.92756198, 2.13027455, 1.21681243, 1.54634793, 1.0057098, 0.1562718, 1.12490521]) * \
						np.array([2.46461917, 1.60269683, 1.58954112, 0.80877512, 0.88861785, 0.88339542, 1.62429312, 0.23827976, 2.53041583, 0.6168911, 0.4152189]) * \
						np.array([1.00880439, 1.06335813, 1.0055039, 1.14999435, 0.82599592, 0.69612143, 0.69493302, 3.1353025, 0.22461961, 1.60766293, 1.78199319]) * \
						np.array([1.15465563, 1.16941404, 1.10697519, 1.32831599, 0.9694735, 3.12589196, 0.65791752, 0.96987609, 0.76162631, 0.3251356, 1.05180651]) * \
						np.array([0.85845614, 0.8430911, 0.79074422, 0.80450109, 1.15593025, 0.29374146, 1.01873169, 0.86851483, 4.94519356, 2.91226124, 0.50199242]) * \
						np.array([1.09514031, 1.16373809, 0.92639821, 1.2773923, 2.29584632, 0.64151696, 0.54354285, 1.18696861, 0.98060184, 0.8953073, 0.79481157])
			# for h=-0.01, T=2.0, L=11, Ni=10
		
		N_init_states_BA = np.array([1.23969588e+02, 5.84450113e+00, 5.50214644e-01, 1.37587470e-01, 6.36924888e-02, 4.42375796e-02, 4.75439693e-02, 1.67227416e-01, 3.56319211e-01, 6.57449723e+01, 3.47409817e+01]) * \
						np.array([1.34723054, 1.37957324, 2.10078518, 1.95356022, 1.53971505, 1.74875067, 1.54874392, 1.22188151, 2.28112138, 0.03512974, 0.32107357]) * \
						np.array([0.94354313, 0.95265428, 0.94368868, 0.90844004, 0.84067221, 1.29158654, 0.23381741, 2.39335866, 0.73958596, 1.85306872, 1.55834244]) * \
						np.array([1.32609833, 1.33163682, 1.27471986, 1.16235763, 0.93947925, 0.70587731, 1.00300579, 0.76772614, 1.99785438, 0.59193796, 0.63287473]) * \
						np.array([0.95995858, 0.98238746, 0.98080595, 0.81573092, 1.48149265, 0.74719818, 6.31016512, 0.6011239, 0.48058931, 0.76629058, 0.85709126])
			# for h=+0.01, T=2.0, L=11, Ni=10
		
		#N_init_states_AB = np.array([1.47642942e+02, 1.22267366e-01, 2.02666403e-02, 7.67226493e-02, 3.56263559e+01]) * \
		#					np.array([0.99277459, 1.17296026, 1.18634628, 0.7368969, 0.98230838]) * \
		#					np.array([1.43402985, 1.52278981, 1.55833454, 1.27131829, 0.23114625])
		
			# for h=-0.01, T=2.0, L=11, Ni=4

		N0 = 1000000
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
		else:
			proc_FFS_AB(L, Temp, h, N_init_states_AB, M_interfaces, to_get_EM=to_get_EM)

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
