import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import sys
import scipy.interpolate

import mylib as my

# ========================== general params ==================
dE_avg = 6   # J
OP_min_default = {}
OP_max_default = {}
OP_peak_default = {}
OP_scale = {}
dm_step = {}
m_std_overestimate = {}
hist_edges_default = {}

OP_C_id = {}
OP_C_id['M'] = 0
OP_C_id['CS'] = 1

OP_step = {}
OP_step['M'] = 2
OP_step['CS'] = 1

title = {}
title['M'] = r'$m = M / L^2$'
title['CS'] = r'cluster size'

peak_guess = {}
peak_guess['M'] = 0
peak_guess['CS'] = 60

fit2_width = {}
fit2_width['M'] = 0.25
fit2_width['CS'] = 25

feature_label = {}
feature_label['M'] = 'm'
feature_label['CS'] = 'Scl'


def set_OP_defaults(L2):
	OP_min_default['M'] = -L2 - 1
	OP_max_default['M'] = L2 + 1
	OP_peak_default['M'] = L2 % 2
	OP_scale['M'] = L2
	dm_step['M'] = OP_step['M'] / OP_scale['M']
	m_std_overestimate['M'] = 2 * L2
	
	OP_min_default['CS'] = -1
	OP_max_default['CS'] = L2 + 1
	OP_peak_default['CS'] = L2 // 2
	OP_scale['CS'] = 1
	dm_step['CS'] = OP_step['CS'] / OP_scale['CS']
	m_std_overestimate['CS'] = L2

# ========================== recompile ==================
if(__name__ == "__main__"):
	to_recompile = True
	if(to_recompile):
		my.run_it('make izing.so')
		my.run_it('mv izing.so.cpython-38-x86_64-linux-gnu.so izing.so')
		print('recompiled izing')

	import izing
	#exit()

# ========================== functions ==================

def get_OP_hist_edges(interface_mode, OP_min=None, OP_max=None):
	OP_min = OP_min_default[interface_mode] if(OP_min is None) else max(OP_min, OP_min_default[interface_mode])
	OP_max = OP_max_default[interface_mode] if(OP_max is None) else min(OP_max, OP_max_default[interface_mode] - 1)
	
	return (np.arange(OP_min + 1 - OP_step[interface_mode] / 2, OP_max + OP_step[interface_mode] * (1/2 + 1e-3), OP_step[interface_mode])) / OP_scale[interface_mode]
	#return      (np.arange( + 1 - OP_step[i_m_def] / 2,  + OP_step[i_m_def] * (1/2 + 1e-3), OP_step[i_m_def]) if(i_m == i_m_def) \
	#		else np.arange(            OP_min_default[i_m_def]  + 1 - OP_step[i_m_def] / 2,             OP_max_default[i_m_def] - 1  + OP_step[i_m_def] * (1/2 + 1e-3), OP_step[i_m_def])) / OP_scale[i_m_def]
#	These formula assume:
#		1. We want edges at [x_min - step/2; .....; x_max + step/2]
#		2. OP_min/max_default = x_min/max -+ 1; '-+1' is independent of 'interface_mode' because we consider only integer OPs, and 1 is the minimal difference, so it's actually ~ '-+eps'
#		3. OP_min = x_min - 1 ; -1 ~ -eps, which reprensenes '(...' interval end
#		4. OP_max = x_max; which reprensenes '...]' interval end
#		5. 1e-3 * step is to include the last value since python (and numpy) do the ranges in the '(...]' way

def get_log_errors(l, dl, lbl=None, err0=0.3, units='1/step'):
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
	x = np.exp(l)
	x_low = np.exp(l - dl)
	x_up = np.exp(l + dl)
	d_x = x * dl
	
	if(lbl is not None):
		if(dl > err0):
			print(lbl + ' \\in [' + '; '.join(my.same_scale_strs([x_low, x_up], x_to_compare_to=[x_up - x_low])) + '] ' + units + ' with ~68% prob')
		else:
			print('%s = (%s) %s, dx/x ~ %s' % (lbl, my.errorbar_str(x, d_x), units, my.f2s(dl)))
	
	return x, x_low, x_up, d_x

def get_average(x_data, axis=0, mode='lin'):
	N_runs = x_data.shape[axis]
	if(mode == 'log'):
		x = np.exp(np.mean(np.log(x_data), axis=axis))
		d_x = x * np.std(np.log(x_data), axis=axis) / np.sqrt(N_runs - 1)   # dx = x * d_lnx = x * dx/x
	else:
		x = np.mean(x_data, axis=axis)
		d_x = np.std(x_data, axis=axis) / np.sqrt(N_runs - 1)
	return x, d_x

def plot_k_distr(k_data, N_bins, lbl, units=None):
	N_runs = len(k_data)
	k_hist, k_edges = np.histogram(k_data, bins=N_bins)
	d_k_hist = np.sqrt(k_hist * (1 - k_hist / N_runs))
	k_centers = (k_edges[1:] + k_edges[:-1]) / 2
	k_lens = k_edges[1:] - k_edges[:-1]
	
	units_inv_str = ('' if(units is None) else (' $(' + units + ')^{-1}$'))
	units_str = ('' if(units is None) else (' $(' + units + ')$'))
	fig, ax = my.get_fig(r'$%s$%s' % (lbl, units_inv_str), r'$\rho$%s' % (units_str), r'$\rho(%s)$' % (lbl))
	
	ax.bar(k_centers, k_hist / k_lens, yerr=d_k_hist / k_lens, width=k_lens, align='center')
	
def flip_M(M, L2, interface_mode):
	if(interface_mode == 'M'):
		return -np.flip(M)
	elif(interface_mode == 'CS'):
		return L2 - np.flip(M)

def get_sigmoid_fit(PB, d_PB, m, h, xi=5, fit_w=0.1):
	PB_sigmoid = -np.log(1 / PB - 1)
	d_PB_sigmoid = d_PB / (PB * (1 - PB))
	small_m_inds = (-fit_w - h * xi < m) & (m < fit_w - h * xi)
	fit1 = np.polyfit(m[small_m_inds], PB_sigmoid[small_m_inds], 1, w=1/d_PB_sigmoid[small_m_inds])
	m0 = - fit1[1] / fit1[0]
	
	return PB_sigmoid, d_PB_sigmoid, small_m_inds, fit1, m0

def plot_PB_AB(ax, ax_log, ax_sgm, h, m, PB, d_PB, title, clr=None, N_fine_points=1000):
	PB_sgm, d_PB_sgm, small_m_inds, fit1, m0 = get_sigmoid_fit(PB, d_PB, m, h)
	ax_log.errorbar(m, PB, yerr=d_PB, fmt='.', label='$' + title + '$', color=clr)
	ax.errorbar(m, PB, yerr=d_PB, fmt='.', label='$' + title + '$', color=clr)
	ax_sgm.errorbar(m, PB_sgm, yerr=d_PB_sgm, fmt='.', label='$' + title + '$', color=clr)
	x_draw = m[small_m_inds]
	ax_sgm.plot(x_draw, np.polyval(fit1, x_draw), label='$m_0 = ' + my.f2s(m0) + '$', color=clr)
	#m_fine = np.linspace(min(m), max(m), N_fine_points)
	#ax_PB.plot(m_AB_fine[:-1], PB_opt_fnc(m_AB_fine[:-1]), label='$P_{B, fit}$', color=my.get_my_color(1))
	#ax_PB_log.plot(m_AB_fine[:-1], PB_opt_fnc(m_AB_fine[:-1]), label='$P_{B, fit}$', color=my.get_my_color(1))
	#ax_PB_sigmoid.plot(m_AB_fine[:-1], -np.log(PB_fit_b / (PB_opt_fnc(m_AB_fine[:-1]) - PB_fit_a) - 1), label='$P_{B, fit}$', color=my.get_my_color(1))

	return PB_sgm, d_PB_sgm, small_m_inds, fit1, m0

def mark_PB_plot(ax, ax_log, ax_sgm, m, PB, interface_mode):
	PB_sgm = -np.log(1/PB - 1)
	
	ax_log.plot([min(m), max(m)], [1/2] * 2, '--', label='$P = 1/2$')
	if(interface_mode == 'M'):
		ax_log.plot([0] * 2, [min(PB), 1], '--', label='$m = 0$')
	
	ax.plot([min(m), max(m)], [1/2] * 2, '--', label='$P = 1/2$')
	if(interface_mode == 'M'):
		ax.plot([0] * 2, [0, 1], '--', label='$m = 0$')
	
	ax_sgm.plot([min(m), max(m)], [0] * 2, '--', label='$P = 1/2$')
	ax_sgm.plot([0] * 2, [min(PB_sgm), max(PB_sgm)], '--', label='$m = 0$')

def proc_FFS(L, Temp, h, \
			N_init_states_AB, N_init_states_BA, \
			M_interfaces_AB, M_interfaces_BA, \
			M_sample_BF_A_to, M_sample_BF_B_to, \
			M_match_BF_A_to, M_match_BF_B_to, \
			interface_mode, def_spin_state, \
			init_gen_mode=-2, \
			verbose=None, to_plot_hists=False):
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
	
	to_plot_hists : bool (True)
		whether or not to plot F profile and other related plots
	
	init_gen_mode : int (-2)
		The way to generate initial states in A to be propagated towards M_0
		-2 - generate an ensemble and in A and sample from this ensemble
		-1 - generate each spin randomly 50/50 to be +-1
		>=0 - generate all spins -1 and then set |mode| random (uniformly distributed over all lattice points) spins to +1
	
	interface_mode : str, ('M'), {'M', 'CS'}
		What is used as the order parameter
	
	M_far_from_A : int, (None)
		The borderline at which sampling of the A state stops
	
	M_far_from_B : int, (None)
		The borderline at which sampling of the B state stops

	Returns
	-------
	1: e^<log> ~= <x>
	2: e^(<log> - d_<log>)
	3: e^(<log> + d_<log>)
	4: e^<log> * d_<log> ~= dx
	"""
	L2 = L**2
	
	# We need to move M_far_from_B, because the simulation is actually done with the reversed field, which is the same as reversed order-parameter (OP). 
	# But 'M_far_from_B' is set in the original OP, so we need to move the 'M_far_from_B' from the original OP axis to the reversed axis.
	M_sample_BF_B_to = flip_M(M_sample_BF_B_to, L2, interface_mode)
	M_match_BF_B_to = flip_M(M_match_BF_B_to, L2, interface_mode)

	probs_AB, d_probs_AB, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, rho_AB, d_rho_AB, M_hist_centers_AB, M_hist_lens_AB = \
		proc_FFS_AB(L, Temp, h, N_init_states_AB, M_interfaces_AB, M_sample_BF_A_to, M_match_BF_A_to, \
					interface_mode, def_spin_state, \
					init_gen_mode=init_gen_mode, to_get_EM=True, verbose=verbose)
	
	probs_BA, d_probs_BA, ln_k_BA, d_ln_k_BA, flux0_BA, d_flux0_BA, rho_BA, d_rho_BA, M_hist_centers_BA, M_hist_lens_BA = \
		proc_FFS_AB(L, Temp, -h, N_init_states_BA, M_interfaces_BA, M_sample_BF_B_to, M_match_BF_B_to, \
					interface_mode, def_spin_state, \
					init_gen_mode=init_gen_mode, to_get_EM=True, verbose=verbose)
	
	# we need to reverse back all the OP-dependent returns sicnce the simulation was done with the reversed OP
	probs_BA = np.flip(probs_BA)
	d_probs_BA = np.flip(d_probs_BA)
	rho_BA = np.flip(rho_BA)
	d_rho_BA = np.flip(d_rho_BA)
	M_hist_centers_BA = flip_M(M_hist_centers_BA, L2, interface_mode)
	M_interfaces_BA = flip_M(M_interfaces_BA, L2, interface_mode)
		
	M_hist_lens_BA = np.flip(M_hist_lens_BA)
	N_init_states_BA = np.flip(N_init_states_BA)
	
	#assert(np.all(abs(M_hist_centers_AB - M_hist_centers_BA) < 1 / L**2 * 1e-3))
	#assert(np.all(abs(M_hist_lens_AB - M_hist_lens_BA) < 1 / L**2 * 1e-3))
	
	steps_used = 1   # 1 step, because [k] = 1/step, and 1 step is used to ln(k)
	k_AB, k_AB_low, k_AB_up, d_k_AB = get_log_errors(ln_k_AB, d_ln_k_AB)
	k_BA, k_BA_low, k_BA_up, d_k_BA = get_log_errors(ln_k_BA, d_ln_k_BA)
	
	k_min = min(k_AB, k_BA)
	k_max = max(k_AB, k_BA)
	ln_p_min = -np.log(1 + k_max / k_min)
	d_ln_p_min = np.sqrt(d_ln_k_AB**2 + d_ln_k_BA**2) / (1 + k_min / k_max)
	
	confidence_thr = 0.3
	if(d_ln_p_min > confidence_thr):
		print('WARNING: poor sampling for p_min, dp/p = ' + my.f2s(d_ln_p_min) + ' > ' + str(confidence_thr))
	
	p_min, p_min_low, p_min_up, d_p_min = get_log_errors(ln_p_min, d_ln_p_min)
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
	
	rho_AB_interp1d = scipy.interpolate.interp1d(M_hist_centers_AB, rho_AB, fill_value='extrapolate')
	d_rho_AB_interp1d = scipy.interpolate.interp1d(M_hist_centers_AB, d_rho_AB, fill_value='extrapolate')
	rho_BA_interp1d = scipy.interpolate.interp1d(M_hist_centers_BA, rho_BA, fill_value='extrapolate')
	d_rho_BA_interp1d = scipy.interpolate.interp1d(M_hist_centers_BA, d_rho_BA, fill_value='extrapolate')
	m = np.sort(np.unique(np.array(list(M_hist_centers_AB) + list(M_hist_centers_BA))))
	m_lens = np.empty(len(m))
	m_lens[0] = m[1] - m[0]
	m_lens[-1] = m[-1] - m[-2]
	m_lens[1:-1] = (m[2:] - m[:-2]) / 2
	rho_fnc = lambda x: p_AB * flux0_AB * rho_AB_interp1d(x) + p_BA * flux0_BA * rho_BA_interp1d(x)
	d_rho_fnc = lambda x: np.sqrt((p_AB * flux0_AB * rho_AB_interp1d(x))**2 * ((d_p_AB / p_AB)**2 + (d_flux0_AB / flux0_AB)**2 + (d_rho_AB_interp1d(x) / rho_AB_interp1d(x))**2) + \
			  (p_BA * flux0_BA * rho_BA_interp1d(x))**2 * ((d_p_BA / p_BA)**2 + (d_flux0_BA / flux0_BA)**2 + (d_rho_BA_interp1d(x) / rho_BA_interp1d(x))**2))
	rho = rho_fnc(m)
	d_rho = d_rho_fnc(m)
	
	F = -Temp * np.log(rho * m_lens)
	d_F = Temp * d_rho / rho
	F = F - min(F)
	
	if(to_plot_hists):
		fig_F, ax_F = my.get_fig(title[interface_mode], r'$F(%s) = -T \ln(\rho(%s) \cdot d%s(%s))$' % (feature_label[interface_mode], feature_label[interface_mode], feature_label[interface_mode], feature_label[interface_mode]), title=r'$F(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_F.errorbar(m, F, yerr=d_F, fmt='.', label='FFS data')
		ax_F.legend()
	
	return F, d_F, m, m_lens, rho_fnc, d_rho_fnc, probs_AB, d_probs_AB, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, probs_BA, d_probs_BA, ln_k_BA, d_ln_k_BA, flux0_BA, d_flux0_BA

# def get_intermediate_vals(x0, s, x1, x2, p1, p2):
	# y1 = np.exp((x0 - x1) / s) + 1
	# y2 = np.exp((x0 - x2) / s) + 1
	# b = (p2 - p1) / (1 / y2 - 1 / y1)
	# a = p1 - b / y1
	# return a, b

# def sigmoid_fnc(tht, x, x1, x2, p1, p2):
	# x0 = tht[0]
	# #s = np.abs(tht[1])
	# s = tht[1]
	# a, b = get_intermediate_vals(x0, s, x1, x2, p1, p2)
	# f = a + b / (1 + np.exp((x0 - x) / s))
	
	# if(np.any(f >= 1)):
		# print('ERROR')
		# print('f:', f, '; tht:', tht, '; x:', x, '; x1:', x1, '; x2:', x2, '; p1:', p1, '; p2:', p2, '; a:', a, '; b:', b)
	
	# return f

# def get_LF_PB(tht, target_fnc, p, d_p):
	# f = target_fnc(tht)
	# LF = np.sum((np.log((1 / f - 1) / (1 / p - 1)) * p * (1 - p) / d_p)**2)
	# return LF

# def PB_fit_optimize(p, d_p, x, x1, x2, p1, p2, tht0=None):
	# if(np.any(p >= 1) or np.any(p <= 0) or np.any(d_p == 0)):
		# print('ERROR')
		# print('p:', p, '; d_p:', d_p)
		# print('If certain d_p are == 0 - it may be due to insufficient statistics, which causes P-s in all the simulations to match since P-s are cumputed as ratios of integers, so there is a >0 probability they will match exactly')
		# #print('log(p):', np.log(p), '; log(d_p):', np.log(d_p))
	
	# target_fnc = lambda tht, x_local: sigmoid_fnc(tht, x_local, x1, x2, p1, p2)
	# d2_fnc = lambda tht: get_LF_PB(tht, lambda tht_local: target_fnc(tht_local, x), p, d_p)
	# tht_opt = scipy.optimize.minimize(d2_fnc, tht0)
	
	# opt_fnc = lambda x: target_fnc(tht_opt.x, x)
	# x0_opt = tht_opt.x[0]
	# s_opt = tht_opt.x[1]
	# a_opt, b_opt = get_intermediate_vals(x0_opt, s_opt, x1, x2, p1, p2)
	# opt_fnc_inv = lambda f: x0_opt - s_opt * np.log(b_opt / (f - a_opt) - 1)
	
	# return tht_opt, opt_fnc, opt_fnc_inv

def proc_order_parameter_FFS(L, Temp, h, flux0, d_flux0, probs, d_probs, M_interfaces, N_init_states, \
							interface_mode, def_spin_state, \
							m_data=None, Nt=None, M_hist_edges=None, memory_time=1, \
							to_plot_time_evol=False, to_plot_hists=False, stride=1, \
							x_lbl='', y_lbl='', Ms_alpha=0.5, OP_match_BF_to=None):
	L2 = L**2
	ln_k_AB = np.log(flux0 * 1) + np.sum(np.log(probs))   # [flux0 * 1] = 1, because [flux] = 1/time = 1/step
	d_ln_k_AB = np.sqrt((d_flux0 / flux0)**2 + np.sum((d_probs / probs)**2))
	
	print('Nt:', Nt)
	#print('N_init_guess: ', np.exp(np.mean(np.log(Nt))) / Nt)
	print('flux0 = (%s) 1/step' % (my.errorbar_str(flux0, d_flux0)))
	print('-log10(P):', -np.log(probs) / np.log(10))
	print('d_P / P:', d_probs / probs)
	k_AB, k_AB_low, k_AB_up, d_k_AB = get_log_errors(ln_k_AB, d_ln_k_AB, lbl='k_AB')
	# if(d_ln_k_AB > 0.3):
		# print('k_AB \\in [' + '; '.join(my.same_scale_strs([k_AB_low, k_AB_up], x_to_compare_to=[k_AB_up - k_AB_low])) + '] 1/step with 68% prob')
	# else:
		# print('k_AB = (%s) 1/step' % my.errorbar_str(k_AB, k_AB * d_ln_k_AB))
	
	m = M_interfaces / OP_scale[interface_mode]
	
	N_M_interfaces = len(m)
	P_B = np.empty(N_M_interfaces)
	d_P_B = np.empty(N_M_interfaces)
	P_B[N_M_interfaces - 1] = 1
	d_P_B[N_M_interfaces - 1] = 0
	for i in range(N_M_interfaces - 2, -1, -1):
		P_B[i] = P_B[i + 1] * probs[i]
		d_P_B[i] = P_B[i] * (1 - P_B[i]) * np.sqrt((d_P_B[i + 1] / P_B[i + 1])**2 + (d_probs[i] / probs[i])**2)
	
	#P_B_opt, P_B_opt_fnc, P_B_opt_fnc_inv = PB_fit_optimize(P_B[:-1], d_P_B[:-1], m[:-1], m[0], m[-1], P_B[0], P_B[-1], tht0=[0, 0.15])
	#P_B_x0_opt = P_B_opt.x[0]
	#P_B_s_opt = P_B_opt.x[1]
	P_B_sigmoid, d_P_B_sigmoid, small_m_inds, fit1, m0 = \
		get_sigmoid_fit(P_B[:-1], d_P_B[:-1], m[:-1], h)
	
	M_optimize_f_interp_vals = 1 - np.log(P_B) / np.log(P_B[0])
	d_M_optimize_f_interp_vals = d_P_B / P_B / np.log(P_B[0])
	#M_optimize_f_interp_fit = lambda m: (1 - np.log(P_B_opt_fnc(m)) / np.log(P_B[0]))
	M_optimize_f_interp_interp1d = scipy.interpolate.interp1d(m, M_optimize_f_interp_vals, fill_value='extrapolate')
	M_optimize_f_desired = np.linspace(0, 1, N_M_interfaces)
	
	# ===== numerical inverse ======
	M_optimize_new_M_interfaces = np.zeros(N_M_interfaces, dtype=np.intc)
	M_optimize_new_M_guess = min(m) + (max(m) - min(m)) * (np.arange(N_M_interfaces) / N_M_interfaces)
	for i in range(N_M_interfaces - 2):
		M_optimize_new_M_interfaces[i + 1] = int(np.round(scipy.optimize.root(lambda x: M_optimize_f_interp_interp1d(x) - M_optimize_f_desired[i + 1], M_optimize_new_M_guess[i + 1]).x[0] * OP_scale[interface_mode]) + 0.1)
	
	if(interface_mode == 'M'):
		M_optimize_new_M_interfaces = M_interfaces[0] + 2 * np.round((M_optimize_new_M_interfaces - M_interfaces[0]) / 2)
		M_optimize_new_M_interfaces = np.int_(M_optimize_new_M_interfaces + 0.1 * np.sign(M_optimize_new_M_interfaces))
	elif(interface_mode == 'CS'):
		M_optimize_new_M_interfaces = np.int_(np.round(M_optimize_new_M_interfaces))
	M_optimize_new_M_interfaces[0] = M_interfaces[0]
	M_optimize_new_M_interfaces[-1] = M_interfaces[-1]
	
	M_optimize_new_M_interfaces_original = np.copy(M_optimize_new_M_interfaces)
	if(np.any(M_optimize_new_M_interfaces[1:] == M_optimize_new_M_interfaces[:-1])):
		print('WARNING: duplicate interfaces, removing duplicates')
		for i in range(N_M_interfaces - 1):
			if(M_optimize_new_M_interfaces[i+1] <= M_optimize_new_M_interfaces[i]):
				if(i == 0):
					M_optimize_new_M_interfaces[1] = M_optimize_new_M_interfaces[0] + OP_step[interface_mode]
				else:
					if(M_optimize_new_M_interfaces[i-1] + OP_step[interface_mode] < M_optimize_new_M_interfaces[i] * (1 - 1e-5 * np.sign(M_optimize_new_M_interfaces[i]))):
						M_optimize_new_M_interfaces[i] = M_optimize_new_M_interfaces[i-1] + OP_step[interface_mode]
					else:
						M_optimize_new_M_interfaces[i+1] = M_optimize_new_M_interfaces[i] + OP_step[interface_mode]
		#M_optimize_new_M_interfaces = np.unique(M_optimize_new_M_interfaces)
	N_new_interfaces = len(M_optimize_new_M_interfaces)
	#print('N_M_i =', N_new_interfaces, '\nM_new:\n', M_optimize_new_M_interfaces, '\nM_new_original:\n', M_optimize_new_M_interfaces_original)
	print('M_new:\n', M_optimize_new_M_interfaces)

	if(m_data is not None):
		Nt_total = len(m_data)
		Nt_totals = np.empty(N_M_interfaces, dtype=np.intc)
		Nt_totals[0] = Nt[0]
		M = [m_data[ : Nt_totals[0]]]
		for i in range(1, N_M_interfaces):
			Nt_totals[i] = Nt_totals[i-1] + Nt[i]
			M.append(m_data[Nt_totals[i-1] : Nt_totals[i]])
		del m_data
		
		# The number of bins cannot be arbitrary because M is descrete, thus it's a good idea if all the bins have 1 descrete value inside or all the bins have 2 or etc. 
		# It's not good if some bins have e.g. 1 possible M value, and others have 2 possible values. 
		# There are 'L^2 + 1' possible values of M, because '{M \in [-L^2; L^2]} and {dM = 2}'
		# Thus, we need edges which cover [-1-1/L^2; 1+1/L^2] with step=2/L^2
		M_hist_lens = (M_hist_edges[1:] - M_hist_edges[:-1])
		M_hist_centers = (M_hist_edges[1:] + M_hist_edges[:-1]) / 2
		N_M_hist = len(M_hist_centers)

		small_rho = 1e-2 / Nt_totals[-1] / N_M_hist
		rho_s = np.zeros((N_M_hist, N_M_interfaces))
		d_rho_s = np.zeros((N_M_hist, N_M_interfaces))
		M_hist_ok_inds = np.zeros((N_M_hist, N_M_interfaces), dtype=bool)
		rho_for_sum = np.empty((N_M_hist, N_M_interfaces - 1))
		d_rho_for_sum = np.zeros((N_M_hist, N_M_interfaces - 1))
		rho_w = np.empty(N_M_interfaces - 1)
		d_rho_w = np.empty(N_M_interfaces - 1)
		timesteps = [np.arange(Nt[0])]
		for i in range(N_M_interfaces):
			#timesteps.append(np.arange(Nt[i]) + (0 if(i == 0) else Nt_totals[i-1]))
			M_hist, _ = np.histogram(M[i], bins=M_hist_edges)
			M_hist = M_hist / memory_time
			M_hist_ok_inds[:, i] = (M_hist > 0)
			#rho_s[M_hist_ok_inds, i] = M_hist[M_hist_ok_inds] / M_hist_lens[M_hist_ok_inds] / Nt[i]   # \pi(q, l_i)
			#d_rho_s[M_hist_ok_inds, i] = np.sqrt(M_hist[M_hist_ok_inds] * (1 - M_hist[M_hist_ok_inds] / Nt[i])) / M_hist_lens[M_hist_ok_inds] / Nt[i]
			rho_s[M_hist_ok_inds[:, i], i] = M_hist[M_hist_ok_inds[:, i]] / M_hist_lens[M_hist_ok_inds[:, i]] / N_init_states[i]   # \pi(q, l_i)
			d_rho_s[M_hist_ok_inds[:, i], i] = np.sqrt(M_hist[M_hist_ok_inds[:, i]] * (1 - M_hist[M_hist_ok_inds[:, i]] / Nt[i])) / M_hist_lens[M_hist_ok_inds[:, i]] / N_init_states[i]
			
			if(i > 0):
				timesteps.append(np.arange(Nt[i]) + Nt_totals[i-1])
				#rho_w[i-1] = (1 if(i == 0) else P_B[0] / P_B[i - 1])
				rho_w[i-1] = P_B[0] / P_B[i - 1]
				#d_rho_w[i] = (0 if(i == 0) else (rho_w[i] * d_P_B[i - 1] / P_B[i - 1]))
				d_rho_w[i-1] = rho_w[i-1] * d_P_B[i - 1] / P_B[i - 1]
				rho_for_sum[:, i-1] = rho_s[:, i] * rho_w[i-1]
				#d_rho_for_sum[M_hist_ok_inds, i] = rho_for_sum[M_hist_ok_inds, i] * np.sqrt((d_rho_s[M_hist_ok_inds, i] / rho_s[M_hist_ok_inds, i])**2 + (0 if(i == 0) else np.sum((d_probs[:i] / probs[:i])**2)))
				d_rho_for_sum[M_hist_ok_inds[:, i], i-1] = rho_for_sum[M_hist_ok_inds[:, i], i-1] * np.sqrt((d_rho_s[M_hist_ok_inds[:, i], i] / rho_s[M_hist_ok_inds[:, i], i])**2 + (d_rho_w[i-1] / rho_w[i-1])**2)
		
		rho = np.sum(rho_for_sum, axis=1)
		d_rho = np.sqrt(np.sum(d_rho_for_sum**2, axis=1))
		M_hist_ok_inds_FFS = (rho > 0)
		F = np.zeros(N_M_hist)
		d_F = np.zeros(N_M_hist)
		F[M_hist_ok_inds_FFS] = -Temp * np.log(rho[M_hist_ok_inds_FFS] * M_hist_lens[M_hist_ok_inds_FFS])
		d_F[M_hist_ok_inds_FFS] = Temp * d_rho[M_hist_ok_inds_FFS] / rho[M_hist_ok_inds_FFS]
		
		BF_inds = M_hist_centers <= OP_match_BF_to / OP_scale[interface_mode]
		F_BF = -Temp * np.log(rho_s[BF_inds, 0] * M_hist_lens[BF_inds])
		d_F_BF = Temp * d_rho_s[BF_inds, 0] / rho_s[BF_inds, 0]
		
		FFS_match_points_inds = (M_hist_centers > M_interfaces[0] / OP_scale[interface_mode]) & BF_inds
		BF_match_points = (M_hist_centers[BF_inds] > M_interfaces[0] / OP_scale[interface_mode])
		#print(d_F.shape, FFS_match_points_inds.shape, d_F_BF.shape, BF_match_points.shape)
		#input('a')
		w2 = 1 / (d_F[FFS_match_points_inds]**2 + d_F_BF[BF_match_points]**2)
		dF_FFS_BF = F[FFS_match_points_inds] - F_BF[BF_match_points]
		#F_shift = np.sum(dF_FFS_BF * w2) / np.sum(w2)
		F_shift = np.average(dF_FFS_BF, weights=w2)
		F_BF = F_BF + F_shift
		
		BF_OP_border = M_interfaces[0] + OP_step[interface_mode] * 3
		N_points_to_match_F_to = (OP_match_BF_to - BF_OP_border) / OP_step[interface_mode]
		if(N_points_to_match_F_to < 5):
			print('WARNING: Only few (%s) points to match F_BF to' % (my.f2s(N_points_to_match_F_to)))
		
		state_A_FFS_inds = (M_hist_centers <= BF_OP_border / OP_scale[interface_mode]) & BF_inds
		state_A_BF_inds = M_hist_centers[BF_inds] <= BF_OP_border / OP_scale[interface_mode]
		rho[state_A_FFS_inds] = np.exp(-F_BF[state_A_BF_inds] / Temp) / M_hist_lens[state_A_FFS_inds]
		d_rho[state_A_FFS_inds] = rho[state_A_FFS_inds] * d_F_BF[state_A_BF_inds] / Temp
		
		F = -Temp * np.log(rho * M_hist_lens)
		F_BF = F_BF - min(F)
		F = F - min(F)
		d_F = Temp * d_rho / rho
		
		if(to_plot_time_evol):
			fig_M, ax_M = my.get_fig('step', y_lbl, title=x_lbl + '(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			for i in range(N_M_interfaces):
				ax_M.plot(timesteps[i][::stride], M[i][::stride], label='data, i=%d' % (i))
				
			ax_M.legend()
		else:
			ax_M = None
		
		if(to_plot_hists):
			fig_Mhists, ax_Mhists = my.get_fig(y_lbl, r'$\rho_i(' + x_lbl + ') \cdot P(i | A)$', title=r'$\rho(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			for i in range(N_M_interfaces - 1):
				ax_Mhists.bar(M_hist_centers, rho_for_sum[:, i], yerr=d_rho_for_sum[:, i], width=M_hist_lens, align='center', label=str(i), alpha=Ms_alpha)
			for i in range(N_M_interfaces):
				ax_Mhists.plot([m[i]] * 2, [min(rho), max(rho)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_Mhists.legend()
			
			fig_F, ax_F = my.get_fig(y_lbl, r'$F(' + x_lbl + r') = -T \ln(\rho(%s) \cdot d%s(%s))$' % (x_lbl, x_lbl, x_lbl), title=r'$F(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_F.errorbar(M_hist_centers, F, yerr=d_F, fmt='.', label='$F_{total}$')
			ax_F.errorbar(M_hist_centers[BF_inds], F_BF, yerr=d_F_BF, fmt='.', label='$F_{BF}$')
			for i in range(N_M_interfaces):
				ax_F.plot([m[i]] * 2, [min(F), max(F)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_F.legend()
			
			fig_PB_log, ax_PB_log = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			fig_PB, ax_PB = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			fig_PB_sgm, ax_PB_sgm = my.get_fig(y_lbl, r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sgm, m, P_B[:-1], interface_mode)
			plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sgm, h, m[:-1], P_B[:-1], d_P_B[:-1], 'P_B')
			ax_PB_log.legend()
			ax_PB.legend()
			ax_PB_sgm.legend()
			
			#ax_PB_log.errorbar(m, P_B, yerr=d_P_B, fmt='.', label='data')
			#ax_PB_log.plot(m[:-1], P_B_opt_fnc(m[:-1]), label='fit')
			#ax_PB.errorbar(m, P_B, yerr=d_P_B, fmt='.', label='data')
			#ax_PB.plot(m[:-1], P_B_opt_fnc(m[:-1]), label=r'fit; $(x_0, \sigma) = (%s, %s)$' % (my.f2s(P_B_x0_opt), my.f2s(P_B_s_opt)))
			
			fig_fl, ax_fl = my.get_fig(y_lbl, r'$1 - \ln[P_B(' + x_lbl + ')] / \ln[P_B(0)] = f(\lambda_i)$', title=r'$f(\lambda_i)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			fig_fl_i, ax_fl_i = my.get_fig(r'$index(' + x_lbl + '_i)$', r'$1 - \ln[P_B(' + x_lbl + '_i)] / \ln[P_B(0)] = f(\lambda_i)$', title=r'$f(\lambda_i)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			m_fine = np.linspace(min(m), max(m), int((max(m) - min(m)) / min(m[1:] - m[:-1]) * 10))
			ax_fl.errorbar(m, M_optimize_f_interp_vals, yerr=d_M_optimize_f_interp_vals, fmt='.', label=r'$f(\lambda)_i$')
			ax_fl_i.errorbar(np.arange(N_M_interfaces), M_optimize_f_interp_vals, yerr=d_M_optimize_f_interp_vals, fmt='.', label=r'$f(\lambda)_i$')
			#ax_fl.plot(m_fine[:-1], M_optimize_f_interp_fit(m_fine[:-1]), label=r'$f_{fit}(\lambda)$')
			ax_fl.plot(m_fine, M_optimize_f_interp_interp1d(m_fine), label=r'$f_{interp}(\lambda)$')
			for i in range(N_new_interfaces):
				ax_fl.plot([M_optimize_new_M_interfaces[i] / OP_scale[interface_mode]] * 2, [0, M_optimize_f_interp_interp1d(M_optimize_new_M_interfaces[i] / OP_scale[interface_mode])], '--', label=('new interfaces' if(i == 0) else None), color=my.get_my_color(3))
				ax_fl.plot([min(m), M_optimize_new_M_interfaces[i] / OP_scale[interface_mode]], [M_optimize_f_interp_interp1d(M_optimize_new_M_interfaces[i] / OP_scale[interface_mode])] * 2, '--', label=None, color=my.get_my_color(3))
			ax_fl.plot([min(m), max(m)], [0, 1], '--', label=r'$interpolation$')
			ax_fl_i.plot([0, N_M_interfaces - 1], [0, 1], '--', label=r'$interpolation$')
			
			ax_fl.legend()
			ax_fl_i.legend()
		else:
			ax_PB_log = None
			ax_PB = None
			ax_fl = None
			ax_fl_i = None
			ax_F = None
			ax_Mhists = None
			
		return ln_k_AB, d_ln_k_AB, rho, d_rho, M_hist_centers, M_hist_lens, \
				ax_M, ax_Mhists, ax_F, ax_PB_log, ax_PB, ax_fl, ax_fl_i
	
	else:
		return ln_k_AB, d_ln_k_AB

def proc_FFS_AB(L, Temp, h, N_init_states, OP_interfaces, OP_sample_BF_to, OP_match_BF_to, 
				interface_mode, def_spin_state, verbose=None, to_get_EM=False, \
				to_plot_time_evol=False, to_plot_hists=False, EM_stride=-3000, \
				init_gen_mode=-2, Ms_alpha=0.5):
	print('=============== running h = %s ==================' % (my.f2s(h)))
	
	# py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> M_interfaces, int to_get_EM, std::optional<int> _verbose)
	# return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M, biggest_cluster_sizes);
	(_states, probs, d_probs, Nt, flux0, d_flux0, _E, _M, _CS) = \
		izing.run_FFS(L, Temp, h, N_init_states, OP_interfaces, verbose=verbose, \
					to_remember_timeevol=to_get_EM, init_gen_mode=init_gen_mode, \
					interface_mode=OP_C_id[interface_mode], default_spin_state=def_spin_state)
	
	L2 = L**2
	N_OP_interfaces = len(OP_interfaces)
	if(to_get_EM):
		Nt_total = len(_M)
		if(EM_stride < 0):
			EM_stride = np.int_(- Nt_total / EM_stride)
	
	states = [_states[ : N_init_states[0] * L2].reshape((N_init_states[0], L, L))]
	for i in range(1, N_OP_interfaces - 1):
		states.append(_states[np.sum(N_init_states[:i]) * L2 : np.sum(N_init_states[:(i+1)] * L2)].reshape((N_init_states[i], L, L)))
	del _states
	
	data = {}
	data['M'] = _M
	data['CS'] = _CS
	
	assert(OP_interfaces[0] < OP_match_BF_to), 'ERROR: OP_interfaces start at %d, but the state is match to BF only through %d' % (OP_interfaces[0], OP_match_BF_to)
	assert(OP_match_BF_to < OP_sample_BF_to), 'ERROR: OP_match_to = %d >= OP_sample_to = %d' % (OP_match_BF_to, OP_sample_BF_to)
	
	ln_k_AB, d_ln_k_AB, rho, d_rho, M_hist_centers, M_hist_lens, \
		ax_M, ax_Mhists, ax_F, ax_PB_log, ax_PB, ax_fl, ax_fl_i = \
			proc_order_parameter_FFS(L, Temp, h, flux0, d_flux0, probs, d_probs, OP_interfaces, N_init_states, \
						interface_mode, def_spin_state, OP_match_BF_to=OP_match_BF_to, \
						m_data=data[interface_mode] / OP_scale[interface_mode], Nt=Nt, M_hist_edges=get_OP_hist_edges(interface_mode, OP_max=OP_interfaces[-1]), \
						memory_time=max(1, m_std_overestimate[interface_mode] / OP_step[interface_mode]), 
						to_plot_time_evol=to_plot_time_evol, to_plot_hists=to_plot_hists, stride=EM_stride, \
						x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], Ms_alpha=Ms_alpha)
	
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
	#assert(a > 0)
	
	x0 = -fit2[1] / (2 * a)
	#assert(x0 > x1)
	
	c = fit2[2] - a * x0**2
	dx = (x0 - x1) * np.sqrt(np.abs(a))
	
	return np.exp(c) / 2 * np.sqrt(np.pi / np.abs(a)) * (scipy.special.erfi(dx) if(a > 0) else scipy.special.erf(dx))

def proc_order_parameter_BF(L, Temp, h, m, E, stab_step, dm_step, m_hist_edges, m_peak_guess, m_fit2_width, \
						 x_lbl=None, y_lbl=None, verbose=None, to_estimate_k=False, hA=None, \
						 to_plot_time_evol=True, to_plot_F=True, to_plot_ETS=False, stride=1):
	Nt = len(m)
	steps = np.arange(Nt)
	L2 = L**2
	stab_ind = (steps > stab_step)
	m_stab = m[stab_ind]
	
	Nt_stab = len(m_stab)
	assert(Nt_stab > 1), 'ERROR: the system may not have reached equlibrium, stab_step = %d, max-present-step = %d' % (stab_step, max(steps))
	
	m_mean = np.mean(m_stab)
	m_std = np.std(m_stab)
	memory_time = max(1, m_std / dm_step)   
	d_m_mean = m_std / np.sqrt(Nt_stab / memory_time)
	# Time of statistical decorelation of the system state.
	# 'the number of flips necessary to cover the typical system states range' * 'the upper-bound estimate on the number of steps for 1 flip to happen'
	print('memory time =', my.f2s(memory_time))
	
	m_hist, m_hist_edges = np.histogram(m_stab, bins=m_hist_edges)
	# The number of bins cannot be arbitrary because M is descrete, thus it's a good idea if all the bins have 1 descrete value inside or all the bins have 2 or etc. 
	# It's not good if some bins have e.g. 1 possible M value, and others have 2 possible values. 
	# There are 'L^2 + 1' possible values of M, because '{M \in [-L^2; L^2]} and {dM = 2}'
	# Thus, we need edges which cover [-1-1/L^2; 1+1/L^2] with step=2/L^2
	m_hist_lens = (m_hist_edges[1:] - m_hist_edges[:-1])
	N_m_points = len(m_hist_lens)
	m_hist[m_hist == 0] = 1   # to not get errors for log(hist)
	m_hist = m_hist / memory_time
	m_hist_centers = (m_hist_edges[1:] + m_hist_edges[:-1]) / 2
	rho = m_hist / m_hist_lens / Nt_stab
	d_rho = np.sqrt(m_hist * (1 - m_hist / Nt_stab)) / m_hist_lens / Nt_stab

	rho_interp1d = scipy.interpolate.interp1d(m_hist_centers, rho, fill_value='extrapolate')
	d_rho_interp1d = scipy.interpolate.interp1d(m_hist_centers, d_rho, fill_value='extrapolate')

	F = -Temp * np.log(rho * m_hist_lens)
	F = F - F[0]
	d_F = Temp * d_rho / rho
	
	E_avg = np.empty(N_m_points)
	for i in range(N_m_points):
		E_for_avg = E[(m_hist_edges[i] < m) & (m < m_hist_edges[i+1])] * L2
		#E_avg[i] = np.average(E_for_avg, weights=np.exp(- E_for_avg / Temp))
		E_avg[i] = np.mean(E_for_avg) if(len(E_for_avg) > 0) else (2 + abs(h)) * L2
	E_avg = E_avg - E_avg[0]   # we can choose the E constant
	S = (E_avg - F) / Temp
	S = S - S[0] # S[m=+-1] = 0 because S = log(N), and N(m=+-1)=1
	
	if(to_estimate_k):
		assert(hA is not None), 'ERROR: k estimation is requested in BF, but no hA provided'
		#k_AB_ratio = np.sum(hA > 0.5) / np.sum(hA < 0.5)
		hA_jump = hA[1:] - hA[:-1]   # jump > 0 => {0->1} => from B to A => k_BA ==> k_ratio = k_BA / k_AB

		N_AB_jumps = np.sum(hA_jump < 0)
		k_AB = N_AB_jumps / np.sum(hA > 0.5)
		d_k_AB = k_AB / np.sqrt(N_AB_jumps)

		N_BA_jumps = np.sum(hA_jump > 0)
		k_BA = N_BA_jumps / np.sum(hA < 0.5)
		d_k_BA = k_BA / np.sqrt(N_BA_jumps)
		
		m_fit2_mmin_ind = np.argmax(m_hist_centers > m_peak_guess - m_fit2_width)
		m_fit2_mmax_ind = np.argmax(m_hist_centers > m_peak_guess + m_fit2_width)
		assert(m_fit2_mmin_ind > 0), 'issues finding analytical border'
		assert(m_fit2_mmax_ind > 0), 'issues finding analytical border'
		m_fit2_inds = (m_fit2_mmin_ind <= np.arange(len(m_hist_centers))) & (m_fit2_mmax_ind >= np.arange(len(m_hist_centers)))
		m_fit2 = np.polyfit(m_hist_centers[m_fit2_inds], F[m_fit2_inds], 2, w = 1/d_F[m_fit2_inds])
		m_peak = -m_fit2[1] / (2 * m_fit2[0])
		m_peak_ind = np.argmin(abs(m_hist_centers - m_peak))
		F_max = np.polyval(m_fit2, m_peak)
		#m_peak_ind = np.argmin(m_hist_centers - m_peak)
		bc_Z_AB = exp_integrate(m_hist_centers[ : m_fit2_mmin_ind + 1], -F[ : m_fit2_mmin_ind + 1] / Temp) + exp2_integrate(- m_fit2 / Temp, m_hist_centers[m_fit2_mmin_ind])
		bc_Z_BA = exp_integrate(m_hist_centers[m_fit2_mmax_ind : ], -F[m_fit2_mmax_ind : ] / Temp) + abs(exp2_integrate(- m_fit2 / Temp, m_hist_centers[m_fit2_mmax_ind]))
		k_bc_AB = (np.mean(abs(m_hist_centers[m_peak_ind + 1] - m_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max / Temp) / bc_Z_AB)
		k_bc_BA = (np.mean(abs(m_hist_centers[m_peak_ind + 1] - m_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max / Temp) / bc_Z_BA)
		
	else:
		k_AB = None
		d_k_AB = None
		k_BA = None
		d_k_BA = None
		k_bc_AB = None
		k_bc_BA = None
	
	if(to_plot_time_evol):
		fig_m, ax_m = my.get_fig('step', y_lbl, title='$' + x_lbl + '$(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_m.plot(steps[::stride], m[::stride], label='data')
		ax_m.plot([stab_step] * 2, [min(m), max(m)], label='equilibr')
		ax_m.plot([stab_step, Nt], [m_mean] * 2, label=('$<' + x_lbl + '> = ' + my.errorbar_str(m_mean, d_m_mean) + '$'))
		ax_m.legend()
		
		fig_hA, ax_hA = my.get_fig('step', '$h_A$', title='$h_{A, ' + x_lbl + '}$(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_hA.plot(steps[::stride], hA[::stride], label='data')
		ax_hA.legend()

		print('k_' + x_lbl + '_AB    =    (' + my.errorbar_str(k_AB, d_k_AB) + ')    (1/step);    dk/k =', my.f2s(d_k_AB / k_AB))
		print('k_' + x_lbl + '_BA    =    (' + my.errorbar_str(k_BA, d_k_BA) + ')    (1/step);    dk/k =', my.f2s(d_k_BA / k_BA))
	else:
		ax_m = None
	
	if(to_plot_F):
		fig_mhist, ax_m_hist = my.get_fig(y_lbl, r'$\rho(' + x_lbl + ')$', title=r'$\rho(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_m_hist.bar(m_hist_centers, rho, yerr=d_rho, width=m_hist_lens, align='center')
		#ax_m_hist.legend()
		
		fig_F, ax_F = my.get_fig(y_lbl, r'$F/J$', title=r'$F(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_F.errorbar(m_hist_centers, F, yerr=d_F, fmt='.', label='$F(' + x_lbl + ')$')
		if(to_estimate_k):
			ax_F.plot(m_hist_centers[m_fit2_inds], np.polyval(m_fit2, m_hist_centers[m_fit2_inds]), label='fit2; $' + x_lbl + '_0 = ' + my.f2s(m_peak) + '$')
		
		if(to_plot_ETS):
			ax_F.plot(m_hist_centers, E_avg, '.', label='<E>(' + x_lbl + ')')
			ax_F.plot(m_hist_centers, S * Temp, '.', label='$T \cdot S(' + x_lbl + ')$')
		ax_F.legend()
		
		print('k_' + x_lbl + '_bc_AB   =   ' + my.f2s(k_bc_AB) + '   (1/step);')
		print('k_' + x_lbl + '_bc_BA   =   ' + my.f2s(k_bc_BA) + '   (1/step);')
	else:
		ax_m_hist = None
		ax_F = None
	
	return F, d_F, m_hist_centers, m_hist_lens, rho_interp1d, d_rho_interp1d, k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, m_mean, m_std, d_m_mean, E_avg, S, ax_m, ax_m_hist, ax_F

def plot_correlation(x, y, x_lbl, y_lbl):
	fig, ax = my.get_fig(x_lbl, y_lbl)
	R = scipy.stats.pearsonr(x, y)[0]
	ax.plot(x, y, '.', label='R = ' + my.f2s(R))
	ax.legend()
	
	return ax, R

def proc_T(L, Temp, h, Nt, interface_mode, def_spin_state, verbose=None, N_spins_up_init=None, \
			to_plot_time_evol=False, to_plot_F=False, to_plot_ETS=False, to_plot_correlations=False, \
			to_estimate_k=False, EM_stride=-3000, OP_min=None, OP_max=None, OP_A=None, OP_B=None):
	L2 = L**2
	
	if(OP_min is None):
		OP_min = OP_min_default[interface_mode]
	if(OP_max is None):
		OP_max = OP_max_default[interface_mode]
	
	if(N_spins_up_init is None):
		if(interface_mode == 'M'):
			N_spins_up_init = (L2 - (OP_min + 1) * def_spin_state) // 2
			if(OP_min == L2 - 2 * N_spins_up_init):
				N_spins_up_init += 1
		elif(interface_mode == 'CS'):
			N_spins_up_init = OP_min + 1
	if(interface_mode == 'M'):
		OP_init = (L2 - 2 * N_spins_up_init) * def_spin_state
	elif(interface_mode == 'CS'):
		OP_init = N_spins_up_init
	
	(E, M, CS, hA) = izing.run_bruteforce(L, Temp, h, Nt, N_spins_up_init=N_spins_up_init, OP_A=OP_A, OP_B=OP_B, \
										OP_min=OP_min, OP_max=OP_max, \
										interface_mode=OP_C_id[interface_mode], default_spin_state=def_spin_state, \
										verbose=verbose)
	
	if(EM_stride < 0):
		EM_stride = np.int_(- Nt / EM_stride)

	stab_step = int(min(L2, Nt / 2)) * 5
	# Each spin has a good chance to flip during this time
	# This does not guarantee the global stable state, but it is sufficient to be sure-enough we are in the optimum of the local metastable-state.
	# The approximation for a global ebulibration would be '1/k_AB', where k_AB is the rate constant. But it's hard to estimate on that stage.
	data = {}
	hist_edges = {}
	
	data['M'] = M
	hist_edges['M'] =  get_OP_hist_edges(interface_mode, OP_min if(interface_mode == 'M') else None, OP_max if(interface_mode == 'M') else None)
	del M

	data['CS'] = CS
	hist_edges['CS'] = get_OP_hist_edges(interface_mode, OP_min if(interface_mode == 'CS') else None, OP_max if(interface_mode == 'CS') else None)
	
	N_features = len(hist_edges.keys())
	
	F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, \
		k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, m_avg, \
			m_std, d_m_avg, E_avg, S_E, \
			ax_time, ax_hist, ax_F = \
				proc_order_parameter_BF(L, Temp, h, data[interface_mode] / OP_scale[interface_mode], E / L2, stab_step, \
								dm_step[interface_mode], hist_edges[interface_mode], peak_guess[interface_mode], fit2_width[interface_mode], \
								x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], verbose=verbose, to_estimate_k=to_estimate_k, hA=hA, \
								to_plot_time_evol=to_plot_time_evol, to_plot_F=to_plot_F, to_plot_ETS=to_plot_ETS, stride=EM_stride)

	BC_cluster_size = L2 / np.pi
	#C = ((m_std['E'] * L2)**2 / Temp**2) / L2   # heat capacity per spin

	if(to_plot_F):
		F_limits = [min(F), max(F)]
		hist_limits = [min(rho_interp1d(hist_centers)), max(rho_interp1d(hist_centers))]
		ax_F.plot([OP_init / OP_scale[interface_mode]] * 2, F_limits, '--', label='init state')
		ax_hist.plot([OP_init / OP_scale[interface_mode]] * 2, hist_limits, '--', label='init state')
		if(interface_mode == 'CS'):
			ax_hist.plot([BC_cluster_size] * 2, hist_limits, '--', label='$L_{BC} = L^2 / \pi$')
			ax_hist.legend()
			ax_F.plot([BC_cluster_size] * 2, F_limits, '--', label='$S_{BC} = L^2 / \pi$')
		ax_F.legend()
	
	# if(to_plot_correlations):
		# feature_labels_ordered = list(hist_edges.keys())
		# for i1 in range(N_features):
			# for i2 in range(i1 + 1, N_features):
				# k1 = feature_labels_ordered[i1]
				# k2 = feature_labels_ordered[i2]
				# plot_correlation(data[k1][::EM_stride] / OP_scale[k1], data[k2][::EM_stride] / OP_scale[k2], \
								# feature_label[k1], feature_label[k2])
	
	return F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, m_avg, d_m_avg, E_avg#, C

def run_many(L, Temp, h, N_runs, interface_mode, def_spin_state, \
			OP_A=None, OP_B=None, N_spins_up_init=None, \
			M_interfaces_AB=None, M_interfaces_BA=None, \
			M_sample_BF_A_to=None, M_sample_BF_B_to=None, \
			M_match_BF_A_to=None, M_match_BF_B_to=None, \
			Nt_per_BF_run=None, verbose=None, to_plot_time_evol=False, \
			EM_stride=-3000, to_plot_k_distr=False, N_k_bins=10, \
			mode='BF', N_init_states_AB=None, N_init_states_BA=None, \
			init_gen_mode=-2, to_plot_committer=None):
	L2 = L**2
	old_seed = izing.get_seed()
	if(mode == 'BF'):
		assert(Nt_per_BF_run is not None), 'ERROR: BF mode but no Nt_per_run provided'
		if(OP_A is None):
			assert(len(M_interfaces_AB) > 0), 'ERROR: nor OP_A neither M_interfaces_AB were provided for run_many(BF)'
			OP_A = M_interfaces_AB[0]
		if(OP_B is None):
			assert(len(M_interfaces_BA) > 0), 'ERROR: nor OP_B neither M_interfaces_BA were provided for run_many(BF)'
			OP_B = flip_M(M_interfaces_BA[0], L2, interface_mode)
		
	elif(mode == 'FFS'):
		assert(N_init_states_AB is not None), 'ERROR: FFS mode but no "N_init_states_AB" provided'
		assert(N_init_states_BA is not None), 'ERROR: FFS mode but no "N_init_states_BA" provided'
		assert(M_interfaces_AB is not None), 'ERROR: FFS mode but no "M_interfaces_AB" provided'
		assert(M_interfaces_BA is not None), 'ERROR: FFS mode but no "M_interfaces_BA" provided'
		
		N_M_interfaces_AB = len(M_interfaces_AB)
		N_M_interfaces_BA = len(M_interfaces_BA)
		m_AB = M_interfaces_AB / OP_scale[interface_mode]
		m_BA = flip_M(M_interfaces_BA, L2, interface_mode) / OP_scale[interface_mode]
		m = np.sort(np.unique(np.concatenate((m_AB, m_BA))))
	
	ln_k_AB_data = np.empty(N_runs) 
	ln_k_BA_data = np.empty(N_runs) 
	F_data = np.empty((L2 + 1, N_runs))
	d_F_data = np.empty((L2 + 1, N_runs))
	rho_fncs = [[]] * N_runs
	d_rho_fncs = [[]] * N_runs
	if(mode == 'BF'):
		ln_k_bc_AB_data = np.empty(N_runs)
		ln_k_bc_BA_data = np.empty(N_runs) 
	elif(mode == 'FFS'):
		flux0_AB_data = np.empty(N_runs) 
		flux0_BA_data = np.empty(N_runs)
		probs_AB_data = np.empty((N_M_interfaces_AB - 1, N_runs))
		probs_BA_data = np.empty((N_M_interfaces_BA - 1, N_runs))

	for i in range(N_runs):
		izing.init_rand(i + old_seed)
		if(mode == 'BF'):
			F_data[:, i], d_F_data[:, i], M_hist_centers_new, M_hist_lens_new, \
				rho_fncs[i], d_rho_fncs[i], k_bc_AB_BF, k_bc_BA_BF, k_AB_BF, _, k_BA_BF, _, _, _, _ = \
					proc_T(L, Temp, h, Nt_per_BF_run, interface_mode, def_spin_state, \
							OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
							verbose=verbose, to_plot_time_evol=to_plot_time_evol, \
							to_plot_F=False, to_plot_correlations=False, to_plot_ETS=False, \
							EM_stride=EM_stride, to_estimate_k=True)
			
			ln_k_bc_AB_data[i] = np.log(k_bc_AB_BF * 1)   # proc_T returns 'k', not 'log(k)'; *1 for units [1/step]
			ln_k_bc_BA_data[i] = np.log(k_bc_BA_BF * 1)
			ln_k_AB_data[i] = np.log(k_AB_BF * 1)
			ln_k_BA_data[i] = np.log(k_BA_BF * 1)
		elif(mode == 'FFS'):
			F_data[:, i], d_F_data[:, i], M_hist_centers_new, M_hist_lens_new, rho_fncs[i], d_rho_fncs[i], probs_AB_data[:, i], _, \
			ln_k_AB_data[i], _, flux0_AB_data[i], _, probs_BA_data[:, i], _, ln_k_BA_data[i], _, flux0_BA_data[i], _ = \
				proc_FFS(L, Temp, h, \
						N_init_states_AB, N_init_states_BA, \
						M_interfaces_AB, M_interfaces_BA, \
						M_sample_BF_A_to, M_sample_BF_B_to, \
						M_match_BF_A_to, M_match_BF_B_to, \
						interface_mode, def_spin_state, \
						init_gen_mode=init_gen_mode)
			
		print('%s %lf %%' % (mode, (i+1) / (N_runs) * 100))
		
		if(i == 0):
			M_hist_centers = M_hist_centers_new
			M_hist_lens = M_hist_lens_new
		else:
			def check_M_centers_match(c, i, c_new, arr_name, scl=1, err_thr=1e-6):
				rel_err = abs(c_new - c) / scl
				miss_inds = rel_err > err_thr
				assert(np.all(~miss_inds)), 'ERROR: %s from i=0 and i=%d does not match:\nold: %s\nnew: %s\nerr: %s' % (arr_name, i, str(c[miss_inds]), str(c_new[miss_inds]), str(rel_err[miss_inds]))
			
			check_M_centers_match(M_hist_centers, i, M_hist_centers_new, 'M_hist_centers', max(abs(M_hist_centers[1:] - M_hist_centers[:-1])))
			check_M_centers_match(M_hist_lens, i, M_hist_lens_new, 'M_hist_lens', max(M_hist_lens))
	
	F, d_F = get_average(F_data, axis=1)
	ln_k_AB, d_ln_k_AB = get_average(ln_k_AB_data)
	ln_k_BA, d_ln_k_BA = get_average(ln_k_BA_data)
	if(mode == 'BF'):
		ln_k_bc_AB, d_ln_k_bc_AB = get_average(ln_k_bc_AB_data)
		ln_k_bc_BA, d_ln_k_bc_BA = get_average(ln_k_bc_BA_data)
		
	elif(mode == 'FFS'):
		
		flux0_AB, d_flux0_AB = get_average(flux0_AB_data, mode='log')
		flux0_BA, d_flux0_BA = get_average(flux0_BA_data, mode='log')
		probs_AB, d_probs_AB = get_average(probs_AB_data, mode='log', axis=1)
		probs_BA, d_probs_BA = get_average(probs_BA_data, mode='log', axis=1)
		
		PB_AB_data = np.empty((N_M_interfaces_AB, N_runs))
		PA_BA_data = np.empty((N_M_interfaces_BA, N_runs))
		PB_AB_data[N_M_interfaces_AB - 1, :] = 1
		PA_BA_data[0, :] = 1
		for i in range(N_M_interfaces_AB - 2, -1, -1):
			PB_AB_data[i, :] =  PB_AB_data[i + 1, :] * probs_AB_data[i, :]   
			# np.prod(probs_AB_data[(i + 1):(N_M_interfaces_AB), :], axis=0)
		for i in range(0, N_M_interfaces_BA - 1, 1):
			PA_BA_data[i + 1, :] = PA_BA_data[i, :] * probs_AB_data[i, :]
			#PA_BA_data[i + 1, :] = np.prod(probs_BA_data[:(i + 1), :], axis=0)
			
		PB_AB, d_PB_AB = get_average(PB_AB_data, mode='log', axis=1)
		PA_BA, d_PA_BA = get_average(PA_BA_data, mode='log', axis=1)
		
		# =========================== 4-param sigmoid =======================
		# if(interface_mode == 'M'):
			# tht0_AB = [0, -0.15]
			# tht0_BA = [tht0_AB[0], - tht0_AB[1]]
		# elif(interface_mode == 'CS'):
			# tht0_AB = [60, 10]
			# tht0_BA = [L2 - tht0_AB[0], - tht0_AB[1]]
		# PB_opt, PB_opt_fnc, PB_opt_fnc_inv = PB_fit_optimize(PB_AB[:-1], d_PB_AB[:-1], m_AB[:-1], m_AB[0], m_AB[-1], PB_AB[0], PB_AB[-1], tht0=tht0_AB)
		# PA_opt, PA_opt_fnc, PA_opt_fnc_inv = PB_fit_optimize(PA_BA[1:], d_PA_BA[1:], m_BA[1:], m_BA[0], m_BA[-1], PA_BA[0], PA_BA[-1], tht0=tht0_BA)
		# m0_AB = PB_opt.x[0]
		# m0_BA = PA_opt.x[0]
		# PB_fit_a, PB_fit_b = get_intermediate_vals(PB_opt.x[0], PB_opt.x[1], m_AB[0], m_AB[-1], PB_AB[0], PB_AB[-1])
		# PA_fit_a, PA_fit_b = get_intermediate_vals(PA_opt.x[0], PA_opt.x[1], m_BA[0], m_BA[-1], PA_BA[0], PA_BA[-1])

		#PB_sigmoid = -np.log(PB_fit_b / (PB_AB[:-1] - PB_fit_a) - 1)
		#d_PB_sigmoid = d_PB_AB[:-1] * PB_fit_b / ((PB_AB[:-1] - PB_fit_a) * (PB_fit_b - PB_AB[:-1] + PB_fit_a))
		#PA_sigmoid = -np.log(PA_fit_b / (PA_BA[:-1] - PA_fit_a) - 1)
		#d_PA_sigmoid = d_PA_BA[1:] * PA_fit_b / ((PA_BA[1:] - PA_fit_a) * (PA_fit_b - PA_BA[1:] + PA_fit_a))

		# =========================== 2-param sigmoid =======================
		PB_sigmoid, d_PB_sigmoid, small_m_AB_inds, fit1_AB, m0_AB = \
			get_sigmoid_fit(PB_AB[:-1], d_PB_AB[:-1], m_AB[:-1], h)
		PA_sigmoid, d_PA_sigmoid, small_m_BA_inds, fit1_BA, m0_BA = \
			get_sigmoid_fit(PA_BA[1:], d_PA_BA[1:], m_BA[1:], h)
		# PB_sigmoid = -np.log(1 / PB_AB[:-1] - 1)
		# d_PB_sigmoid = d_PB_AB[:-1] / (PB_AB[:-1] * (1 - PB_AB[:-1]))
		# PA_sigmoid = -np.log(1 / PA_BA[1:] - 1)
		# d_PA_sigmoid = d_PA_BA[1:] / (PA_BA[1:] * (1 - PA_BA[1:]))
		
		# xi = 5   # shift of the maxima due to 'h'
		# small_m_AB_inds = (-0.1 - h * xi < m_AB[:-1]) & (m_AB[:-1] < 0.1 - h * xi)
		# small_m_BA_inds = (-0.1 - h * xi < m_BA[1:]) & (m_BA[1:] < 0.1 - h * xi)
		# fit1_AB = np.polyfit(m_AB[:-1][small_m_AB_inds], PB_sigmoid[small_m_AB_inds], 1, w=1/d_PB_sigmoid[small_m_AB_inds])
		# fit1_BA = np.polyfit(m_BA[1:][small_m_BA_inds], PA_sigmoid[small_m_BA_inds], 1, w=1/d_PA_sigmoid[small_m_BA_inds])
		# m0_AB = - fit1_AB[1] / fit1_AB[0]
		# m0_BA = - fit1_BA[1] / fit1_BA[0]
		
	if(to_plot_k_distr):
		plot_k_distr(np.exp(ln_k_AB_data) / 1, N_k_bins, 'k_{AB}', units='step')
		plot_k_distr(ln_k_AB_data, N_k_bins, '\ln(k_{AB} \cdot 1step)')
		plot_k_distr(np.exp(ln_k_BA_data) / 1, N_k_bins, 'k_{BA}', units='step')
		plot_k_distr(ln_k_BA_data, N_k_bins, '\ln(k_{BA} \cdot 1step)')
		if(mode == 'BF'):
			plot_k_distr(np.exp(ln_k_bc_AB_data) / 1, N_k_bins, 'k_{bc, AB}', units='step')
			plot_k_distr(ln_k_bc_AB_data, N_k_bins, '\ln(k_{bc, AB} \cdot 1step)')
			plot_k_distr(np.exp(ln_k_bc_BA_data) / 1, N_k_bins, 'k_{bc, BA}', units='step')
			plot_k_distr(ln_k_bc_BA_data, N_k_bins, '\ln(k_{bc, BA} \cdot 1step)')
	
	if(to_plot_committer is None):
		to_plot_committer = (mode == 'FFS')
	else:
		assert(not ((mode == 'BF') and to_plot_committer)), 'ERROR: BF mode used but commiter requested'
	
	if(to_plot_committer):
		fig_PB_log, ax_PB_log = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
		fig_PB, ax_PB = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		fig_PB_sigmoid, ax_PB_sigmoid = my.get_fig(title[interface_mode], r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		
		mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sigmoid, m, PB_AB, interface_mode)
		# ax_PB_log.plot([min(m), max(m)], [1/2] * 2, '--', label='$P = 1/2$')
		# if(interface_mode == 'M'):
			# ax_PB_log.plot([0] * 2, [min(PB_AB), 1], '--', label='$m = 0$')
		
		# ax_PB.plot([min(m), max(m)], [1/2] * 2, '--', label='$P = 1/2$')
		# if(interface_mode == 'M'):
			# ax_PB.plot([0] * 2, [0, 1], '--', label='$m = 0$')
		
		# ax_PB_sigmoid.plot([min(m), max(m)], [0] * 2, '--', label='$P = 1/2$')
		# ax_PB_sigmoid.plot([0] * 2, [min(PB_sigmoid), max(PB_sigmoid)], '--', label='$m = 0$')
		
		plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sgm, h, m_AB[:-1], PB_AB[:-1], d_PB_AB[:-1], 'P_B', my.get_my_color(1))
		plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sgm, h, m_BA[1:], PA_BA[1:], d_PA_BA[1:], 'P_A', my.get_my_color(2))
		
		ax_PB_log.legend()
		ax_PB.legend()
		ax_PB_sigmoid.legend()
	
	izing.init_rand(old_seed)
	
	if(mode == 'FFS'):
		return F, d_F, M_hist_centers, M_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, flux0_AB, d_flux0_AB, flux0_BA, d_flux0_BA, probs_AB, d_probs_AB, probs_BA, d_probs_BA
	elif(mode == 'BF'):
		return F, d_F, M_hist_centers, M_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, ln_k_bc_AB, d_ln_k_bc_AB, ln_k_bc_BA, d_ln_k_bc_BA

def get_init_states(Ns, N_min, N0=1):
	# first value is '1' because it's the size of the states' set from which the initial states for simulations in A will be picked. 
	# Currently we start all the simulations from 'all spins -1', so there is really only 1 initial state, so values >1 will just mean storing copies of the same state
	return np.array([N0] + list(Ns / min(Ns) * N_min), dtype=np.intc)

# =========================== these result in ~equal computation times ===============
# N_init_states = {}
# N_init_states['AB'] = {}
# N_init_states['BA'] = {}
# N_init_states['AB'][10] = {}   # Ni
# N_init_states['BA'][10] = {}   
# N_init_states['AB'][20] = {}   
# N_init_states['BA'][20] = {}   
# N_init_states['AB'][30] = {}   
# N_init_states['BA'][30] = {}   
# N_init_states['AB'][10][-101] = np.array([5.86494782, 0.20780331, 0.64108675, 0.45093819, 0.59296037, 0.63323846, 0.82477221, 1.02855918, 1.90803348, 2.81926314, 1.65642474]) * \
							# np.array([0.97378795, 1.06975615, 1.02141394, 0.97481339, 0.93258517, 1.05343075, 0.94876457, 0.96916528, 0.9942971, 0.98931741, 1.08498797]) * \
							# np.array([34.44184113, 3.56362745, 0.21981169, 0.3781434, 0.3940806, 0.51355653, 0.62098097, 0.82222466, 0.85768272, 0.75359971, 1.46758935]) * \
							# np.array([0.98588286, 1.04962581, 0.99399027, 1.00544771, 1.03247692, 0.99962003, 1.01109933, 1.02817106, 0.91720793, 0.96675774, 1.01633903])   
# N_init_states['BA'][10][-101] = np.array([5.04242695, 0.24306255, 0.64430179, 0.46376104, 0.55123104, 0.67255364, 0.857448, 1.02944648, 1.75315376, 2.67029485, 1.78240891]) * \
							# np.array([1.03900965, 1.02364505, 0.97349883, 0.94509095, 0.9860851, 0.95276988, 0.97621008, 1.03575653, 1.06265366, 1.02110462, 0.9914176]) * \
							# np.array([31.83003551, 3.57493802, 0.23064763, 0.37330525, 0.42065408, 0.55995892, 0.63625423, 0.82914687, 0.7983409, 0.70740199, 1.45439774]) * \
							# np.array([1.01404907, 0.95053083, 1.01731895, 1.01836105, 0.9710529, 0.99439959, 0.98608457, 1.00517813, 1.01599291, 1.01267056, 1.01694163])  

# N_init_states['AB'][10][-117] = np.array([5.86494782, 0.20780331, 0.64108675, 0.45093819, 0.59296037, 0.63323846, 0.82477221, 1.02855918, 1.90803348, 2.81926314, 1.65642474]) * \
							# np.array([0.97378795, 1.06975615, 1.02141394, 0.97481339, 0.93258517, 1.05343075, 0.94876457, 0.96916528, 0.9942971, 0.98931741, 1.08498797]) * \
							# np.array([0.98588286, 1.04962581, 0.99399027, 1.00544771, 1.03247692, 0.99962003, 1.01109933, 1.02817106, 0.91720793, 0.96675774, 1.01633903])  
# N_init_states['BA'][10][-117] = np.array([5.04242695, 0.24306255, 0.64430179, 0.46376104, 0.55123104, 0.67255364, 0.857448, 1.02944648, 1.75315376, 2.67029485, 1.78240891]) * \
							# np.array([1.03900965, 1.02364505, 0.97349883, 0.94509095, 0.9860851, 0.95276988, 0.97621008, 1.03575653, 1.06265366, 1.02110462, 0.9914176]) * \
							# np.array([1.01404907, 0.95053083, 1.01731895, 1.01836105, 0.9710529, 0.99439959, 0.98608457, 1.00517813, 1.01599291, 1.01267056, 1.01694163])

# N_init_states['AB'][20][-117] = np.array([104.58137675, 4.52651586, 0.9389709, 0.43419338, 0.26126731, 0.32707436, 0.33881547, 0.35159559, 0.37967477, 0.43670435, 0.42026171, 0.581579, 0.61943924, 0.68033028, 0.879528, 1.3003374, 1.5022591, 1.99221445, 2.59343701, 2.24145835, 1.49785133])
# N_init_states['BA'][20][-117] = np.array([1.80245815, 5.05678955, 1.10000627, 0.4953472, 0.31701429, 0.38355781, 0.40640283, 0.42864111, 0.47713556, 0.5551159, 0.48965043, 0.61552494, 0.79758504, 0.93919693, 1.24596722, 1.70109637, 1.90688393, 2.65422979, 3.17230448, 2.68977524, 1.73665982])

# N_init_states['AB'][20][-101] = np.array([2.33868958, 0.51789641, 1.42604944, 1.2149675, 0.94619927, 0.5871483, 0.69429377, 0.66489963, 0.51122314, 0.69779038, 0.72645796, 0.77746991, 0.67691482, 0.9581531, 0.98233807, 1.0223547, 1.62734606, 1.98560952, 1.95467, 2.9860634, 0.75066852])
# N_init_states['BA'][20][-101] = np.array([2.17348042, 0.48972788, 1.40017212, 1.20991913, 0.9091747, 0.66442213, 0.70013649, 0.62187844, 0.56822718, 0.63563581, 0.77455908, 0.80039118, 0.68343128, 0.86287252, 1.02853796, 0.99677704, 1.70917241, 2.13442205, 2.27402395, 2.51349462, 0.74694019])

# N_init_states['AB'][20][-105] = np.array([104.58137675, 4.52651586, 0.9389709, 0.43419338, 0.26126731, 0.32707436, 0.33881547, 0.35159559, 0.37967477, 0.43670435, 0.42026171, 0.581579, 0.61943924, 0.68033028, 0.879528, 1.3003374, 1.5022591, 1.99221445, 2.59343701, 2.24145835, 1.49785133]) * \
								# np.array([0.06106036, 0.98176699, 2.30443882, 2.35634132, 3.54067755, 1.8742149, 1.97612934, 1.44598417, 1.68630663, 1.17602421, 1.55196021, 0.95460315, 1.13741294, 0.94035963, 1.05375956, 0.70059487, 0.85913742, 0.638303, 0.71075303, 0.75435223, 0.23751162])
# N_init_states['BA'][20][-105] = np.array([1.80245815, 5.05678955, 1.10000627, 0.4953472, 0.31701429, 0.38355781, 0.40640283, 0.42864111, 0.47713556, 0.5551159, 0.48965043, 0.61552494, 0.79758504, 0.93919693, 1.24596722, 1.70109637, 1.90688393, 2.65422979, 3.17230448, 2.68977524, 1.73665982]) * \
								# np.array([3.68439, 0.80209611, 1.8185674, 1.86597074, 2.86834148, 1.61846811, 1.61488631, 1.10829792, 1.21118591, 0.89695017, 1.31468914, 0.93021863, 0.95640146, 0.71338216, 0.75667659, 0.55211605, 0.71024789, 0.50997712, 0.59888962, 0.66036154, 0.2212385])

# N_init_states['AB'][30][-117] = np.array([59.23722756, 9.19233736, 2.61266382, 1.21618065, 0.76656654, 0.61166114, 0.57317545, 0.51439522, 0.45957972, 0.45246327, 0.43193812, 0.43959542, 0.45809447, 0.44831959, 0.51318388, 0.38675889, 0.49759893, 0.52024278, 0.58332438, 0.65210667, 0.70142344, 0.78572439, 1.10507602, 1.35863037, 1.45500263, 1.50165668, 2.05613283, 2.13398826, 2.29724441, 1.77473994, 0.81564219]) * \
								# np.array([1.01134707, 1.01668083, 0.99770083, 0.99693929, 1.02841981, 1.03022968, 0.9648083, 1.00411582, 1.00274245, 0.98991819, 1.00155996, 0.99328857, 0.93081579, 1.02280934, 0.94989452, 0.98469081, 1.06015326, 1.05398132, 0.99421305, 0.99792504, 1.01645643, 1.16205045, 0.84636068, 0.86532691, 0.96942996, 1.14203607, 0.92927976, 1.05460351, 0.99708638, 1.01788432, 1.02577104])
# N_init_states['BA'][30][-117] = np.array([55.55835609, 8.52465298, 2.38771005, 1.16204589, 0.78008274, 0.59651014, 0.54800459, 0.44661705, 0.44707403, 0.39778446, 0.41945624, 0.40592288, 0.42940058, 0.43707759, 0.45712177, 0.443849, 0.55194654, 0.55967198, 0.62966159, 0.69834416, 0.80396093, 0.96261902, 1.05299106, 1.39556854, 1.53041928, 1.53895228, 2.03392523, 2.28837613, 2.37871075, 1.73388344, 0.829685]) * \
								# np.array([0.00470185, 1.20204266, 1.2353155, 1.1844567, 1.13462661, 1.21798712, 1.14197801, 1.30594286, 1.2326769, 1.2785938, 1.17933553, 1.32219787, 1.162985, 1.2492839, 1.19178171, 1.08671672, 1.1178604, 1.36132457, 1.21534674, 1.17862562, 1.07802381, 1.11321367, 1.17841559, 1.06123281, 1.17551723, 1.32906336, 1.20508691, 1.12206798, 1.20344888, 1.22687804, 1.24334004])

# =========================== these result in ~equal computation times ===============
M_interfaces_table = {}

M_interfaces_table[('CS', 11, 'AB', 20, 4, 100)] = np.array([4, 5, 7, 8, 9, 11, 13, 15, 18, 21, 24, 27, 31, 36, 40, 45, 51, 58, 69, 100], dtype=int)
M_interfaces_table[('CS', 11, 'BA', 20, 4, 100)] = np.array([4, 6, 7, 9, 11, 15, 18, 22, 25, 28, 31, 35, 38, 41, 45, 49, 54, 60, 69, 100], dtype=int)

M_interfaces_table[('M', 11, 'AB', 20, -117, 117)] = np.array([-117, -115, -113, -109, -105, -97, -93, -85, -81, -75, -69, -61, -53, -47, -37, -29, -17, -3, 19, 117], dtype=int)
M_interfaces_table[('M', 11, 'BA', 20, -117, 117)] = np.array([-117, -115, -113, -109, -105, -101, -95, -89, -85, -77, -73, -65, -59, -51, -43, -33, -23, -9, 11, 117], dtype=int)
M_interfaces_table[('M', 11, 'AB', 30, -117, 117)] = np.array([-117, -115, -113, -111, -109, -107, -105, -103, -99, -95, -91, -85, -81, -75, -71, -65, -59, -53, -47, -41, -35, -29, -23, -15,  -7,  1, 11, 27, 51, 117], dtype=int)
M_interfaces_table[('M', 11, 'BA', 30, -117, 117)] = np.array([-117, -115, -113, -111, -109, -107, -105, -101, -99, -95, -91, -87, -83, -77, -73, -69, -63, -59, -53, -49, -43, -37, -31, -25, -17, -9,  1, 15, 49, 117], dtype=int)

M_interfaces_table[('M', 11, 'AB', 13, -101, 101)] = np.array([-101, -97, -93, -89, -79, -69, -53, -35, -9, -3, 3, 9, 101], dtype=int)
M_interfaces_table[('M', 11, 'BA', 13, -101, 101)] = np.array([-101, -97, -93, -89, -81, -69, -53, -37, -9, -3, 3, 9, 101], dtype=int)

M_interfaces_table[('M', 11, 'AB', 20, -101, 101)] = np.array([-101, -99, -97, -95, -93, -91, -89, -85, -81, -77, -73, -65, -59, -51, -43, -33, -23, -9, 11, 101], dtype=int)
M_interfaces_table[('M', 11, 'BA', 20, -101, 101)] = np.array([-101, -99, -97, -95, -93, -91, -89, -87, -83, -77, -73, -69, -61, -53, -45, -37, -25, -13, 11, 101], dtype=int)
M_interfaces_table[('M', 11, 'AB', 20, -91, 91)] = np.array([-91, -89, -87, -85, -83, -81, -79, -77, -73, -69, -63, -59, -53, -45, -37, -27, -17, -3, 15, 91], dtype=int)
M_interfaces_table[('M', 11, 'BA', 20, -91, 91)] = np.array([-91, -89, -87, -85, -83, -81, -79, -77, -75, -71, -67, -63, -55, -51, -43, -33, -23, -9, 11, 91], dtype=int)

M_interfaces_table[('CS', 11, 'AB', 20, 4, 100)] = np.array([4, 5, 7, 8, 9, 11, 13, 15, 18, 21, 24, 27, 31, 36, 40, 45, 51, 58, 69, 100], dtype=int)
M_interfaces_table[('CS', 11, 'BA', 20, 4, 100)] = np.array([4, 6, 7, 9, 11, 15, 18, 22, 25, 28, 31, 35, 38, 41, 45, 49, 54, 60, 69, 100], dtype=int)
M_interfaces_table[('CS', 11, 'AB', 20, 5, 105)] = np.array([5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 19, 23, 26, 31, 36, 41, 47, 55, 67, 105], dtype=int)
M_interfaces_table[('CS', 11, 'BA', 20, 5, 105)] = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 31, 34, 37, 41, 45, 51, 58, 68, 116], dtype=int)
M_interfaces_table[('CS', 11, 'AB', 20, 7, 101)] = np.array([7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 21, 25, 29, 33, 37, 43, 49, 57, 69, 101], dtype=int)
M_interfaces_table[('CS', 11, 'BA', 20, 7, 101)] = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 34, 37, 40, 43, 47, 52, 58, 68, 114], dtype=int)

def main():
	# TODO: remove outdated inputs
	[L, h, Temp, mode, Nt, N_states_FFS, N_init_states_FFS, to_recomp, to_get_EM, verbose, my_seed, N_M_interfaces, N_runs, init_gen_mode, dM_0, dM_max, interface_mode, OP_min_BF, OP_max_BF, Nt_sample_A, Nt_sample_B, def_spin_state, N_spins_up_init, to_plot_ETS], _ = \
		my.parse_args(sys.argv, ['-L', '-h', '-Temp', '-mode', '-Nt', '-N_states_FFS', '-N_init_states_FFS', '-to_recomp', '-to_get_EM', '-verbose', '-my_seed', '-N_M_interfaces', '-N_runs', '-init_gen_mode', '-dM_0', '-dM_max', '-interface_mode', '-OP_min_BF', '-OP_max_BF', '-Nt_sample_A', '-Nt_sample_B', '-def_spin_state', '-N_spins_up_init', '-to_plot_ETS'], \
					  possible_arg_numbers=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], \
					  default_values=[['11'], ['-0.01'], ['2.1'], ['BF'], ['-1000000'], ['-5000'], ['10000'], [my.no_flags[0]], ['1'], ['1'], ['23'], ['-30'], ['3'], ['-3'], ['20'], [None], ['M'], [None], [None], ['-1000000'], ['-1000000'], ['-1'], [None], [my.no_flags[0]]])
	
	# efficiency info:
	# [M0=20, NM=20] : Nt / N_states_FFS ~ 1e4
	
	L = int(L[0])
	L2 = L**2
	h = float(h[0])   # arbitrary small number that gives nice pictures
	
	set_OP_defaults(L2)

	my_seed = int(my_seed[0])
	to_recomp = (to_recomp[0] in my.yes_flags)
	mode = mode[0]
	to_get_EM = (to_get_EM[0] in my.yes_flags)
	verbose = int(verbose[0])
	#N_init_states_FFS = int(N_init_states_FFS[0])
	Nt = int(Nt[0])
	N_states_FFS = int(N_states_FFS[0])
	N_M_interfaces = int(N_M_interfaces[0])
	N_runs = int(N_runs[0])
	init_gen_mode = int(init_gen_mode[0])
	dM_0 = int(dM_0[0])
	dM_max = dM_max[0]
	interface_mode = interface_mode[0]
	def_spin_state = int(def_spin_state[0])
	N_spins_up_init = (None if(N_spins_up_init[0] is None) else int(N_spins_up_init[0]))
	to_plot_ETS = (to_plot_ETS[0] in my.yes_flags)
	
	OP_min_BF = OP_min_default[interface_mode] if(OP_min_BF[0] is None) else int(OP_min_BF[0])
	OP_max_BF = OP_max_default[interface_mode] if(OP_max_BF[0] is None) else int(OP_max_BF[0])

	# -------- T < Tc, transitions ---------
	Temp = float(Temp[0])  # T_c = 2.27
	# -------- T > Tc, fluctuations around 0 ---------
	#Temp = 3.0
	#Temp = 2.5   # looks like Tc for L=16
	
	print('T=%lf, L=%d, h=%lf, mode=%s, Nt=%d, N_states_FFS=%d, N_M_interfaces=%d, N_runs=%d, init_gen_mode=%d, dM_0=%d, dM_max=%s, def_spin_state=%d, N_spins_up_init=%s, OP_min_BF=%d, OP_max_BF=%d, to_get_EM=%r, verbose=%d, my_seed=%d, to_recomp=%r' % \
		(Temp, L, h, mode, Nt, N_states_FFS, N_M_interfaces, N_runs, init_gen_mode, dM_0, str(dM_max), def_spin_state, str(N_spins_up_init), OP_min_BF, OP_max_BF, to_get_EM, verbose, my_seed, to_recomp))
	
	izing.init_rand(my_seed)
	izing.set_verbose(verbose)

	Nt, N_states_FFS = \
		tuple([((int(-n * np.exp(3 - dE_avg/Temp)) + 1) if(n < 0) else n) for n in \
				[Nt, N_states_FFS]])

	if(interface_mode == 'M'):
		M_0 = -L2 + dM_0
		if(dM_max is None):
			M_max = -M_0
		else:
			dM_max = int(dM_max)
			M_max = L2 - dM_max
	elif(interface_mode == 'CS'):
		M_0 = dM_0
		assert(dM_max is not None), 'ERROR: no dM_max provided for CS interface_mode'
		dM_max = int(dM_max)
		M_max = L**2 - dM_max

	if('FFS' in mode):
		M_left = OP_min_default[interface_mode]
		M_right = OP_max_default[interface_mode] - 1
		if(interface_mode == 'M'):
			M_match_BF_A_to = M_0 + int(L2 * 0.25)   # this should be sufficiently < M_sample_to, because BF_profile is distorted near the M_sample_to; But is also should be > OP_interface[1] since we need enough points to match F curves.
			M_sample_BF_A_to = M_match_BF_A_to + int(L2 * 0.2)
			M_match_BF_B_to = M_max - int(L2 * 0.25)
			M_sample_BF_B_to = M_match_BF_B_to - int(L2 * 0.2)
		elif(interface_mode == 'CS'):
			M_match_BF_A_to = M_0 + 20   # this should be sufficiently < M_sample_to, because BF_profile is distorted near the M_sample_to; But is also should be > OP_interface[1] since we need enough points to match F curves.
			M_sample_BF_A_to = M_match_BF_A_to + 20
			M_match_BF_B_to = M_max - 30
			M_sample_BF_B_to = M_match_BF_B_to - 20
			
		else:
			print('ERROR: %s interface_mode is not supported' % (interface_mode))
			return

		# =========== if the necessary set in not in the table - this will work and nothing will erase its results ========
		M_interfaces_AB = M_0 + np.round(np.arange(abs(N_M_interfaces)) * (M_max - M_0) / (abs(N_M_interfaces) - 1) / 2) * 2
			# this gives Ms such that there are always pairs +-M[i], so flipping this does not move the intefraces, which is (is it?) good for backwords FFS (B->A)
		M_interfaces_BA = flip_M(M_interfaces_AB, L2, interface_mode)

		if(N_M_interfaces > 0):
			interface_AB_UID = (interface_mode, L, 'AB' if(h < 0) else 'BA', N_M_interfaces, M_0, M_max)
			interface_BA_UID = (interface_mode, L, 'BA' if(h < 0) else 'AB', N_M_interfaces, M_0, M_max)
			if(interface_AB_UID in M_interfaces_table):
				M_interfaces_AB = M_interfaces_table[interface_AB_UID]
			if(interface_BA_UID in M_interfaces_table):
				M_interfaces_BA = M_interfaces_table[interface_BA_UID]
			
		N_M_interfaces = len(M_interfaces_AB)
		
		N_init_states_AB = np.ones(N_M_interfaces, dtype=np.intc) * N_states_FFS
		N_init_states_BA = np.ones(N_M_interfaces, dtype=np.intc) * N_states_FFS
		
		print('Mi_AB:', M_interfaces_AB)
		print('Mi_BA:', M_interfaces_BA)
		
	if(mode == 'BF_1'):
		proc_T(L, Temp, h, Nt, interface_mode, def_spin_state, \
				OP_A=M_0, OP_B=M_max, N_spins_up_init=N_spins_up_init, \
				to_plot_time_evol=True, to_plot_F=True, to_plot_ETS=to_plot_ETS, to_plot_correlations=True, \
				OP_min=OP_min_BF, OP_max=OP_max_BF, to_estimate_k=True)
		
	elif(mode == 'BF_many'):
		run_many(L, Temp, h, N_runs, interface_mode, def_spin_state, Nt_per_BF_run=Nt, \
				OP_A=M_0, OP_B=M_max, N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=int(np.round(np.sqrt(N_runs) / 2) + 1), \
				mode='BF')
	
	elif(mode == 'FFS_AB'):
		proc_FFS_AB(L, Temp, h, N_init_states_AB, M_interfaces_AB, M_sample_BF_A_to, M_match_BF_A_to, \
					interface_mode, def_spin_state, \
					to_get_EM=to_get_EM, init_gen_mode=init_gen_mode, to_plot_time_evol=True, to_plot_hists=True)
		
	elif(mode == 'FFS'):
		proc_FFS(L, Temp, h, \
				N_init_states_AB, N_init_states_BA, \
				M_interfaces_AB, M_interfaces_BA, \
				M_sample_BF_A_to, M_sample_BF_B_to, \
				M_match_BF_A_to, M_match_BF_B_to, \
				interface_mode, def_spin_state, \
				init_gen_mode=init_gen_mode, \
				to_plot_hists=True)
		
	elif(mode == 'FFS_many'):
		N_k_bins = int(np.round(np.sqrt(N_runs) / 2)) + 1
		run_many(L, Temp, h, N_runs, interface_mode, def_spin_state, \
				M_interfaces_AB, M_interfaces_BA, \
				M_sample_BF_A_to=M_sample_BF_A_to, M_sample_BF_B_to=M_sample_BF_B_to, \
				M_match_BF_A_to=M_match_BF_A_to, M_match_BF_B_to=M_match_BF_B_to, \
				N_init_states_AB=N_init_states_AB, N_init_states_BA=N_init_states_BA, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, mode='FFS', \
				init_gen_mode=init_gen_mode)
	
	elif(mode in ['FFS_BF_compare', 'BF_FFS_compare']):
		BC_cluster_size = L2 / np.pi
		N_k_bins = int(np.round(np.sqrt(N_runs) / 2)) + 1
		F_FFS, d_F_FFS, M_hist_centers_FFS, M_hist_lens_FFS, ln_k_AB_FFS, d_ln_k_AB_FFS, ln_k_BA_FFS, d_ln_k_BA_FFS, flux0_AB_FFS, d_flux0_AB_FFS, flux0_BA_FFS, d_flux0_BA_FFS, prob_AB_FFS, d_prob_AB_FFS, prob_BA_FFS, d_prob_BA_FFS = \
			run_many(L, Temp, h, N_runs, interface_mode, def_spin_state, \
				N_init_states_AB=N_init_states_AB, \
				N_init_states_BA=N_init_states_BA, \
				M_interfaces_AB=M_interfaces_AB, \
				M_interfaces_BA=M_interfaces_BA, \
				to_plot_k_distr=False, N_k_bins=N_k_bins, \
				mode='FFS', init_gen_mode=init_gen_mode, \
				M_sample_BF_A_to=M_sample_BF_A_to, M_sample_BF_B_to=M_sample_BF_B_to, \
				M_match_BF_A_to=M_match_BF_A_to, M_match_BF_B_to=M_match_BF_B_to)

		F_BF, d_F_BF, M_hist_centers_BF, M_hist_lens_BF, ln_k_AB_BF, d_ln_k_AB_BF, ln_k_BA_BF, d_ln_k_BA_BF, ln_k_bc_AB_BF, d_ln_k_bc_AB_BF, ln_k_bc_BA_BF, d_ln_k_bc_BA_BF = \
			run_many(L, Temp, h, N_runs, interface_mode, def_spin_state, \
					OP_A=M_0, OP_B=M_max, \
					Nt_per_BF_run=Nt, mode='BF', \
					to_plot_k_distr=False, N_k_bins=N_k_bins)
		
		k_AB_FFS_mean, k_AB_FFS_min, k_AB_FFS_max, d_k_AB_FFS = get_log_errors(ln_k_AB_FFS, d_ln_k_AB_FFS, lbl='k_AB_FFS')
		k_BA_FFS_mean, k_BA_FFS_min, k_BA_FFS_max, d_k_BA_FFS = get_log_errors(ln_k_BA_FFS, d_ln_k_BA_FFS, lbl='k_BA_FFS')
		k_AB_BF_mean, k_AB_BF_min, k_AB_BF_max, d_k_AB_BF = get_log_errors(ln_k_AB_BF, d_ln_k_AB_BF, lbl='k_AB_BF')
		k_BA_BF_mean, k_BA_BF_min, k_BA_BF_max, d_k_BA_BF = get_log_errors(ln_k_BA_BF, d_ln_k_BA_BF, lbl='k_BA_BF')
		k_bc_AB_BF_mean, k_bc_AB_BF_min, k_bc_AB_BF_max, d_k_bc_AB_BF = get_log_errors(ln_k_bc_AB_BF, d_ln_k_bc_AB_BF, lbl='k_bc_AB_BF')
		k_bc_BA_BF_mean, k_bc_BA_BF_min, k_bc_BA_BF_max, d_k_bc_BA_BF = get_log_errors(ln_k_bc_BA_BF, d_ln_k_bc_BA_BF, lbl='k_bc_BA_BF')
		
		#assert(np.all(abs(M_hist_centers_BF - M_hist_centers_FFS) < (M_hist_centers_BF[0] - M_hist_centers_BF[1]) * 1e-3)), 'ERROR: returned M_centers do not match'
		#assert(np.all(abs(M_hist_lens_BF - M_hist_lens_FFS) < M_hist_lens_BF[0] * 1e-3)), 'ERROR: returned M_centers do not match'
		
		fig_F, ax_F = my.get_fig(title[interface_mode], r'$F / J$', title=r'$F(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_F.errorbar(M_hist_centers_BF, F_BF - min(F_BF), yerr=d_F_BF, label='BF')
		ax_F.errorbar(M_hist_centers_FFS, F_FFS, yerr=d_F_FFS, label='FFS')
		if(interface_mode == 'CS'):
			ax_F.plot([BC_cluster_size] * 2, [min(F_FFS), max(F_FFS)], '--', label='$S_{BC} = L^2 / \pi$')
		ax_F.legend()
		
	elif(mode == 'XXX'):
		fig_E, ax_E = my.get_fig('T', '$E/L^2$', xscl='log')
		fig_M, ax_M = my.get_fig('T', '$M/L^2$', xscl='log')
		fig_C, ax_C = my.get_fig('T', '$C/L^2$', xscl='log', yscl='log')
		fig_M2, ax_M2 = my.get_fig('T', '$(M/L^2)^2$')
		
		T_arr = np.power(10, np.linspace(T_min_log10, T_max_log10, N_T))
		
		for n in N_s:
			T_arr = proc_N(n, Nt, my_seed, ax_C=ax_C, ax_E=ax_E, ax_M=ax_M, ax_M2=ax_M2, to_recomp=to_recomp, T_min_log10=T_min_log10, T_max_log10=T_max_log10, N_T=N_T)
		
		ax_E.plot(T_arr, -np.tanh(1/T_arr)*2, '--', label='$-2 th(1/T)$')
		T_simulated_ind = (min(T_arr) < T_table) & (T_table < max(T_arr))
		ax_E.plot(T_table[T_simulated_ind], E_table[T_simulated_ind], '.', label='Onsager')

		ax_C.legend()
		ax_E.legend()
		ax_M.legend()
	
	else:
		print('ERROR: mode=%s is not supported' % (mode))

	plt.show()

if(__name__ == "__main__"):
	main()

