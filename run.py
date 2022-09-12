import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import sys
import scipy.interpolate

'''
TODO:
1) find 'jumps_distr(L)' for BF
2) find L0 : V M > L0 |j_distr(M) - j_distr(L0)| < eps
3) get smallest_ok_dist_between_interfaces = max(jumps(L0 + 1))
4) find L1, N_OP : min(OP_interfaces_dists) > smallest_ok_dist_between_interfaces
*) verify CS as an OP
	'k_AB by CS' should be ~= 'k_AB by M'
**) find the "big" L from Nc(L)
***) use CS_FFS for Nc(h) and k_AB(h)
'''

import mylib as my
import table_data

# ========================== general params ==================
dE_avg = 6   # J
print_scale_k = 1e5
print_scale_flux = 1e2
OP_min_default = {}
OP_max_default = {}
OP_peak_default = {}
OP_scale = {}
dOP_step = {}
OP_std_overestimate = {}
OP_hist_edges_default = {}
OP_possible_jumps = {}

OP_C_id = {}
OP_C_id['M'] = 0
OP_C_id['CS'] = 1

OP_step = {}
OP_step['M'] = 2
OP_step['CS'] = 1

title = {}
title['M'] = r'$m = M / L^2$'
title['CS'] = r'cluster size'

OP_peak_guess = {}

OP_fit2_width = {}
OP_fit2_width['M'] = 0.2
OP_fit2_width['CS'] = 12

feature_label = {}
feature_label['M'] = 'm'
feature_label['CS'] = 'Scl'


def set_OP_defaults(L2):
	OP_min_default['M'] = -L2 - 1
	OP_max_default['M'] = L2 + 1
	OP_peak_default['M'] = L2 % 2
	OP_scale['M'] = L2
	dOP_step['M'] = OP_step['M'] / OP_scale['M']
	OP_std_overestimate['M'] = 2 * L2
	OP_possible_jumps['M'] = np.array([-2, 2])
	OP_peak_guess['M'] = 0
	
	OP_min_default['CS'] = -1
	OP_max_default['CS'] = L2 + 1
	OP_peak_default['CS'] = L2 // 2
	OP_scale['CS'] = 1
	dOP_step['CS'] = OP_step['CS'] / OP_scale['CS']
	OP_std_overestimate['CS'] = L2
	OP_possible_jumps['CS'] = np.arange(-L2 // 2 - 1, L2 // 2 + 1)
	OP_peak_guess['CS'] = L2 // 2

# ========================== recompile ==================
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

def get_log_errors(l, dl, lbl=None, err0=0.3, units='1/step', print_scale=1):
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
	
	print_scale_str = ('' if(print_scale == 1) else ('%1.1e' % (1/print_scale)))
	
	if(lbl is not None):
		if(dl > err0):
			print(lbl + '   \\in   [' + '; '.join(my.same_scale_strs([x_low * print_scale, x_up * print_scale], x_to_compare_to=[(x_up - x_low) * print_scale])) + ']' + print_scale_str + '  ' + units + '  with ~68% prob')
		else:
			print('%s   =   (%s)%s  %s,    dx/x ~ %s' % (lbl, my.errorbar_str(x * print_scale, d_x * print_scale), print_scale_str, units, my.f2s(dl, 2)))
	
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
	
def flip_OP(OP, L2, interface_mode):
	if(interface_mode == 'M'):
		return -np.flip(OP)
	elif(interface_mode == 'CS'):
		return L2 - np.flip(OP)

def get_sigmoid_fit(PB, d_PB, OP, h, Nc_data, fit_w=None):
	#OP0_guess = OP_peak_guess[interface_mode] - h * xi / OP_scale[interface_mode]
	OP0_guess = Nc_data[1, np.argmin(abs(Nc_data[0, :] - h))]
	if(fit_w is None):
		fit_w = abs(OP[0] - OP0_guess) / 5
	
	ok_inds = PB < 1
	N_OP = len(OP)
	PB_sigmoid = np.empty(N_OP)
	d_PB_sigmoid = np.empty(N_OP)
	
	PB_sigmoid[ok_inds] = -np.log(1 / PB[ok_inds] - 1)
	d_PB_sigmoid[ok_inds] = d_PB[ok_inds] / (PB[ok_inds] * (1 - PB[ok_inds]))
	PB_sigmoid[~ok_inds] = max(PB_sigmoid[ok_inds]) * 1.1
	d_PB_sigmoid[~ok_inds] = max(d_PB_sigmoid[ok_inds]) * 1.1
	
	linfit_inds = (OP0_guess - fit_w < OP) & (OP < OP0_guess + fit_w) & ok_inds
	N_min_points_for_fit = 2
	if(np.sum(linfit_inds) < N_min_points_for_fit):
		dOP = abs(OP - OP0_guess)
		linfit_inds = (dOP <= np.sort(dOP)[N_min_points_for_fit - 1])
	linfit = np.polyfit(OP[linfit_inds], PB_sigmoid[linfit_inds], 1, w=1/d_PB_sigmoid[linfit_inds])
	OP0 = - linfit[1] / linfit[0]
	
	return PB_sigmoid, d_PB_sigmoid, linfit, linfit_inds, OP0

def plot_PB_AB(ax, ax_log, ax_sgm, h, title, x_lbl, OP, PB, d_PB=None, PB_sgm=None, d_PB_sgm=None, linfit=None, linfit_inds=None, OP0=None, d_OP0=None, clr=None, N_fine_points=-10):
	ax_log.errorbar(OP, PB, yerr=d_PB, fmt='.', label='$' + title + '$', color=clr)
	
	OP0_str = x_lbl + '_{1/2} = ' + (my.f2s(OP0) if(d_OP0 is None) else my.errorbar_str(OP0, d_OP0))
	ax.errorbar(OP, PB, yerr=d_PB, fmt='.', label='$' + title + '$', color=clr)
	if(linfit_inds is not None):
		OP_near_OP0 = OP[linfit_inds]
		N_fine_points_near_OP0 = N_fine_points if(N_fine_points > 0) else int(-(max(OP_near_OP0) - min(OP_near_OP0)) / min(OP_near_OP0[1:] - OP_near_OP0[:-1]) * N_fine_points)
		OP_near_OP0_fine = np.linspace(min(OP_near_OP0), max(OP_near_OP0), N_fine_points_near_OP0)
		if(linfit is not None):
			sigmoid_points_near_OP0 = 1 / (1 + np.exp((OP0 - OP_near_OP0_fine) * linfit[0]))
			ax.plot(OP_near_OP0_fine, sigmoid_points_near_OP0, label='$' + OP0_str + '$', color=clr)
			ax_log.plot(OP_near_OP0_fine, sigmoid_points_near_OP0, label='$' + OP0_str + '$', color=clr)
	
	if(PB_sgm is not None):
		ax_sgm.errorbar(OP, PB_sgm, yerr=d_PB_sgm, fmt='.', label='$' + title + '$', color=clr)
		if(linfit_inds is not None):
			ax_sgm.plot(OP_near_OP0_fine, np.polyval(linfit, OP_near_OP0_fine), label='$' + OP0_str + '$', color=clr)
	# if(N_fine_points < 0):
		# N_fine_points = int(-(max(OP) - min(OP)) / min(OP[1:] - OP[:-1]) * N_fine_points)
	#m_fine = np.linspace(min(OP), max(OP), N_fine_points)
	#ax_PB.plot(m_AB_fine[:-1], PB_opt_fnc(m_AB_fine[:-1]), label='$P_{B, fit}$', color=my.get_my_color(1))
	#ax_PB_log.plot(m_AB_fine[:-1], PB_opt_fnc(m_AB_fine[:-1]), label='$P_{B, fit}$', color=my.get_my_color(1))
	#ax_PB_sigmoid.plot(m_AB_fine[:-1], -np.log(PB_fit_b / (PB_opt_fnc(m_AB_fine[:-1]) - PB_fit_a) - 1), label='$P_{B, fit}$', color=my.get_my_color(1))

def mark_PB_plot(ax, ax_log, ax_sgm, OP, PB, PB_sgm, interface_mode):
	ax_log.plot([min(OP), max(OP)], [1/2] * 2, '--', label='$P = 1/2$')
	ax.plot([min(OP), max(OP)], [1/2] * 2, '--', label='$P = 1/2$')
	ax_sgm.plot([min(OP), max(OP)], [0] * 2, '--', label='$P = 1/2$')
	
	if(interface_mode == 'M'):
		ax_sgm.plot([0] * 2, [min(PB_sgm), max(PB_sgm)], '--', label='$m = 0$')
		ax_log.plot([0] * 2, [min(PB), 1], '--', label='$m = 0$')
		ax.plot([0] * 2, [0, 1], '--', label='$m = 0$')

def proc_FFS(L, Temp, h, \
			N_init_states_AB, N_init_states_BA, \
			OP_interfaces_AB, OP_interfaces_BA, \
			OP_sample_BF_A_to, OP_sample_BF_B_to, \
			OP_match_BF_A_to, OP_match_BF_B_to, \
			interface_mode, def_spin_state, \
			init_gen_mode=-2, to_get_timeevol=False, \
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
	
	N_init_states_AB : np.array(N_OP_interfaces + 1, int)
		The number of states to save on each interface [OP_0, M_1, ..., M_N-1, L2]. 
		See 'OP_interfaces' for more info
	N_init_states_BA : np.array(N_OP_interfaces + 1, int)
		same as N_init_states_AB, but for BA process
	
	OP_interfaces_AB : np.array(N_OP_interfaces + 2, float)
		All the interfaces for the run. +2 because theare is an interface at -L2-1 and at +L2 to formally complete the picture
		So, for L=11, OP_0 = -101, N_OP_interfaces=10, the interfaces might be:
		-122, -101, -79, -57, -33, -11, 11, 33, 57, 79, 101, 121
	
	OP_interfaces_BA : 
		same as N_init_states_AB, but for BA process
	
	verbose : int, (None)
		How hoisy the computation is, {0, 1, 2, 3} are now possible
		None means the verbosity will be taken from the global value, which is a part of the namespace program state in the 'izing' module
		A number means it will be passed to all the deeper functions and used imstead of the default-state value
	
	to_plot_hists : bool (True)
		whether or not to plot F profile and other related plots
	
	init_gen_mode : int (-2)
		The way to generate initial states in A to be propagated towards OP_0
		-2 - generate an ensemble and in A and sample from this ensemble
		-1 - generate each spin randomly 50/50 to be +-1
		>=0 - generate all spins -1 and then set |mode| random (uniformly distributed over all lattice points) spins to +1
	
	interface_mode : str, ('M'), {'M', 'CS'}
		What is used as the order parameter
	
	Returns
	-------
	1: e^<log> ~= <x>
	2: e^(<log> - d_<log>)
	3: e^(<log> + d_<log>)
	4: e^<log> * d_<log> ~= dx
	"""
	L2 = L**2
	
	if(interface_mode == 'CS'):
		print('ERROR: CS mode is not supported for FFS (only FFS_AB is supported)')
	
	# We need to move M_far_from_B, because the simulation is actually done with the reversed field, which is the same as reversed order-parameter (OP). 
	# But 'M_far_from_B' is set in the original OP, so we need to move the 'M_far_from_B' from the original OP axis to the reversed axis.
	OP_sample_BF_B_to = flip_OP(OP_sample_BF_B_to, L2, interface_mode)
	OP_match_BF_B_to = flip_OP(OP_match_BF_B_to, L2, interface_mode)

	probs_AB, d_probs_AB, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, rho_AB, d_rho_AB, OP_hist_centers_AB, OP_hist_lens_AB, \
		PB, d_PB, PB_sigmoid, d_PB_sigmoid, linfit_AB, linfit_AB_inds, OP0_AB = \
			proc_FFS_AB(L, Temp, h, N_init_states_AB, OP_interfaces_AB, interface_mode, def_spin_state, \
					OP_sample_BF_A_to, OP_match_BF_A_to, \
					init_gen_mode=init_gen_mode, to_get_timeevol=to_get_timeevol, verbose=verbose)
	
	probs_BA, d_probs_BA, ln_k_BA, d_ln_k_BA, flux0_BA, d_flux0_BA, rho_BA, d_rho_BA, OP_hist_centers_BA, OP_hist_lens_BA, \
		PA, d_PA, PA_sigmoid, d_PA_sigmoid, linfit_BA, linfit_BA_inds, OP0_BA= \
			proc_FFS_AB(L, Temp, -h, N_init_states_BA, OP_interfaces_BA, interface_mode, def_spin_state, \
					OP_sample_BF_B_to, OP_match_BF_B_to, \
					init_gen_mode=init_gen_mode, to_get_timeevol=to_get_timeevol, verbose=verbose)
	
	# we need to reverse back all the OP-dependent returns sicnce the simulation was done with the reversed OP
	probs_BA = np.flip(probs_BA)
	d_probs_BA = np.flip(d_probs_BA)
	OP_hist_lens_BA = np.flip(OP_hist_lens_BA)
	N_init_states_BA = np.flip(N_init_states_BA)
	PA = np.flip(PA)
	d_PA = np.flip(d_PA)
	PA_sigmoid = np.flip(PA_sigmoid)
	d_PA_sigmoid = np.flip(d_PA_sigmoid)
	linfit_BA_inds = np.flip(linfit_BA_inds)
	linfit_BA[0] = -linfit_BA[0]   #   TODO: this has to be interface_mode-spesific
	if(to_get_timeevol):
		rho_BA = np.flip(rho_BA)
		d_rho_BA = np.flip(d_rho_BA)
		OP_hist_centers_BA = flip_OP(OP_hist_centers_BA, L2, interface_mode)

	OP_interfaces_BA = flip_OP(OP_interfaces_BA, L2, interface_mode)
	OP0_BA = flip_OP(OP0_BA, L2, interface_mode)
	
	#assert(np.all(abs(OP_hist_centers_AB - OP_hist_centers_BA) < 1 / L**2 * 1e-3))
	#assert(np.all(abs(OP_hist_lens_AB - OP_hist_lens_BA) < 1 / L**2 * 1e-3))
	
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
	
	if(to_get_timeevol):
		rho_AB_interp1d = scipy.interpolate.interp1d(OP_hist_centers_AB, rho_AB, fill_value='extrapolate')
		d_rho_AB_interp1d = scipy.interpolate.interp1d(OP_hist_centers_AB, d_rho_AB, fill_value='extrapolate')
		rho_BA_interp1d = scipy.interpolate.interp1d(OP_hist_centers_BA, rho_BA, fill_value='extrapolate')
		d_rho_BA_interp1d = scipy.interpolate.interp1d(OP_hist_centers_BA, d_rho_BA, fill_value='extrapolate')
		OP = np.sort(np.unique(np.array(list(OP_hist_centers_AB) + list(OP_hist_centers_BA))))
		OP_lens = np.empty(len(OP))
		OP_lens[0] = OP[1] - OP[0]
		OP_lens[-1] = OP[-1] - OP[-2]
		OP_lens[1:-1] = (OP[2:] - OP[:-2]) / 2
		rho_fnc = lambda x: p_AB * flux0_AB * rho_AB_interp1d(x) + p_BA * flux0_BA * rho_BA_interp1d(x)
		d_rho_fnc = lambda x: np.sqrt((p_AB * flux0_AB * rho_AB_interp1d(x))**2 * ((d_p_AB / p_AB)**2 + (d_flux0_AB / flux0_AB)**2 + (d_rho_AB_interp1d(x) / rho_AB_interp1d(x))**2) + \
				  (p_BA * flux0_BA * rho_BA_interp1d(x))**2 * ((d_p_BA / p_BA)**2 + (d_flux0_BA / flux0_BA)**2 + (d_rho_BA_interp1d(x) / rho_BA_interp1d(x))**2))
		rho = rho_fnc(OP)
		d_rho = d_rho_fnc(OP)
		
		F = -Temp * np.log(rho * OP_lens)
		d_F = Temp * d_rho / rho
		#F = F - min(F)
		F = F - F[0]
	
		if(to_plot_hists):
			fig_F, ax_F = my.get_fig(title[interface_mode], r'$F(%s) = -T \ln(\rho(%s) \cdot d%s(%s))$' % (feature_label[interface_mode], feature_label[interface_mode], feature_label[interface_mode], feature_label[interface_mode]), title=r'$F(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_F.errorbar(OP, F, yerr=d_F, fmt='.', label='FFS data')
			ax_F.legend()
	else:
		F = None
		d_F = None
		OP = None
		OP_lens = None
		rho_fnc = None
		d_rho_fnc = None
	
	return F, d_F, OP, OP_lens, rho_fnc, d_rho_fnc, \
			probs_AB, d_probs_AB, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, probs_BA, d_probs_BA, ln_k_BA, d_ln_k_BA, flux0_BA, d_flux0_BA, \
			PB, d_PB, PA, d_PA, PB_sigmoid, d_PB_sigmoid, PA_sigmoid, d_PA_sigmoid, linfit_AB, linfit_BA, linfit_AB_inds, linfit_BA_inds, OP0_AB, OP0_BA

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

def proc_order_parameter_FFS(L, Temp, h, flux0, d_flux0, probs, \
							d_probs, OP_interfaces, N_init_states, \
							interface_mode, def_spin_state, \
							OP_data=None, Nt=None, OP_hist_edges=None, memory_time=1, \
							to_plot_time_evol=False, to_plot_hists=False, stride=1, \
							x_lbl='', y_lbl='', Ms_alpha=0.5, OP_match_BF_to=None, states=None):
	L2 = L**2
	ln_k_AB = np.log(flux0 * 1) + np.sum(np.log(probs))   # [flux0 * 1] = 1, because [flux] = 1/time = 1/step
	d_ln_k_AB = np.sqrt((d_flux0 / flux0)**2 + np.sum((d_probs / probs)**2))
	
	print('Nt:', Nt)
	#print('N_init_guess: ', np.exp(np.mean(np.log(Nt))) / Nt)
	print('flux0 = (%s) 1/step' % (my.errorbar_str(flux0, d_flux0)))
	print('-log10(P_i):', -np.log(probs) / np.log(10))
	print('d_P_i / P_i:', d_probs / probs)
	k_AB, k_AB_low, k_AB_up, d_k_AB = get_log_errors(ln_k_AB, d_ln_k_AB, lbl='k_AB', print_scale=print_scale_k)
	
	OP_interfaces_scaled = OP_interfaces / OP_scale[interface_mode]
	
	N_OP_interfaces = len(OP_interfaces_scaled)
	P_B = np.empty(N_OP_interfaces)
	d_P_B = np.empty(N_OP_interfaces)
	P_B[N_OP_interfaces - 1] = 1
	d_P_B[N_OP_interfaces - 1] = 0
	for i in range(N_OP_interfaces - 2, -1, -1):
		P_B[i] = P_B[i + 1] * probs[i]
		d_P_B[i] = P_B[i] * (1 - P_B[i]) * np.sqrt((d_P_B[i + 1] / P_B[i + 1])**2 + (d_probs[i] / probs[i])**2) if(np.all(d_probs[i] / probs[i] < 10)) else P_B[i] * 0.99
	
	#P_B_opt, P_B_opt_fnc, P_B_opt_fnc_inv = PB_fit_optimize(P_B[:-1], d_P_B[:-1], OP_interfaces_scaled[:-1], OP_interfaces_scaled[0], OP_interfaces_scaled[-1], P_B[0], P_B[-1], tht0=[0, 0.15])
	#P_B_x0_opt = P_B_opt.x[0]
	#P_B_s_opt = P_B_opt.x[1]
	P_B_sigmoid, d_P_B_sigmoid, linfit, linfit_inds, OP0 = \
		get_sigmoid_fit(P_B[:-1], d_P_B[:-1], OP_interfaces_scaled[:-1], h, table_data.Nc_reference_data[str(Temp)])
	
	OP_optimize_f_interp_vals = 1 - np.log(P_B) / np.log(P_B[0])
	d_OP_optimize_f_interp_vals = d_P_B / P_B / np.log(P_B[0])
	#M_optimize_f_interp_fit = lambda OP: (1 - np.log(P_B_opt_fnc(OP)) / np.log(P_B[0]))
	OP_optimize_f_interp_interp1d = scipy.interpolate.interp1d(OP_interfaces_scaled, OP_optimize_f_interp_vals, fill_value='extrapolate')
	OP_optimize_f_desired = np.linspace(0, 1, N_OP_interfaces)
	
	# ===== numerical inverse ======
	OP_optimize_new_OP_interfaces = np.zeros(N_OP_interfaces, dtype=np.intc)
	OP_optimize_new_OP_guess = min(OP_interfaces_scaled) + (max(OP_interfaces_scaled) - min(OP_interfaces_scaled)) * (np.arange(N_OP_interfaces) / N_OP_interfaces)
	for i in range(N_OP_interfaces - 2):
		OP_optimize_new_OP_interfaces[i + 1] = int(np.round(scipy.optimize.root(lambda x: OP_optimize_f_interp_interp1d(x) - OP_optimize_f_desired[i + 1], OP_optimize_new_OP_guess[i + 1]).x[0] * OP_scale[interface_mode]) + 0.1)
	
	if(interface_mode == 'M'):
		OP_optimize_new_OP_interfaces = OP_interfaces[0] + 2 * np.round((OP_optimize_new_OP_interfaces - OP_interfaces[0]) / 2)
		OP_optimize_new_OP_interfaces = np.int_(OP_optimize_new_OP_interfaces + 0.1 * np.sign(OP_optimize_new_OP_interfaces))
	elif(interface_mode == 'CS'):
		OP_optimize_new_OP_interfaces = np.int_(np.round(OP_optimize_new_OP_interfaces))
	OP_optimize_new_OP_interfaces[0] = OP_interfaces[0]
	OP_optimize_new_OP_interfaces[-1] = OP_interfaces[-1]
	
	OP_optimize_new_OP_interfaces_original = np.copy(OP_optimize_new_OP_interfaces)
	if(np.any(OP_optimize_new_OP_interfaces[1:] == OP_optimize_new_OP_interfaces[:-1])):
		print('WARNING: duplicate interfaces, removing duplicates')
		for i in range(N_OP_interfaces - 1):
			if(OP_optimize_new_OP_interfaces[i+1] <= OP_optimize_new_OP_interfaces[i]):
				if(i == 0):
					OP_optimize_new_OP_interfaces[1] = OP_optimize_new_OP_interfaces[0] + OP_step[interface_mode]
				else:
					if(OP_optimize_new_OP_interfaces[i-1] + OP_step[interface_mode] < OP_optimize_new_OP_interfaces[i] * (1 - 1e-5 * np.sign(OP_optimize_new_OP_interfaces[i]))):
						OP_optimize_new_OP_interfaces[i] = OP_optimize_new_OP_interfaces[i-1] + OP_step[interface_mode]
					else:
						OP_optimize_new_OP_interfaces[i+1] = OP_optimize_new_OP_interfaces[i] + OP_step[interface_mode]
		#OP_optimize_new_OP_interfaces = np.unique(OP_optimize_new_OP_interfaces)
	N_new_interfaces = len(OP_optimize_new_OP_interfaces)
	#print('N_M_i =', N_new_interfaces, '\nM_new:\n', OP_optimize_new_OP_interfaces, '\nM_new_original:\n', OP_optimize_new_OP_interfaces_original)
	print('OP_new:\n', OP_optimize_new_OP_interfaces)
	
	if(OP_hist_edges is not None):
		OP_hist_lens = (OP_hist_edges[1:] - OP_hist_edges[:-1])
		OP_hist_centers = (OP_hist_edges[1:] + OP_hist_edges[:-1]) / 2
		N_OP_hist = len(OP_hist_centers)
	
	
	if(to_plot_hists):
		if(states is not None):
			OP_init_states = []
			OP_init_hist = np.empty((N_OP_hist, N_OP_interfaces))
			d_OP_init_hist = np.empty((N_OP_hist, N_OP_interfaces))
			for i in range(N_OP_interfaces):
				OP_init_states.append(np.empty(N_init_states[i], dtype=int))
				for j in range(N_init_states[i]):
					if(interface_mode == 'M'):
						OP_init_states[i][j] = np.sum(states[j, :, :], ) / L2
					elif(interface_mode == 'CS'):
						(cluster_element_inds, cluster_sizes) = izing.cluster_state(states[i][j, :, :].reshape((L2)), default_state=def_spin_state)
						OP_init_states[i][j] = max(cluster_sizes)
				
				assert(OP_hist_edges is not None), 'ERROR: no "OP_hist_edges" provided but "OP_init_states" is asked to be analyzed (_states != None)'
				OP_init_hist[:, i], _ = np.histogram(OP_init_states[i], bins=OP_hist_edges)
				OP_init_hist[:, i] = OP_init_hist[:, i] / N_init_states[i]
				d_OP_init_hist[:, i] = np.sqrt(OP_init_hist[:, i] * (1 - OP_init_hist[:, i]) / N_init_states[i])
				
			fig_OPinit_hists, ax_OPinit_hists = my.get_fig(y_lbl, r'$p_i(' + x_lbl + ')$', title=r'$p(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			for i in range(N_OP_interfaces):
				ax_OPinit_hists.bar(OP_hist_centers, OP_init_hist[:, i], yerr=d_OP_init_hist[:, i], width=OP_hist_lens, align='center', label='OP = ' + str(OP_interfaces[i]), alpha=Ms_alpha)
			ax_OPinit_hists.legend()
		
	rho = None
	d_rho = None
	OP_hist_centers = None
	OP_hist_lens = None
	
	ax_PB_log = None
	ax_PB = None
	ax_PB_sgm = None
	ax_fl = None
	ax_fl_i = None
	ax_OP = None
	ax_F = None
	ax_OPhists = None
	if(OP_data is not None):
		assert(OP_interfaces[0] < OP_match_BF_to), 'ERROR: OP_interfaces start at %d, but the state is match to BF only through %d' % (OP_interfaces[0], OP_match_BF_to)
		#assert(OP_match_BF_to < OP_sample_BF_to), 'ERROR: OP_match_to = %d >= OP_sample_to = %d' % (OP_match_BF_to, OP_sample_BF_to)
		
		Nt_total = len(OP_data)
		Nt_totals = np.empty(N_OP_interfaces, dtype=np.intc)
		Nt_totals[0] = Nt[0]
		OP = [OP_data[ : Nt_totals[0]]]
		
		def get_jumps(OP, OP_interfaces, k):
			OP_next = OP_interfaces[k + 1]
			OP_init = OP_interfaces[k]
			OP_A = OP_interfaces[0]
			#OP[(OP <= OP_A) | (OP >= OP_next)] = OP_init
			#OP = np.concatenate([OP_init], OP)
			traj_end_inds = np.sort(np.concatenate(([0], \
											np.where((OP <= OP_A) | (OP >= OP_next))[0], \
											np.where((OP[1:] > OP[:-1] * 2 + 1) | (OP[1:] < OP[:-1] // 2))[0] + 1)))
			N_trajs = len(traj_end_inds) - 1
			jumps = []
			for i in range(N_trajs):
				OP_traj = np.concatenate(([OP_init], OP[traj_end_inds[i]:traj_end_inds[i + 1]]))
				traj_jumps = OP_traj[1:] - OP_traj[:-1]
				jumps.append(traj_jumps)
			return np.concatenate(tuple(jumps))
		
		OP_jumps = [OP[0][1:] - OP[0][:-1]]   # 1st run is different from others - it's BF and with no stops (it's a single trajectory)
		for i in range(1, N_OP_interfaces):
			Nt_totals[i] = Nt_totals[i-1] + Nt[i]
			OP.append(OP_data[Nt_totals[i-1] : Nt_totals[i]])
			OP_jumps.append(get_jumps(OP[i], OP_interfaces, i - 1))
		del OP_data
		
		# The number of bins cannot be arbitrary because OP is descrete, thus it's a good idea if all the bins have 1 descrete value inside or all the bins have 2 or etc. 
		# It's not good if some bins have e.g. 1 possible OP value, and others have 2 possible values. 
		# There are 'L^2 + 1' possible values of OP, because '{OP \in [-L^2; L^2]} and {dM = 2}'
		# Thus, we need edges which cover [-1-1/L^2; 1+1/L^2] with step=2/L^2
		OP_jumps_hist_edges = np.concatenate(([OP_possible_jumps[interface_mode][0] - OP_step[interface_mode] / 2], \
											(OP_possible_jumps[interface_mode][1:] + OP_possible_jumps[interface_mode][:-1]) / 2, \
											[OP_possible_jumps[interface_mode][-1] + OP_step[interface_mode] / 2]))
		OP_jumps_hist_lens = (OP_jumps_hist_edges[1:] - OP_jumps_hist_edges[:-1])

		small_rho = 1e-2 / Nt_totals[-1] / N_OP_hist
		rho_s = np.zeros((N_OP_hist, N_OP_interfaces))
		d_rho_s = np.zeros((N_OP_hist, N_OP_interfaces))
		OP_hist_ok_inds = np.zeros((N_OP_hist, N_OP_interfaces), dtype=bool)
		rho_for_sum = np.empty((N_OP_hist, N_OP_interfaces - 1))
		d_rho_for_sum = np.zeros((N_OP_hist, N_OP_interfaces - 1))
		rho_w = np.empty(N_OP_interfaces - 1)
		d_rho_w = np.empty(N_OP_interfaces - 1)
		OP_jumps_hists = np.empty((len(OP_jumps_hist_edges) - 1, N_OP_interfaces))
		timesteps = [np.arange(Nt[0])]
		for i in range(N_OP_interfaces):
			#timesteps.append(np.arange(Nt[i]) + (0 if(i == 0) else Nt_totals[i-1]))
			print(OP[i].shape)
			OP_hist, _ = np.histogram(OP[i], bins=OP_hist_edges)
			OP_jumps_hists[:, i], _ = np.histogram(OP_jumps[i], bins=OP_jumps_hist_edges)
			OP_jumps_hists[:, i] = OP_jumps_hists[:, i] / Nt[i]
			OP_hist = OP_hist / memory_time
			OP_hist_ok_inds[:, i] = (OP_hist > 0)
			#rho_s[OP_hist_ok_inds, i] = OP_hist[OP_hist_ok_inds] / OP_hist_lens[OP_hist_ok_inds] / Nt[i]   # \pi(q, l_i)
			#d_rho_s[OP_hist_ok_inds, i] = np.sqrt(OP_hist[OP_hist_ok_inds] * (1 - OP_hist[OP_hist_ok_inds] / Nt[i])) / OP_hist_lens[OP_hist_ok_inds] / Nt[i]
			rho_s[OP_hist_ok_inds[:, i], i] = OP_hist[OP_hist_ok_inds[:, i]] / OP_hist_lens[OP_hist_ok_inds[:, i]] / N_init_states[i]   # \pi(q, l_i)
			d_rho_s[OP_hist_ok_inds[:, i], i] = np.sqrt(OP_hist[OP_hist_ok_inds[:, i]] * (1 - OP_hist[OP_hist_ok_inds[:, i]] / Nt[i])) / OP_hist_lens[OP_hist_ok_inds[:, i]] / N_init_states[i]
			
			if(i > 0):
				timesteps.append(np.arange(Nt[i]) + Nt_totals[i-1])
				#rho_w[i-1] = (1 if(i == 0) else P_B[0] / P_B[i - 1])
				rho_w[i-1] = P_B[0] / P_B[i - 1]
				#d_rho_w[i] = (0 if(i == 0) else (rho_w[i] * d_P_B[i - 1] / P_B[i - 1]))
				d_rho_w[i-1] = rho_w[i-1] * d_P_B[i - 1] / P_B[i - 1]
				rho_for_sum[:, i-1] = rho_s[:, i] * rho_w[i-1]
				#d_rho_for_sum[OP_hist_ok_inds, i] = rho_for_sum[OP_hist_ok_inds, i] * np.sqrt((d_rho_s[OP_hist_ok_inds, i] / rho_s[OP_hist_ok_inds, i])**2 + (0 if(i == 0) else np.sum((d_probs[:i] / probs[:i])**2)))
				d_rho_for_sum[OP_hist_ok_inds[:, i], i-1] = rho_for_sum[OP_hist_ok_inds[:, i], i-1] * np.sqrt((d_rho_s[OP_hist_ok_inds[:, i], i] / rho_s[OP_hist_ok_inds[:, i], i])**2 + (d_rho_w[i-1] / rho_w[i-1])**2)
		
		rho = np.sum(rho_for_sum, axis=1)
		d_rho = np.sqrt(np.sum(d_rho_for_sum**2, axis=1))
		OP_hist_ok_inds_FFS = (rho > 0)
		F = np.zeros(N_OP_hist)
		d_F = np.zeros(N_OP_hist)
		F[OP_hist_ok_inds_FFS] = -Temp * np.log(rho[OP_hist_ok_inds_FFS] * OP_hist_lens[OP_hist_ok_inds_FFS])
		d_F[OP_hist_ok_inds_FFS] = Temp * d_rho[OP_hist_ok_inds_FFS] / rho[OP_hist_ok_inds_FFS]
		
		BF_inds = OP_hist_centers <= OP_match_BF_to / OP_scale[interface_mode]
		F_BF = -Temp * np.log(rho_s[BF_inds, 0] * OP_hist_lens[BF_inds])
		d_F_BF = Temp * d_rho_s[BF_inds, 0] / rho_s[BF_inds, 0]
		
		FFS_match_points_inds = (OP_hist_centers > OP_interfaces[0] / OP_scale[interface_mode]) & BF_inds
		BF_match_points = (OP_hist_centers[BF_inds] > OP_interfaces[0] / OP_scale[interface_mode])
		#print(d_F.shape, FFS_match_points_inds.shape, d_F_BF.shape, BF_match_points.shape)
		#input('a')
		w2 = 1 / (d_F[FFS_match_points_inds]**2 + d_F_BF[BF_match_points]**2)
		dF_FFS_BF = F[FFS_match_points_inds] - F_BF[BF_match_points]
		#F_shift = np.sum(dF_FFS_BF * w2) / np.sum(w2)
		F_shift = np.average(dF_FFS_BF, weights=w2)
		F_BF = F_BF + F_shift
		
		BF_OP_border = OP_interfaces[0] + OP_step[interface_mode] * 3
		N_points_to_match_F_to = (OP_match_BF_to - BF_OP_border) / OP_step[interface_mode]
		if(N_points_to_match_F_to < 5):
			print('WARNING: Only few (%s) points to match F_BF to' % (my.f2s(N_points_to_match_F_to)))
		
		state_A_FFS_inds = (OP_hist_centers <= BF_OP_border / OP_scale[interface_mode]) & BF_inds
		state_A_BF_inds = OP_hist_centers[BF_inds] <= BF_OP_border / OP_scale[interface_mode]
		rho[state_A_FFS_inds] = np.exp(-F_BF[state_A_BF_inds] / Temp) / OP_hist_lens[state_A_FFS_inds]
		d_rho[state_A_FFS_inds] = rho[state_A_FFS_inds] * d_F_BF[state_A_BF_inds] / Temp
		
		F = -Temp * np.log(rho * OP_hist_lens)
		#F_BF = F_BF - min(F)
		#F = F - min(F)
		F_BF = F_BF - F[0]
		F = F - F[0]
		d_F = Temp * d_rho / rho
		
		if(to_plot_time_evol):
			fig_OP, ax_OP = my.get_fig('step', y_lbl, title=x_lbl + '(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			for i in range(N_OP_interfaces):
				ax_OP.plot(timesteps[i][::stride], OP[i][::stride], label='data, i=%d' % (i))
				
			ax_OP.legend()
		
		if(to_plot_hists):
			fig_OPhists, ax_OPhists = my.get_fig(y_lbl, r'$\rho_i(' + x_lbl + ') \cdot P(i | A)$', title=r'$\rho(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			for i in range(N_OP_interfaces - 1):
				ax_OPhists.bar(OP_hist_centers, rho_for_sum[:, i], yerr=d_rho_for_sum[:, i], width=OP_hist_lens, align='center', label=str(i), alpha=Ms_alpha)
			
			for i in range(N_OP_interfaces):
				ax_OPhists.plot([OP_interfaces_scaled[i]] * 2, [min(rho), max(rho)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_OPhists.legend()
			
			fig_OPjumps_hists, ax_OPjumps_hists = my.get_fig('d' + x_lbl, 'probability', title=x_lbl + ' jumps distribution; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			for i in range(N_OP_interfaces):
				ax_OPjumps_hists.plot(OP_possible_jumps[interface_mode], OP_jumps_hists[:, i], label='$' + x_lbl + ' \in (' + str((OP_min_default[interface_mode]) if(i == 0) else OP_interfaces_scaled[i-1]) + ';' + str(OP_interfaces_scaled[i]) + ']$')
			ax_OPjumps_hists.legend()
			
			fig_F, ax_F = my.get_fig(y_lbl, r'$F(' + x_lbl + r') = -T \ln(\rho(%s) \cdot d%s(%s))$' % (x_lbl, x_lbl, x_lbl), title=r'$F(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			ax_F.errorbar(OP_hist_centers[BF_inds], F_BF, yerr=d_F_BF, fmt='.', label='$F_{BF}$')
			ax_F.errorbar(OP_hist_centers, F, yerr=d_F, fmt='.', label='$F_{total}$')
			for i in range(N_OP_interfaces):
				ax_F.plot([OP_interfaces_scaled[i]] * 2, [min(F), max(F)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_F.legend()
			
			fig_PB_log, ax_PB_log = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
			fig_PB, ax_PB = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			fig_PB_sgm, ax_PB_sgm = my.get_fig(y_lbl, r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sgm, OP_interfaces_scaled, P_B[:-1], P_B_sigmoid, interface_mode)
			plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sgm, h, 'P_B', x_lbl, OP_interfaces_scaled[:-1], P_B[:-1][linfit_inds], d_PB=d_P_B[:-1][linfit_inds], \
						PB_sgm=P_B_sigmoid, d_PB_sgm=d_P_B_sigmoid, linfit=linfit, linfit_inds=linfit_inds, OP0=OP0, \
						clr=my.get_my_color(1))
			ax_PB_log.legend()
			ax_PB.legend()
			ax_PB_sgm.legend()
			
			fig_fl, ax_fl = my.get_fig(y_lbl, r'$1 - \ln[P_B(' + x_lbl + ')] / \ln[P_B(0)] = f(\lambda_i)$', title=r'$f(\lambda_i)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			fig_fl_i, ax_fl_i = my.get_fig(r'$index(' + x_lbl + '_i)$', r'$1 - \ln[P_B(' + x_lbl + '_i)] / \ln[P_B(0)] = f(\lambda_i)$', title=r'$f(\lambda_i)$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
			
			m_fine = np.linspace(min(OP_interfaces_scaled), max(OP_interfaces_scaled), int((max(OP_interfaces_scaled) - min(OP_interfaces_scaled)) / min(OP_interfaces_scaled[1:] - OP_interfaces_scaled[:-1]) * 10))
			ax_fl.errorbar(OP_interfaces_scaled, OP_optimize_f_interp_vals, yerr=d_OP_optimize_f_interp_vals, fmt='.', label=r'$f(\lambda)_i$')
			ax_fl_i.errorbar(np.arange(N_OP_interfaces), OP_optimize_f_interp_vals, yerr=d_OP_optimize_f_interp_vals, fmt='.', label=r'$f(\lambda)_i$')
			#ax_fl.plot(m_fine[:-1], M_optimize_f_interp_fit(m_fine[:-1]), label=r'$f_{fit}(\lambda)$')
			ax_fl.plot(m_fine, OP_optimize_f_interp_interp1d(m_fine), label=r'$f_{interp}(\lambda)$')
			for i in range(N_new_interfaces):
				ax_fl.plot([OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode]] * 2, [0, OP_optimize_f_interp_interp1d(OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode])], '--', label=('new interfaces' if(i == 0) else None), color=my.get_my_color(3))
				ax_fl.plot([min(OP_interfaces_scaled), OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode]], [OP_optimize_f_interp_interp1d(OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode])] * 2, '--', label=None, color=my.get_my_color(3))
			ax_fl.plot([min(OP_interfaces_scaled), max(OP_interfaces_scaled)], [0, 1], '--', label=r'$interpolation$')
			ax_fl_i.plot([0, N_OP_interfaces - 1], [0, 1], '--', label=r'$interpolation$')
			
			ax_fl.legend()
			ax_fl_i.legend()

	return ln_k_AB, d_ln_k_AB, P_B, d_P_B, P_B_sigmoid, d_P_B_sigmoid, linfit, linfit_inds, OP0, \
			rho, d_rho, OP_hist_centers, OP_hist_lens, \
			ax_OP, ax_OPhists, ax_F, ax_PB_log, ax_PB, ax_PB_sgm, ax_fl, ax_fl_i

def proc_FFS_AB(L, Temp, h, N_init_states, OP_interfaces, interface_mode, def_spin_state, \
				verbose=None, to_get_timeevol=False, \
				OP_sample_BF_to=None, OP_match_BF_to=None, 
				to_plot_time_evol=False, to_plot_hists=False, timeevol_stride=-3000, \
				init_gen_mode=-2, Ms_alpha=0.5):
	print('=============== running h = %s ==================' % (my.f2s(h)))
	
	# py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces, int to_get_timeevol, std::optional<int> _verbose)
	# return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M, biggest_cluster_sizes);
	(_states, probs, d_probs, Nt, flux0, d_flux0, _E, _M, _CS) = \
		izing.run_FFS(L, Temp, h, N_init_states, OP_interfaces, verbose=verbose, \
					to_remember_timeevol=to_get_timeevol, init_gen_mode=init_gen_mode, \
					interface_mode=OP_C_id[interface_mode], default_spin_state=def_spin_state)
	
	# print('timeevol', to_get_timeevol)
	# print('states:', _states.shape)
	# print('p:', probs.shape, probs)
	# print('d_p:', d_probs.shape, d_probs)
	# print('Nt', Nt.shape, Nt)
	# print('f:', flux0)
	# print('d_f:', d_flux0)
	# print('_E:', _E.shape)
	# print('_M:', _M.shape)
	# print('_CS:', _CS.shape)
	# input('goon')
	
	L2 = L**2
	N_OP_interfaces = len(OP_interfaces)
	if(to_get_timeevol):
		Nt_total = max(len(_M), len(_CS))
		if(timeevol_stride < 0):
			timeevol_stride = np.int_(- Nt_total / timeevol_stride)
	#print('N_init_states:', N_init_states)
	#input('goon')
	states = [_states[ : N_init_states[0] * L2].reshape((N_init_states[0], L, L))]
	for i in range(1, N_OP_interfaces):
		states.append(_states[np.sum(N_init_states[:i]) * L2 : np.sum(N_init_states[:(i+1)] * L2)].reshape((N_init_states[i], L, L)))
	del _states
	
	data = {}
	data['M'] = _M
	data['CS'] = _CS
	
	#print('f:', flux0)
	#print('p:', probs)
	#input('goon')
	
	ln_k_AB, d_ln_k_AB, P_B, d_P_B, P_B_sigmoid, d_P_B_sigmoid, linfit, linfit_inds, OP0, \
		rho, d_rho, OP_hist_centers, OP_hist_lens, \
		ax_OP, ax_OPhists, ax_F, ax_PB_log, ax_PB, ax_PB_sgm, ax_fl, ax_fl_i = \
			proc_order_parameter_FFS(L, Temp, h, flux0, d_flux0, probs, d_probs, OP_interfaces, N_init_states, \
						interface_mode, def_spin_state, OP_match_BF_to=OP_match_BF_to, \
						OP_data=(data[interface_mode] / OP_scale[interface_mode] if(to_get_timeevol) else None), \
						Nt=Nt, OP_hist_edges=get_OP_hist_edges(interface_mode, OP_max=OP_interfaces[-1]), \
						memory_time=max(1, OP_std_overestimate[interface_mode] / OP_step[interface_mode]), 
						to_plot_time_evol=to_plot_time_evol, to_plot_hists=to_plot_hists, stride=timeevol_stride, \
						x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], Ms_alpha=Ms_alpha, states=states)
	
	return probs, d_probs, ln_k_AB, d_ln_k_AB, flux0, d_flux0, rho, d_rho, OP_hist_centers, OP_hist_lens, \
			P_B, d_P_B, P_B_sigmoid, d_P_B_sigmoid, linfit, linfit_inds, OP0

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

def proc_order_parameter_BF(L, Temp, h, m, E, stab_step, dOP_step, OP_hist_edges, OP_OP_peak_guess, OP_OP_fit2_width, \
						 x_lbl=None, y_lbl=None, verbose=None, to_estimate_k=False, hA=None, OP_A=None, OP_B=None, \
						 to_plot_time_evol=True, to_plot_F=True, to_plot_ETS=False, stride=1, OP_jumps_hist_edges=None, \
						 possible_jumps=None):
	Nt = len(m)
	steps = np.arange(Nt)
	L2 = L**2
	stab_ind = (steps > stab_step)
	OP_stab = m[stab_ind]
	
	Nt_stab = len(OP_stab)
	assert(Nt_stab > 1), 'ERROR: the system may not have reached equlibrium, stab_step = %d, max-present-step = %d' % (stab_step, max(steps))
	
	OP_jumps = OP_stab[1:] - OP_stab[:-1]
	OP_mean = np.mean(OP_stab)
	OP_std = np.std(OP_stab)
	memory_time = max(1, OP_std / dOP_step)   
	d_OP_mean = OP_std / np.sqrt(Nt_stab / memory_time)
	# Time of statistical decorelation of the system state.
	# 'the number of flips necessary to cover the typical system states range' * 'the upper-bound estimate on the number of steps for 1 flip to happen'
	print('memory time =', my.f2s(memory_time))
	
	OP_hist, _ = np.histogram(OP_stab, bins=OP_hist_edges)
	OP_jumps_hist, _ = np.histogram(OP_jumps, bins=OP_jumps_hist_edges)
	OP_jumps_hist = OP_jumps_hist / Nt_stab
	OP_hist_lens = (OP_hist_edges[1:] - OP_hist_edges[:-1])
	N_m_points = len(OP_hist_lens)
	OP_hist[OP_hist == 0] = 1   # to not get errors for log(hist)
	OP_hist = OP_hist / memory_time
	OP_hist_centers = (OP_hist_edges[1:] + OP_hist_edges[:-1]) / 2
	rho = OP_hist / OP_hist_lens / Nt_stab
	d_rho = np.sqrt(OP_hist * (1 - OP_hist / Nt_stab)) / OP_hist_lens / Nt_stab

	rho_interp1d = scipy.interpolate.interp1d(OP_hist_centers, rho, fill_value='extrapolate')
	d_rho_interp1d = scipy.interpolate.interp1d(OP_hist_centers, d_rho, fill_value='extrapolate')

	F = -Temp * np.log(rho * OP_hist_lens)
	F = F - F[0]
	d_F = Temp * d_rho / rho
	
	E_avg = np.empty(N_m_points)
	for i in range(N_m_points):
		E_for_avg = E[(OP_hist_edges[i] < m) & (m < OP_hist_edges[i+1])] * L2
		#E_avg[i] = np.average(E_for_avg, weights=np.exp(- E_for_avg / Temp))
		E_avg[i] = np.mean(E_for_avg) if(len(E_for_avg) > 0) else (2 + abs(h)) * L2
	E_avg = E_avg - E_avg[0]   # we can choose the E constant
	S = (E_avg - F) / Temp
	S = S - S[0] # S[m=+-1] = 0 because S = log(N), and N(m=+-1)=1
	
	if(to_estimate_k):
		assert(hA is not None), 'ERROR: k estimation is requested in BF, but no hA provided'
		#k_AB_ratio = np.sum(hA > 0.5) / np.sum(hA < 0.5)
		hA_jump = hA[1:] - hA[:-1]   # jump > 0 => {0->1} => from B to A => k_BA ==> k_ratio = k_BA / k_AB
		jump_inds = np.where(hA_jump != 0)[0]

		N_AB_jumps = np.sum(hA_jump < 0)
		k_AB = N_AB_jumps / np.sum(hA > 0.5)
		d_k_AB = k_AB / np.sqrt(N_AB_jumps)

		N_BA_jumps = np.sum(hA_jump > 0)
		k_BA = N_BA_jumps / np.sum(hA < 0.5)
		d_k_BA = k_BA / np.sqrt(N_BA_jumps)
		
		OP_fit2_OPmin_ind = np.argmax(OP_hist_centers > OP_OP_peak_guess - OP_OP_fit2_width)
		OP_fit2_OPmax_ind = np.argmax(OP_hist_centers > OP_OP_peak_guess + OP_OP_fit2_width)
		assert(OP_fit2_OPmin_ind > 0), 'issues finding analytical border'
		assert(OP_fit2_OPmax_ind > 0), 'issues finding analytical border'
		OP_fit2_inds = (OP_fit2_OPmin_ind <= np.arange(len(OP_hist_centers))) & (OP_fit2_OPmax_ind >= np.arange(len(OP_hist_centers)))
		OP_fit2 = np.polyfit(OP_hist_centers[OP_fit2_inds], F[OP_fit2_inds], 2, w = 1/d_F[OP_fit2_inds])
		OP_peak = -OP_fit2[1] / (2 * OP_fit2[0])
		m_peak_ind = np.argmin(abs(OP_hist_centers - OP_peak))
		F_max = np.polyval(OP_fit2, OP_peak)
		#m_peak_ind = np.argmin(OP_hist_centers - OP_peak)
		bc_Z_AB = exp_integrate(OP_hist_centers[ : OP_fit2_OPmin_ind + 1], -F[ : OP_fit2_OPmin_ind + 1] / Temp) + exp2_integrate(- OP_fit2 / Temp, OP_hist_centers[OP_fit2_OPmin_ind])
		bc_Z_BA = exp_integrate(OP_hist_centers[OP_fit2_OPmax_ind : ], -F[OP_fit2_OPmax_ind : ] / Temp) + abs(exp2_integrate(- OP_fit2 / Temp, OP_hist_centers[OP_fit2_OPmax_ind]))
		k_bc_AB = (np.mean(abs(OP_hist_centers[m_peak_ind + 1] - OP_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max / Temp) / bc_Z_AB)
		k_bc_BA = (np.mean(abs(OP_hist_centers[m_peak_ind + 1] - OP_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max / Temp) / bc_Z_BA)
		
		get_log_errors(np.log(k_AB), d_k_AB / k_AB, lbl='k_AB_BF', print_scale=print_scale_k)
		get_log_errors(np.log(k_BA), d_k_BA / k_BA, lbl='k_BA_BF', print_scale=print_scale_k)
	else:
		k_AB = None
		d_k_AB = None
		k_BA = None
		d_k_BA = None
		k_bc_AB = None
		k_bc_BA = None
	
	if(to_plot_time_evol):
		fig_OP, ax_OP = my.get_fig('step', y_lbl, title='$' + x_lbl + '$(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_OP.plot(steps[::stride], m[::stride], label='data')
		ax_OP.plot([stab_step] * 2, [min(m), max(m)], '--', label='equilibr')
		ax_OP.plot([stab_step, Nt], [OP_mean] * 2, '--', label=('$<' + x_lbl + '> = ' + my.errorbar_str(OP_mean, d_OP_mean) + '$'))
		if(to_estimate_k):
			ax_OP.plot([0, Nt], [OP_A] * 2, '--', label='state A, B', color=my.get_my_color(1))
			ax_OP.plot([0, Nt], [OP_B] * 2, '--', label=None, color=my.get_my_color(1))
		ax_OP.legend()
		
		fig_hA, ax_hA = my.get_fig('step', '$h_A$', title='$h_{A, ' + x_lbl + '}$(step); T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_hA.plot(steps[::stride], hA[::stride], label='$h_A$')
		ax_hA.plot(steps[jump_inds+1], hA_jump[jump_inds], '.', label='$h_A$ jumps')
		ax_hA.legend()

		#print('k_' + x_lbl + '_AB    =    (' + my.errorbar_str(k_AB, d_k_AB) + ')    (1/step);    dk/k =', my.f2s(d_k_AB / k_AB))
		#print('k_' + x_lbl + '_BA    =    (' + my.errorbar_str(k_BA, d_k_BA) + ')    (1/step);    dk/k =', my.f2s(d_k_BA / k_BA))
	else:
		ax_OP = None
	
	if(to_plot_F):
		fig_OP_hist, ax_OP_hist = my.get_fig(y_lbl, r'$\rho(' + x_lbl + ')$', title=r'$\rho(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_OP_hist.bar(OP_hist_centers, rho, yerr=d_rho, width=OP_hist_lens, label=r'$\rho(' + x_lbl + ')$', align='center')
		if(to_estimate_k):
			ax_OP_hist.plot([OP_A] * 2, [0, max(rho)], '--', label='state A,B', color=my.get_my_color(1))
			ax_OP_hist.plot([OP_B] * 2, [0, max(rho)], '--', label=None, color=my.get_my_color(1))
		ax_OP_hist.legend()
		
		fig_OP_jumps_hist, ax_OP_jumps_hist = my.get_fig('d' + x_lbl, 'probability', title=x_lbl + ' jumps distribution; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
		ax_OP_jumps_hist.plot(possible_jumps, OP_jumps_hist, '.')
		
		fig_F, ax_F = my.get_fig(y_lbl, r'$F/J$', title=r'$F(' + x_lbl + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		ax_F.errorbar(OP_hist_centers, F, yerr=d_F, fmt='.', label='$F(' + x_lbl + ')$')
		if(to_estimate_k):
			ax_F.plot([OP_A] * 2, [min(F), max(F)], '--', label='state A, B', color=my.get_my_color(1))
			ax_F.plot([OP_B] * 2, [min(F), max(F)], '--', label=None, color=my.get_my_color(1))
			ax_F.plot(OP_hist_centers[OP_fit2_inds], np.polyval(OP_fit2, OP_hist_centers[OP_fit2_inds]), label='fit2; $' + x_lbl + '_0 = ' + my.f2s(OP_peak) + '$')
		
		if(to_plot_ETS):
			ax_F.plot(OP_hist_centers, E_avg, '.', label='<E>(' + x_lbl + ')')
			ax_F.plot(OP_hist_centers, S * Temp, '.', label='$T \cdot S(' + x_lbl + ')$')
		ax_F.legend()
		
		if(to_estimate_k):
			print('k_' + x_lbl + '_bc_AB   =   ' + my.f2s(k_bc_AB * print_scale_k) + '   (%e/step);' % (1/print_scale_k))
			print('k_' + x_lbl + '_bc_BA   =   ' + my.f2s(k_bc_BA * print_scale_k) + '   (%e/step);' % (1/print_scale_k))
	else:
		ax_OP_hist = None
		ax_F = None
	
	return F, d_F, OP_hist_centers, OP_hist_lens, rho_interp1d, d_rho_interp1d, k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_mean, OP_std, d_OP_mean, E_avg, S, ax_OP, ax_OP_hist, ax_F

def plot_correlation(x, y, x_lbl, y_lbl):
	fig, ax = my.get_fig(x_lbl, y_lbl)
	R = scipy.stats.pearsonr(x, y)[0]
	ax.plot(x, y, '.', label='R = ' + my.f2s(R))
	ax.legend()
	
	return ax, R

def proc_T(L, Temp, h, Nt, interface_mode, def_spin_state, verbose=None, N_spins_up_init=None, to_get_timeevol=True, \
			to_plot_time_evol=False, to_plot_F=False, to_plot_ETS=False, to_plot_correlations=False, \
			to_estimate_k=False, timeevol_stride=-3000, OP_min=None, OP_max=None, OP_A=None, OP_B=None):
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
	
	(E, M, CS, hA, k_AB_BFcount) = izing.run_bruteforce(L, Temp, h, Nt, N_spins_up_init=N_spins_up_init, OP_A=OP_A, OP_B=OP_B, \
										OP_min=OP_min, OP_max=OP_max, to_remember_timeevol=to_get_timeevol, \
										interface_mode=OP_C_id[interface_mode], default_spin_state=def_spin_state, \
										verbose=verbose)
	
	if(to_get_timeevol):
		if(timeevol_stride < 0):
			timeevol_stride = np.int_(- Nt / timeevol_stride)

		stab_step = int(min(L2, Nt / 2)) * 5
		# Each spin has a good chance to flip during this time
		# This does not guarantee the global stable state, but it is sufficient to be sure-enough we are in the optimum of the local metastable-state.
		# The approximation for a global ebulibration would be '1/k_AB', where k_AB is the rate constant. But it's hard to estimate on that stage.
		data = {}
		hist_edges = {}
		OP_jumps_hist_edges = np.concatenate(([OP_possible_jumps[interface_mode][0] - OP_step[interface_mode] / 2], \
										(OP_possible_jumps[interface_mode][1:] + OP_possible_jumps[interface_mode][:-1]) / 2, \
										[OP_possible_jumps[interface_mode][-1] + OP_step[interface_mode] / 2]))	
		
		data['M'] = M
		hist_edges['M'] =  get_OP_hist_edges(interface_mode, OP_min if(interface_mode == 'M') else None, OP_max if(interface_mode == 'M') else None)
		del M

		data['CS'] = CS
		hist_edges['CS'] = get_OP_hist_edges(interface_mode, OP_min if(interface_mode == 'CS') else None, OP_max if(interface_mode == 'CS') else None)
		
		N_features = len(hist_edges.keys())
		
		F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, \
			k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_avg, \
				OP_std, d_OP_avg, E_avg, S_E, \
				ax_time, ax_hist, ax_F = \
					proc_order_parameter_BF(L, Temp, h, data[interface_mode] / OP_scale[interface_mode], E / L2, stab_step, \
									dOP_step[interface_mode], hist_edges[interface_mode], OP_peak_guess[interface_mode], OP_fit2_width[interface_mode], \
									x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], verbose=verbose, to_estimate_k=to_estimate_k, hA=hA, \
									to_plot_time_evol=to_plot_time_evol, to_plot_F=to_plot_F, to_plot_ETS=to_plot_ETS, stride=timeevol_stride, \
									OP_A=OP_A / OP_scale[interface_mode], OP_B=OP_B / OP_scale[interface_mode], OP_jumps_hist_edges=OP_jumps_hist_edges, \
									possible_jumps=OP_possible_jumps[interface_mode])

		BC_cluster_size = L2 / np.pi
		#C = ((OP_std['E'] * L2)**2 / Temp**2) / L2   # heat capacity per spin

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
					# plot_correlation(data[k1][::timeevol_stride] / OP_scale[k1], data[k2][::timeevol_stride] / OP_scale[k2], \
									# feature_label[k1], feature_label[k2])
		
		return F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_avg, d_OP_avg, E_avg#, C
	else:
		if(k_AB_BFcount > 0):
			d_k_AB_BFcount = k_AB_BFcount / np.sqrt(k_AB_BFcount * Nt)
			print('k_AB_BFcount = (' + my.errorbar_str(k_AB_BFcount, d_k_AB_BFcount) + ') 1/step')
		else:
			d_k_AB_BFcount = 0
			print('k_AB_BFcount = 0')
		
		
		return k_AB_BFcount, d_k_AB_BFcount
	
	

def run_many(L, Temp, h, N_runs, interface_mode, def_spin_state, \
			OP_A=None, OP_B=None, N_spins_up_init=None, \
			OP_interfaces_AB=None, OP_interfaces_BA=None, \
			OP_sample_BF_A_to=None, OP_sample_BF_B_to=None, \
			OP_match_BF_A_to=None, OP_match_BF_B_to=None, \
			Nt_per_BF_run=None, verbose=None, to_plot_time_evol=False, \
			timeevol_stride=-3000, to_plot_k_distr=False, N_k_bins=10, \
			mode='BF', N_init_states_AB=None, N_init_states_BA=None, \
			init_gen_mode=-2, to_plot_committer=None, to_get_timeevol=None):
	L2 = L**2
	old_seed = izing.get_seed()
	if(to_get_timeevol is None):
		to_get_timeevol = ('BF' == mode)
	
	if('BF' in mode):
		assert(Nt_per_BF_run is not None), 'ERROR: BF mode but no Nt_per_run provided'
		if(OP_A is None):
			assert(len(OP_interfaces_AB) > 0), 'ERROR: nor OP_A neither OP_interfaces_AB were provided for run_many(BF)'
			OP_A = OP_interfaces_AB[0]
		if('AB' not in mode):
			if(OP_B is None):
				assert(len(OP_interfaces_BA) > 0), 'ERROR: nor OP_B neither OP_interfaces_BA were provided for run_many(BF)'
				OP_B = flip_OP(OP_interfaces_BA[0], L2, interface_mode)
		
	elif('FFS' in mode):
		assert(N_init_states_AB is not None), 'ERROR: FFS_AB mode but no "N_init_states_AB" provided'
		assert(OP_interfaces_AB is not None), 'ERROR: FFS_AB mode but no "OP_interfaces_AB" provided'
		
		N_OP_interfaces_AB = len(OP_interfaces_AB)
		OP_AB = OP_interfaces_AB / OP_scale[interface_mode]
		if(mode == 'FFS'):
			assert(N_init_states_BA is not None), 'ERROR: FFS mode but no "N_init_states_BA" provided'
			assert(OP_interfaces_BA is not None), 'ERROR: FFS mode but no "OP_interfaces_BA" provided'
			
			N_OP_interfaces_BA = len(OP_interfaces_BA)
			OP_BA = flip_OP(OP_interfaces_BA, L2, interface_mode) / OP_scale[interface_mode]
			OP = np.sort(np.unique(np.concatenate((OP_AB, OP_BA))))
		elif(mode == 'FFS_AB'):
			OP = np.copy(OP_AB)
	
	ln_k_AB_data = np.empty(N_runs)
	ln_k_BA_data = np.empty(N_runs) 
	if(to_get_timeevol):
		rho_fncs = [[]] * N_runs
		d_rho_fncs = [[]] * N_runs
	if(mode == 'BF'):
		ln_k_bc_AB_data = np.empty(N_runs)
		ln_k_bc_BA_data = np.empty(N_runs) 
	elif('FFS' in mode):
		flux0_AB_data = np.empty(N_runs) 
		OP0_AB_data = np.empty(N_runs) 
		probs_AB_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		PB_AB_data = np.empty((N_OP_interfaces_AB, N_runs))
		d_PB_AB_data = np.empty((N_OP_interfaces_AB, N_runs))
		PB_sigmoid_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		d_PB_sigmoid_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		PB_linfit_data = np.empty((2, N_runs))
		PB_linfit_inds_data = np.empty((N_OP_interfaces_AB - 1, N_runs), dtype=bool)
		
		if(mode == 'FFS'):
			flux0_BA_data = np.empty(N_runs)
			OP0_BA_data = np.empty(N_runs)
			probs_BA_data = np.empty((N_OP_interfaces_BA - 1, N_runs))
			PA_BA_data = np.empty((N_OP_interfaces_BA, N_runs))
			d_PA_BA_data = np.empty((N_OP_interfaces_BA, N_runs))
			PA_sigmoid_data = np.empty((N_OP_interfaces_BA - 1, N_runs))
			d_PA_sigmoid_data = np.empty((N_OP_interfaces_BA - 1, N_runs))
			PA_linfit_data = np.empty((2, N_runs))
			PA_linfit_inds_data = np.empty((N_OP_interfaces_BA - 1, N_runs), dtype=bool)
	
	for i in range(N_runs):
		izing.init_rand(i + old_seed)
		if(mode == 'BF'):
			F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, \
				rho_fncs[i], d_rho_fncs[i], k_bc_AB_BF, k_bc_BA_BF, k_AB_BF, _, k_BA_BF, _, _, _, _ = \
					proc_T(L, Temp, h, Nt_per_BF_run, interface_mode, def_spin_state, \
							OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
							verbose=verbose, to_plot_time_evol=to_plot_time_evol, \
							to_plot_F=False, to_plot_correlations=False, to_plot_ETS=False, \
							timeevol_stride=timeevol_stride, to_estimate_k=True, 
							to_get_timeevol=True)
			
			ln_k_bc_AB_data[i] = np.log(k_bc_AB_BF * 1)   # proc_T returns 'k', not 'log(k)'; *1 for units [1/step]
			ln_k_bc_BA_data[i] = np.log(k_bc_BA_BF * 1)
			ln_k_AB_data[i] = np.log(k_AB_BF * 1)
			ln_k_BA_data[i] = np.log(k_BA_BF * 1)
		elif(mode  == 'BF_AB'):
			k_AB_BFcount, _ = \
					proc_T(L, Temp, h, Nt_per_BF_run, interface_mode, def_spin_state, \
							OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
							verbose=verbose, to_plot_time_evol=to_plot_time_evol, \
							to_plot_F=False, to_plot_correlations=False, to_plot_ETS=False, \
							timeevol_stride=timeevol_stride, to_estimate_k=False, 
							to_get_timeevol=True)
			
			ln_k_AB_data[i] = np.log(k_AB_BFcount * 1)
		elif(mode == 'FFS_AB'):
			probs_AB_data[:, i], _, ln_k_AB_data[i], _, flux0_AB_data[i], _, rho_new, d_rho_new, OP_hist_centers_new, OP_hist_lens_new, \
			PB_AB_data[:, i], d_PB_AB_data[:, i], PB_sigmoid_data[:, i], d_PB_sigmoid_data[:, i], PB_linfit_data[:, i], PB_linfit_inds_data[:, i], OP0_AB_data[i] = \
				proc_FFS_AB(L, Temp, h, N_init_states_AB, OP_interfaces_AB, interface_mode, def_spin_state, \
							OP_sample_BF_to=OP_sample_BF_A_to, OP_match_BF_to=OP_match_BF_A_to, init_gen_mode=init_gen_mode, to_get_timeevol=to_get_timeevol)
			if(to_get_timeevol):
				rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, rho_new, fill_value='extrapolate')
				d_rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, d_rho_new, fill_value='extrapolate')
				rho_AB = rho_fncs[i](OP_hist_centers_new)
				d_rho_AB = d_rho_fncs[i](OP_hist_centers_new)
				F_new = -Temp * np.log(rho_AB * OP_hist_lens_new)
				d_F_new = Temp * d_rho_AB / rho_AB
				F_new = F_new - F_new[0]

		elif(mode == 'FFS'):
			F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, rho_fncs[i], d_rho_fncs[i], probs_AB_data[:, i], _, \
				ln_k_AB_data[i], _, flux0_AB_data[i], _, probs_BA_data[:, i], _, ln_k_BA_data[i], _, flux0_BA_data[i], _, \
				PB_AB_data[:, i], d_PB_AB_data[:, i], PA_BA_data[:, i], d_PA_BA_data[:, i], \
				PB_sigmoid_data[:, i], d_PB_sigmoid_data[:, i], PA_sigmoid_data[:, i], d_PA_sigmoid_data[:, i], \
				PB_linfit_data[:, i], PA_linfit_data[:, i], \
				PB_linfit_inds_data[:, i], PA_linfit_inds_data[:, i], \
				OP0_AB_data[i], OP0_BA_data[i] = \
					proc_FFS(L, Temp, h, \
						N_init_states_AB, N_init_states_BA, \
						OP_interfaces_AB, OP_interfaces_BA, \
						OP_sample_BF_A_to, OP_sample_BF_B_to, \
						OP_match_BF_A_to, OP_match_BF_B_to, \
						interface_mode, def_spin_state, \
						init_gen_mode=init_gen_mode, \
						to_get_timeevol=to_get_timeevol)
			
		print(r'%s %lf %%' % (mode, (i+1) / (N_runs) * 100))
		
		if(to_get_timeevol):
			if(i == 0):
				OP_hist_centers = OP_hist_centers_new
				OP_hist_lens = OP_hist_lens_new
				N_OP_hist_points = len(OP_hist_centers)
				F_data = np.empty((N_OP_hist_points, N_runs))
				d_F_data = np.empty((N_OP_hist_points, N_runs))
			else:
				def check_OP_centers_match(c, i, c_new, arr_name, scl=1, err_thr=1e-6):
					rel_err = abs(c_new - c) / scl
					miss_inds = rel_err > err_thr
					assert(np.all(~miss_inds)), r'ERROR: %s from i=0 and i=%d does not match:\nold: %s\nnew: %s\nerr: %s' % (arr_name, i, str(c[miss_inds]), str(c_new[miss_inds]), str(rel_err[miss_inds]))
				
				check_OP_centers_match(OP_hist_centers, i, OP_hist_centers_new, 'OP_hist_centers', max(abs(OP_hist_centers[1:] - OP_hist_centers[:-1])))
				check_OP_centers_match(OP_hist_lens, i, OP_hist_lens_new, 'OP_hist_lens', max(OP_hist_lens))
			F_data[:, i] = F_new
			d_F_data[:, i] = d_F_new
	
	ln_k_AB, d_ln_k_AB = get_average(ln_k_AB_data)
	
	if(to_get_timeevol):
		F, d_F = get_average(F_data, axis=1)
	else:
		F = None
		d_F = None
		OP_hist_centers = None
		OP_hist_lens = None
	
	if('AB' not in mode):
		ln_k_BA, d_ln_k_BA = get_average(ln_k_BA_data)

	if(mode == 'BF'):
		ln_k_bc_AB, d_ln_k_bc_AB = get_average(ln_k_bc_AB_data)
		ln_k_bc_BA, d_ln_k_bc_BA = get_average(ln_k_bc_BA_data)
		
	elif('FFS' in mode):
		flux0_AB, d_flux0_AB = get_average(flux0_AB_data, mode='log')
		OP0_AB, d_OP0_AB = get_average(OP0_AB_data)
		probs_AB, d_probs_AB = get_average(probs_AB_data, mode='log', axis=1)
		PB_AB, d_PB_AB = get_average(PB_AB_data, mode='log', axis=1)

		PB_sigmoid, d_PB_sigmoid, linfit_AB, linfit_AB_inds, m0_AB_fit = \
			get_sigmoid_fit(PB_AB[:-1], d_PB_AB[:-1], OP_AB[:-1], h, table_data.Nc_reference_data[str(Temp)])

		if('AB' not in mode):
			flux0_BA, d_flux0_BA = get_average(flux0_BA_data, mode='log')
			OP0_BA, d_OP0_BA = get_average(OP0_BA_data)
			probs_BA, d_probs_BA = get_average(probs_BA_data, mode='log', axis=1)
			PA_BA, d_PA_BA = get_average(PA_BA_data, mode='log', axis=1)
		
			PA_sigmoid, d_PA_sigmoid, linfit_BA, linfit_BA_inds, m0_BA_fit = \
				get_sigmoid_fit(PA_BA[1:], d_PA_BA[1:], OP_BA[1:], h, table_data.Nc_reference_data[str(Temp)])

		# =========================== 4-param sigmoid =======================
		# if(interface_mode == 'M'):
			# tht0_AB = [0, -0.15]
			# tht0_BA = [tht0_AB[0], - tht0_AB[1]]
		# elif(interface_mode == 'CS'):
			# tht0_AB = [60, 10]
			# tht0_BA = [L2 - tht0_AB[0], - tht0_AB[1]]
		# PB_opt, PB_opt_fnc, PB_opt_fnc_inv = PB_fit_optimize(PB_AB[:-1], d_PB_AB[:-1], OP_AB[:-1], OP_AB[0], OP_AB[-1], PB_AB[0], PB_AB[-1], tht0=tht0_AB)
		# PA_opt, PA_opt_fnc, PA_opt_fnc_inv = PB_fit_optimize(PA_BA[1:], d_PA_BA[1:], OP_BA[1:], OP_BA[0], OP_BA[-1], PA_BA[0], PA_BA[-1], tht0=tht0_BA)
		# OP0_AB = PB_opt.x[0]
		# OP0_BA = PA_opt.x[0]
		# PB_fit_a, PB_fit_b = get_intermediate_vals(PB_opt.x[0], PB_opt.x[1], OP_AB[0], OP_AB[-1], PB_AB[0], PB_AB[-1])
		# PA_fit_a, PA_fit_b = get_intermediate_vals(PA_opt.x[0], PA_opt.x[1], OP_BA[0], OP_BA[-1], PA_BA[0], PA_BA[-1])

		#PB_sigmoid = -np.log(PB_fit_b / (PB_AB[:-1] - PB_fit_a) - 1)
		#d_PB_sigmoid = d_PB_AB[:-1] * PB_fit_b / ((PB_AB[:-1] - PB_fit_a) * (PB_fit_b - PB_AB[:-1] + PB_fit_a))
		#PA_sigmoid = -np.log(PA_fit_b / (PA_BA[:-1] - PA_fit_a) - 1)
		#d_PA_sigmoid = d_PA_BA[1:] * PA_fit_b / ((PA_BA[1:] - PA_fit_a) * (PA_fit_b - PA_BA[1:] + PA_fit_a))
		
	if(to_plot_k_distr):
		plot_k_distr(np.exp(ln_k_AB_data) / 1, N_k_bins, 'k_{AB}', units='step')
		plot_k_distr(ln_k_AB_data, N_k_bins, r'\ln(k_{AB} \cdot 1step)')
		if('AB' not in mode):
			plot_k_distr(np.exp(ln_k_BA_data) / 1, N_k_bins, 'k_{BA}', units='step')
			plot_k_distr(ln_k_BA_data, N_k_bins, r'\ln(k_{BA} \cdot 1step)')
		if(mode == 'BF'):
			plot_k_distr(np.exp(ln_k_bc_AB_data) / 1, N_k_bins, r'k_{bc, AB}', units='step')
			plot_k_distr(ln_k_bc_AB_data, N_k_bins, r'\ln(k_{bc, AB} \cdot 1step)')
			plot_k_distr(np.exp(ln_k_bc_BA_data) / 1, N_k_bins, r'k_{bc, BA}', units='step')
			plot_k_distr(ln_k_bc_BA_data, N_k_bins, r'\ln(k_{bc, BA} \cdot 1step)')
	
	if(to_plot_committer is None):
		to_plot_committer = ('FFS' in mode)
	else:
		assert(not (('BF' in mode) and to_plot_committer)), 'ERROR: BF mode used but commiter requested'
	
	if(to_plot_committer):
		fig_PB_log, ax_PB_log = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
		fig_PB, ax_PB = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		fig_PB_sigmoid, ax_PB_sigmoid = my.get_fig(title[interface_mode], r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		
		mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sigmoid, OP, PB_AB, PB_sigmoid, interface_mode)
		# for i in range(N_runs):
			# plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_{B' + str(i) + '}', feature_label[interface_mode], OP_AB[:-1], PB_AB_data[:-1, i], d_PB=d_PB_AB_data[:-1, i], \
						# PB_sgm=PB_sigmoid_data[:, i], d_PB_sgm=d_PB_sigmoid_data[:, i], \
						# linfit=PB_linfit_data[:, i], linfit_inds=PB_linfit_inds_data[:, i], \
						# OP0=OP0_AB_data[i], d_OP0=None, \
						# clr=my.get_my_color(-i-1))
			# plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_{A' + str(i) + '}', feature_label[interface_mode], OP_BA[1:], PA_BA_data[1:, i], d_PB=d_PA_BA_data[1:, i], \
						# PB_sgm=PA_sigmoid_data[:, i], d_PB_sgm=d_PA_sigmoid_data[:, i], \
						# linfit=PA_linfit_data[:, i], linfit_inds=PA_linfit_inds_data[:, i], \
						# OP0=OP0_BA_data[i], d_OP0=None, \
						# clr=my.get_my_color(i+1))
		plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_B', feature_label[interface_mode], \
					OP_AB[:-1], PB_AB[:-1], d_PB=d_PB_AB[:-1], \
					PB_sgm=PB_sigmoid, d_PB_sgm=d_PB_sigmoid, \
					linfit=linfit_AB, linfit_inds=linfit_AB_inds, \
					OP0=OP0_AB, d_OP0=d_OP0_AB, \
					clr=my.get_my_color(1))
					
		if('AB' not in mode):
			plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_A', feature_label[interface_mode], \
						OP_BA[1:], PA_BA[1:], d_PB=d_PA_BA[1:], \
						PB_sgm=PA_sigmoid, d_PB_sgm=d_PA_sigmoid, \
						linfit=linfit_BA, linfit_inds=linfit_BA_inds, \
						OP0=OP0_BA, d_OP0=d_OP0_BA, \
						clr=my.get_my_color(2))
		
		ax_PB_log.legend()
		ax_PB.legend()
		ax_PB_sigmoid.legend()
	
	izing.init_rand(old_seed)
	
	if(mode == 'FFS'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, flux0_AB, d_flux0_AB, flux0_BA, d_flux0_BA, probs_AB, d_probs_AB, probs_BA, d_probs_BA, OP0_AB, d_OP0_AB, OP0_BA, d_OP0_BA
	elif(mode == 'FFS_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, probs_AB, d_probs_AB, OP0_AB, d_OP0_AB
	elif(mode == 'BF'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, ln_k_bc_AB, d_ln_k_bc_AB, ln_k_bc_BA, d_ln_k_bc_BA
	elif(mode == 'BF_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB

# def get_init_states(Ns, N_min, N0=1):
	# # first value is '1' because it's the size of the states' set from which the initial states for simulations in A will be picked. 
	# # Currently we start all the simulations from 'all spins -1', so there is really only 1 initial state, so values >1 will just mean storing copies of the same state
	# return np.array([N0] + list(Ns / min(Ns) * N_min), dtype=np.intc)
	

def get_h_dependence(h_arr, L, Temp, N_runs, interface_mode, def_spin_state, N_init_states, OP_interfaces, init_gen_mode=-3, to_plot=False):
	L2 = L**2
	N_h = len(h_arr)
	
	ln_k_AB = np.empty(N_h)
	d_ln_k_AB = np.empty(N_h)
	flux0_A = np.empty(N_h)
	d_flux0_A = np.empty(N_h)
	OP0_AB = np.empty(N_h)
	d_OP0_AB = np.empty(N_h)
	#prob_AB = np.empty((N_OP_interfaces - 1, N_h))
	#d_prob_AB = np.empty((N_OP_interfaces - 1, N_h))
	prob_AB = []
	d_prob_AB = []
	k_AB_mean = np.empty(N_h)
	k_AB_min = np.empty(N_h)
	k_AB_max = np.empty(N_h)
	d_k_AB = np.empty(N_h)
	N_OP_interfaces = np.empty(N_h, dtype=int)
	
	for i_h in range(N_h):
		N_OP_interfaces[i_h] = len(OP_interfaces[i_h])
		_, _, _, _, ln_k_AB[i_h], d_ln_k_AB[i_h], flux0_A[i_h], \
			d_flux0_A[i_h], prob_AB_new, d_prob_AB_new, \
			OP0_AB[i_h], d_OP0_AB[i_h] = \
				run_many(L, Temp, h_arr[i_h], N_runs, interface_mode, def_spin_state, \
					N_init_states_AB=N_init_states, \
					OP_interfaces_AB=OP_interfaces[i_h], \
					mode='FFS_AB', init_gen_mode=init_gen_mode, \
					to_get_timeevol=False, to_plot_committer=False)
		prob_AB.append(prob_AB_new)
		d_prob_AB.append(d_prob_AB_new)
		#print(ln_k_AB, d_ln_k_AB, flux0_A, d_flux0_A)
		#print(prob_AB.shape)
		#print(d_prob_AB.shape)
		#print((N_OP_interfaces - 1, N_h))
		#sinput('hi')
		k_AB_mean[i_h], k_AB_min[i_h], k_AB_max[i_h], d_k_AB[i_h] = get_log_errors(ln_k_AB[i_h], d_ln_k_AB[i_h], lbl='k_AB_' + str(i_h))
	
	if(to_plot):
		fig_k, ax_k = my.get_fig('$h/J$', '$k_{AB} / L^2$ [trans/(sweep $\cdot l^2$)]', '$k_{AB}(h)$; T/J = ' + str(Temp) + '; L = ' + str(L), yscl='log')
		fig_Nc, ax_Nc = my.get_fig('$h/J$', '$N_c$', '$N_c(h)$; T/J = ' + str(Temp) + '; L = ' + str(L), xscl='log', yscl='log')

		k_AB_errbars = np.concatenate(((k_AB_mean - k_AB_min)[np.newaxis, :], (k_AB_max - k_AB_mean)[np.newaxis, :]), axis=0)
		ax_k.errorbar(h_arr, k_AB_mean, yerr=k_AB_errbars, fmt='.', label='data')
		
		ax_Nc.errorbar(h_arr, OP0_AB, yerr=d_OP0_AB, fmt='.', label='data')
		
		Temp_key = str(Temp)
		if((Temp_key in table_data.k_AB_reference_data) and (Temp_key in table_data.Nc_reference_data)):
			h_kAB_ref = table_data.k_AB_reference_data[Temp_key][0, :]
			h_Nc_ref = table_data.Nc_reference_data[Temp_key][0, :]
			k_AB_ref = table_data.k_AB_reference_data[Temp_key][1, :] / table_data.L_reference ** 2
			Nc_AB_ref = table_data.Nc_reference_data[Temp_key][1, :]
			
			ax_k.plot(h_kAB_ref, k_AB_ref, '-o', label='reference')
			ax_Nc.plot(h_Nc_ref, Nc_AB_ref, '-o', label='reference')
			
			fig_k_err, ax_k_err = my.get_fig('$h/J$', '$[max(k_{my, ref}) / min(k_{my, ref}) - 1]$', '$k_{error}(h)$; T/J = ' + str(Temp) + '; L = ' + str(L), yscl='log')
			fig_Nc_err, ax_Nc_err = my.get_fig('$h/J$', '$[max(Nc_{my, ref}) / min(Nc_{my, ref}) - 1]$', '$Nc_{error}(h)$; T/J = ' + str(Temp) + '; L = ' + str(L), xscl='log', yscl='log')
			def get_k_err_vals(k, d_ln_k, k_ref):
				r = np.exp(np.abs(np.log(k / k_ref)))
				# d(k/k0 - 1) / (k/k0) = dk/k
				# d(k0/k - 1) / (k0/k) = dk*k0/k^2 / (k0/k) / dk/k
				# So, 'd_ans / r == d_k/k'
				return r - 1, r * d_ln_k
			h_kAB_ref_inds = np.array([np.argmin(np.abs(h_kAB_ref - h_my)) for h_my in h_arr])
			h_Nc_ref_inds = np.array([np.argmin(np.abs(h_Nc_ref - h_my)) for h_my in h_arr])
			k_err, d_k_err = get_k_err_vals(k_AB_mean, d_ln_k_AB, k_AB_ref[h_kAB_ref_inds])
			Nc_err, d_Nc_err = get_k_err_vals(OP0_AB, d_OP0_AB / OP0_AB, Nc_AB_ref[h_Nc_ref_inds])
			
			ax_k_err.errorbar(h_arr, k_err, yerr=d_k_err, fmt='.', label='data')
			ax_k_err.legend()
			
			ax_Nc_err.errorbar(h_arr, Nc_err, yerr=d_Nc_err, fmt='.', label='data')
			ax_Nc_err.legend()
		ax_k.legend()
		ax_Nc.legend()
		

def main():
	# TODO: remove outdated inputs
	[L, h, Temp, mode, Nt, N_states_FFS, N_init_states_FFS, to_recomp, to_get_timeevol, verbose, my_seed, N_OP_interfaces, N_runs, init_gen_mode, OP_0, OP_max, interface_mode, OP_min_BF, OP_max_BF, Nt_sample_A, Nt_sample_B, def_spin_state, N_spins_up_init, to_plot_ETS, interface_set_mode, timeevol_stride, to_plot_timeevol], _ = \
		my.parse_args(sys.argv,            ['-L',  '-h', '-Temp', '-mode',        '-Nt', '-N_states_FFS', '-N_init_states_FFS',     '-to_recomp', '-to_get_timeevol', '-verbose', '-my_seed', '-N_OP_interfaces', '-N_runs', '-init_gen_mode', '-OP_0', '-OP_max', '-interface_mode', '-OP_min_BF', '-OP_max_BF', '-Nt_sample_A', '-Nt_sample_B', '-def_spin_state', '-N_spins_up_init',   '-to_plot_ETS', '-interface_set_mode', '-timeevol_stride', '-to_plot_timeevol'], \
					  possible_arg_numbers=[[1],  ['+'],     [1],     [1],       [0, 1],          [0, 1],               [0, 1],           [0, 1],             [0, 1],     [0, 1],     [0, 1],                [1],    [0, 1],           [0, 1],   ['+'],     ['+'],               [1],       [0, 1],       [0, 1],         [0, 1],         [0, 1],            [0, 1],             [0, 1],           [0, 1],                [0, 1],             [0, 1],              [0, 1]], \
					  default_values=      [None,  None,    None,    None, ['-1000000'],       ['-5000'],             ['5000'], [my.no_flags[0]],              ['1'],      ['1'],     ['23'],               None,     ['3'],           ['-3'],    None,      None,              None,       [None],       [None],   ['-1000000'],   ['-1000000'],            ['-1'],             [None], [my.no_flags[0]],           ['optimal'],          ['-3000'],     [my.no_flags[0]]])
	
	# efficiency info:
	# [M0=20, NM=20] : Nt / N_states_FFS ~ 1e4
	
	L = int(L)
	L2 = L**2
	h = np.array([float(x) for x in h])   # arbitrary small number that gives nice pictures
	N_h = len(h)
	
	set_OP_defaults(L2)

	my_seed = int(my_seed[0])
	to_recomp = (to_recomp[0] in my.yes_flags)
	#mode = mode[0]
	to_get_timeevol = (to_get_timeevol[0] in my.yes_flags)
	verbose = int(verbose[0])
	N_init_states_FFS = int(N_init_states_FFS[0])
	Nt = int(Nt[0])
	N_states_FFS = int(N_states_FFS[0])
	N_OP_interfaces = int(N_OP_interfaces)
	N_runs = int(N_runs[0])
	init_gen_mode = int(init_gen_mode[0])
	OP_0 = np.array(([int(OP_0[0])] * N_h) if(len(OP_0) == 1) else [int(x) for x in OP_0], dtype=int)
	OP_max = np.array(([int(OP_max[0])] * N_h) if(len(OP_max) == 1) else [int(x) for x in OP_max], dtype=int)
	interface_mode = interface_mode
	def_spin_state = int(def_spin_state[0])
	N_spins_up_init = (None if(N_spins_up_init[0] is None) else int(N_spins_up_init[0]))
	to_plot_ETS = (to_plot_ETS[0] in my.yes_flags)
	interface_set_mode = interface_set_mode[0]
	timeevol_stride = int(timeevol_stride[0])
	to_plot_timeevol = (to_plot_timeevol[0] in my.yes_flags)
	
	OP_min_BF = OP_min_default[interface_mode] if(OP_min_BF[0] is None) else int(OP_min_BF[0])
	OP_max_BF = OP_max_default[interface_mode] if(OP_max_BF[0] is None) else int(OP_max_BF[0])
	
	assert(interface_mode == 'CS'), 'ERROR: interface_mode==M is temporary not supported'

	# -------- T < Tc, transitions ---------
	Temp = float(Temp)  # T_c = 2.27
	# -------- T > Tc, fluctuations around 0 ---------
	#Temp = 3.0
	#Temp = 2.5   # looks like Tc for L=16
	
	print('T=%lf, L=%d, h=%s, mode=%s, Nt=%d, N_states_FFS=%d, N_OP_interfaces=%d, interface_mode=%s, N_runs=%d, init_gen_mode=%d, OP_0=%s, OP_max=%s, def_spin_state=%d, N_spins_up_init=%s, OP_min_BF=%d, OP_max_BF=%d, to_get_timeevol=%r, verbose=%d, my_seed=%d, to_recomp=%r' % \
		(Temp, L, str(h), mode, Nt, N_states_FFS, N_OP_interfaces, interface_mode, N_runs, init_gen_mode, str(OP_0), str(OP_max), def_spin_state, str(N_spins_up_init), OP_min_BF, OP_max_BF, to_get_timeevol, verbose, my_seed, to_recomp))
	
	izing.init_rand(my_seed)
	izing.set_verbose(verbose)

	Nt, N_states_FFS = \
		tuple([((int(-n * np.exp(3 - dE_avg/Temp)) + 1) if(n < 0) else n) for n in \
				[Nt, N_states_FFS]])

	BC_cluster_size = L2 / np.pi

	if(('compare' in mode) or ('many' in mode)):
		N_k_bins=int(np.round(np.sqrt(N_runs) / 2) + 1)
	
	if('FFS' in mode):
		assert(not(('AB' not in mode) and (interface_mode == 'CS'))), 'ERROR: BA transitions is not supported for CS'
		OP_left = OP_min_default[interface_mode]
		OP_right = OP_max_default[interface_mode] - 1
		
		OP_interfaces_AB = []
		OP_interfaces_BA = []
		OP_match_BF_A_to = np.empty(N_h, dtype=int)
		OP_sample_BF_A_to = np.empty(N_h, dtype=int)
		for i in range(N_h):
			if(N_OP_interfaces > 0):
				interface_AB_UID = (interface_mode, L, h[i], N_OP_interfaces, OP_0[i], OP_max[i], interface_set_mode)
				if(interface_AB_UID in table_data.OP_interfaces_table):
					OP_interfaces_AB.append(table_data.OP_interfaces_table[interface_AB_UID])
					assert(N_OP_interfaces == len(OP_interfaces_AB[i])), 'ERROR: OP_interfaces number of elements does not match N_OP_interfaces'
				else:
					assert(input('WARNING: no custom interfaces found for %s, using equly-spaced interfaces, proceed? (y/N)\n' % (str(interface_AB_UID))) in my.yes_flags), 'exiting'
				if('AB' not in mode):
					interface_BA_UID = (interface_mode, L, -h[i], N_OP_interfaces, OP_0[i], OP_max[i], interface_set_mode)
					if(interface_BA_UID in table_data.OP_interfaces_table):
						OP_interfaces_BA.append(table_data.OP_interfaces_table[interface_BA_UID])
					else:
						assert(input('WARNING: no custom interfaces found for %s, using equly-spaced interfaces, proceed? (y/N)\n' % (str(interface_BA_UID))) in my.yes_flags), 'exiting'
			else:
				# =========== if the necessary set in not in the table - this will work and nothing will erase its results ========
				OP_interfaces_AB.append(OP_0[i] + np.round(np.arange(abs(N_OP_interfaces)) * (OP_max[i] - OP_0[i]) / (abs(N_OP_interfaces) - 1) / OP_step[interface_mode]) * OP_step[interface_mode])
					# this gives Ms such that there are always pairs +-M[i], so flipping this does not move the intefraces, which is (is it?) good for backwords FFS (B->A)
				if('AB' not in mode):
					OP_interfaces_BA.append(flip_OP(OP_interfaces_AB[0], L2, interface_mode))
			
			# ======= for high 'h' : Nc << L2/2 ========
			OP_match_BF_A_to[i] = OP_interfaces_AB[i][-3]
			OP_sample_BF_A_to[i] = OP_interfaces_AB[i][-2]

			# ======= for small 'h' : Nc ~ L2/2 ========
			#if(interface_mode == 'M'):
				# for h ~ 0
				# OP_0 ~ -L2 * 5/6
				# OP_max ~ -OP_0
				#OP_match_BF_A_to = OP_0 + int(L2 * 0.25)   # this should be sufficiently < M_sample_to, because BF_profile is distorted near the M_sample_to; But is also should be > OP_interface[1] since we need enough points to match F curves.
				#OP_sample_BF_A_to = OP_match_BF_A_to + int(L2 * 0.2)
				#OP_match_BF_B_to = OP_max - int(L2 * 0.25)
				#OP_sample_BF_B_to = OP_match_BF_B_to - int(L2 * 0.2)
			#elif(interface_mode == 'CS'):
				# OP_0 ~ 
				# OP_max ~
				# OP_match_BF_A_to = OP_0 + 20   # this should be sufficiently < M_sample_to, because BF_profile is distorted near the M_sample_to; But is also should be > OP_interface[1] since we need enough points to match F curves.
				# OP_sample_BF_A_to = OP_match_BF_A_to + 20
				# if(OP_sample_BF_A_to > L2 * 0.9):
					# print('WARNING: too high OP_sample_A_to = %d > L * 0.9 = %d' % (OP_sample_BF_A_to, int(L2 * 0.9)))
					# assert(OP_sample_BF_A_to < L2), 'ERROR: too high OP_sample_A_to = %d >= L = %d' % (OP_sample_BF_A_to, L2)
				# if('AB' not in mode):
					# OP_match_BF_B_to = OP_max - 30
					# OP_sample_BF_B_to = OP_match_BF_B_to - 20
					# if(OP_sample_BF_B_to < L2 * 0.1):
						# print('WARNING: too low OP_sample_B_to = %d < L * 0.1 = %d' % (OP_sample_BF_B_to, int(L2 * 0.1)))
						# assert(OP_sample_BF_B_to > 0), 'ERROR: too low OP_sample_B_to = %d <= 0' % (OP_sample_BF_B_to)
		
		N_init_states_AB = np.ones(N_OP_interfaces, dtype=np.intc) * N_states_FFS
		N_init_states_BA = np.ones(N_OP_interfaces, dtype=np.intc) * N_states_FFS
		N_init_states_AB[0] = N_init_states_FFS
		N_init_states_BA[0] = N_init_states_FFS
		
		print('OP_AB:', OP_interfaces_AB)
		print('OP_BA:', OP_interfaces_BA)
		
	if(mode == 'BF_1'):
		proc_T(L, Temp, h[0], Nt, interface_mode, def_spin_state, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_time_evol=to_plot_timeevol, to_plot_F=True, to_plot_ETS=to_plot_ETS, \
				to_plot_correlations=True and to_plot_timeevol, to_get_timeevol=True, \
				OP_min=OP_min_BF, OP_max=OP_max_BF, to_estimate_k=True, timeevol_stride=timeevol_stride)
		
	if(mode == 'BF_AB'):
		proc_T(L, Temp, h[0], Nt, interface_mode, def_spin_state, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_time_evol=to_plot_timeevol, to_plot_F=True, to_plot_ETS=to_plot_ETS, \
				to_plot_correlations=True and to_plot_timeevol, to_get_timeevol=to_get_timeevol, \
				OP_min=OP_min_BF, OP_max=OP_max[0], to_estimate_k=False, timeevol_stride=timeevol_stride)
	
	elif(mode == 'BF_many'):
		run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, Nt_per_BF_run=Nt, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, \
				mode='BF')
	
	elif(mode == 'BF_AB_many'):
		run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, Nt_per_BF_run=Nt, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, \
				mode='BF_AB')
	
	elif(mode == 'FFS_AB'):
		proc_FFS_AB(L, Temp, h[0], N_init_states_AB, OP_interfaces_AB[0], interface_mode, def_spin_state, \
					OP_sample_BF_to=OP_sample_BF_A_to[0], OP_match_BF_to=OP_match_BF_A_to[0], timeevol_stride=timeevol_stride, \
					to_get_timeevol=to_get_timeevol, init_gen_mode=init_gen_mode, to_plot_time_evol=to_plot_timeevol, to_plot_hists=True)
		
	elif(mode == 'FFS'):
		proc_FFS(L, Temp, h[0], \
				N_init_states_AB, N_init_states_BA, \
				OP_interfaces_AB[0], OP_interfaces_BA[0], \
				OP_sample_BF_A_to[0], OP_sample_BF_B_to[0], \
				OP_match_BF_A_to[0], OP_match_BF_B_to[0], \
				interface_mode, def_spin_state, \
				init_gen_mode=init_gen_mode, \
				to_plot_hists=True)
		
	elif(mode == 'FFS_many'):
		run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, \
				OP_interfaces_AB=OP_interfaces_AB[0], \
				OP_interfaces_BA=OP_interfaces_BA[0], \
				OP_sample_BF_A_to=OP_sample_BF_A_to[0], OP_sample_BF_B_to=OP_sample_BF_B_to[0], \
				OP_match_BF_A_to=OP_match_BF_A_to[0], OP_match_BF_B_to=OP_match_BF_B_to[0], \
				N_init_states_AB=N_init_states_AB, N_init_states_BA=N_init_states_BA, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, mode='FFS', \
				init_gen_mode=init_gen_mode)
	elif(mode == 'FFS_AB_many'):
		F_FFS, d_F_FFS, M_hist_centers_FFS, M_hist_lens_FFS, ln_k_AB_FFS, d_ln_k_AB_FFS, \
			flux0_AB_FFS, d_flux0_AB_FFS, prob_AB_FFS, d_prob_AB_FFS, OP0_AB, d_OP0_AB = \
			run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, \
				N_init_states_AB=N_init_states_AB, \
				OP_interfaces_AB=OP_interfaces_AB[0], \
				to_plot_k_distr=False, N_k_bins=N_k_bins, \
				mode='FFS_AB', init_gen_mode=init_gen_mode, \
				OP_sample_BF_A_to=OP_sample_BF_A_to[0], \
				OP_match_BF_A_to=OP_match_BF_A_to[0], \
				to_get_timeevol=to_get_timeevol)
		
		get_log_errors(np.log(flux0_AB_FFS), d_flux0_AB_FFS / flux0_AB_FFS, lbl='flux0_AB', print_scale=print_scale_flux)
		k_AB_FFS_mean, k_AB_FFS_min, k_AB_FFS_max, d_k_AB_FFS = get_log_errors(ln_k_AB_FFS, d_ln_k_AB_FFS, lbl='k_AB_FFS', print_scale=print_scale_k)
		
	elif(mode == 'FFS_AB_h'):
		get_h_dependence(h, L, Temp, N_runs, interface_mode, def_spin_state, N_init_states_AB, OP_interfaces_AB, init_gen_mode=init_gen_mode, to_plot=True)
	
	elif('compare' in mode):
		if('AB' in mode):
			F_FFS, d_F_FFS, M_hist_centers_FFS, M_hist_lens_FFS, ln_k_AB_FFS, d_ln_k_AB_FFS, \
				flux0_AB_FFS, d_flux0_AB_FFS, prob_AB_FFS, d_prob_AB_FFS, OP0_AB, d_OP0_AB = \
					run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, \
						N_init_states_AB=N_init_states_AB, \
						OP_interfaces_AB=OP_interfaces_AB[0], \
						to_plot_k_distr=False, N_k_bins=N_k_bins, \
						mode='FFS_AB', init_gen_mode=init_gen_mode, \
						OP_sample_BF_A_to=OP_sample_BF_A_to[0], \
						OP_match_BF_A_to=OP_match_BF_A_to[0], \
						to_get_timeevol=to_get_timeevol)
		else:
			F_FFS, d_F_FFS, M_hist_centers_FFS, M_hist_lens_FFS, \
				ln_k_AB_FFS, d_ln_k_AB_FFS, ln_k_BA_FFS, d_ln_k_BA_FFS, \
				flux0_AB_FFS, d_flux0_AB_FFS, flux0_BA_FFS, d_flux0_BA_FFS, \
				prob_AB_FFS, d_prob_AB_FFS, prob_BA_FFS, d_prob_BA_FFS, \
				OP0_AB, d_OP0_AB, OP0_BA, d_OP0_BA = \
					run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, \
						N_init_states_AB=N_init_states_AB, \
						N_init_states_BA=N_init_states_BA, \
						OP_interfaces_AB=OP_interfaces_AB[0], \
						OP_interfaces_BA=OP_interfaces_BA, \
						to_plot_k_distr=False, N_k_bins=N_k_bins, \
						mode='FFS', init_gen_mode=init_gen_mode, \
						OP_sample_BF_A_to=OP_sample_BF_A_to[0], OP_sample_BF_B_to=OP_sample_BF_B_to[0], \
						OP_match_BF_A_to=OP_match_BF_A_to[0], OP_match_BF_B_to=OP_match_BF_B_to[0], \
						to_get_timeevol=to_get_timeevol)

		F_BF, d_F_BF, M_hist_centers_BF, M_hist_lens_BF, ln_k_AB_BF, d_ln_k_AB_BF, ln_k_BA_BF, d_ln_k_BA_BF, \
			ln_k_bc_AB_BF, d_ln_k_bc_AB_BF, ln_k_bc_BA_BF, d_ln_k_bc_BA_BF = \
				run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, \
						OP_A=OP_0[0], OP_B=OP_max[0], \
						Nt_per_BF_run=Nt, mode='BF', \
						to_plot_k_distr=False, N_k_bins=N_k_bins)
		#F_BF = F_BF - min(F_BF)
		F_BF = F_BF - F_BF[0]
		
		#assert(np.all(abs(M_hist_centers_BF - M_hist_centers_FFS) < (M_hist_centers_BF[0] - M_hist_centers_BF[1]) * 1e-3)), 'ERROR: returned M_centers do not match'
		#assert(np.all(abs(M_hist_lens_BF - M_hist_lens_FFS) < M_hist_lens_BF[0] * 1e-3)), 'ERROR: returned M_centers do not match'

		k_bc_AB_BF_mean, k_bc_AB_BF_min, k_bc_AB_BF_max, d_k_bc_AB_BF = get_log_errors(ln_k_bc_AB_BF, d_ln_k_bc_AB_BF, lbl='k_bc_AB_BF', print_scale=print_scale_k)
		k_AB_BF_mean, k_AB_BF_min, k_AB_BF_max, d_k_AB_BF = get_log_errors(ln_k_AB_BF, d_ln_k_AB_BF, lbl='k_AB_BF', print_scale=print_scale_k)
		get_log_errors(np.log(flux0_AB_FFS), d_flux0_AB_FFS / flux0_AB_FFS, lbl='flux0_AB', print_scale=print_scale_flux)
		k_AB_FFS_mean, k_AB_FFS_min, k_AB_FFS_max, d_k_AB_FFS = get_log_errors(ln_k_AB_FFS, d_ln_k_AB_FFS, lbl='k_AB_FFS', print_scale=print_scale_k)

		k_bc_BA_BF_mean, k_bc_BA_BF_min, k_bc_BA_BF_max, d_k_bc_BA_BF = get_log_errors(ln_k_bc_BA_BF, d_ln_k_bc_BA_BF, lbl='k_bc_BA_BF', print_scale=print_scale_k)
		k_BA_BF_mean, k_BA_BF_min, k_BA_BF_max, d_k_BA_BF = get_log_errors(ln_k_BA_BF, d_ln_k_BA_BF, lbl='k_BA_BF', print_scale=print_scale_k)
		if('AB' not in mode):
			get_log_errors(np.log(flux0_BA_FFS), d_flux0_BA_FFS / flux0_BA_FFS, lbl='flux0_BA', print_scale=print_scale_flux)
			k_BA_FFS_mean, k_BA_FFS_min, k_BA_FFS_max, d_k_BA_FFS = get_log_errors(ln_k_BA_FFS, d_ln_k_BA_FFS, lbl='k_BA_FFS', print_scale=print_scale_k)
		
		fig_F, ax_F = my.get_fig(title[interface_mode], r'$F / J$', title=r'$F(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h[0]))
		ax_F.errorbar(M_hist_centers_BF, F_BF, yerr=d_F_BF, label='BF')
		if(to_get_timeevol):
			ax_F.errorbar(M_hist_centers_FFS, F_FFS, yerr=d_F_FFS, label='FFS')
			all_plotted_F = np.concatenate((F_BF, F_FFS))
			F_lims = [min(all_plotted_F), max(all_plotted_F)]
		else:
			F_lims = [min(F_BF), max(F_BF)]
		for i in range(N_OP_interfaces):
			ax_F.plot([OP_interfaces_AB[0][i] / OP_scale[interface_mode]] * 2, F_lims, '--', label='interfaces' if(i == 0) else None, color=my.get_my_color(2))
		#ax_F.plot([OP_0 / OP_scale[interface_mode]] * 2, F_lims, '--', label='states A, B', color=my.get_my_color(3))
		#ax_F.plot([OP_max / OP_scale[interface_mode]] * 2, F_lims, '--', label=None, color=my.get_my_color(3))
		if(interface_mode == 'CS'):
			ax_F.plot([BC_cluster_size] * 2, F_lims, '--', label='$S_{BC} = L^2 / \pi$')
		ax_F.legend()
		
	elif(mode == 'XXX'):
		fig_E, ax_E = my.get_fig('T', '$E/L^2$', xscl='log')
		fig_OP, ax_OP = my.get_fig('T', '$M/L^2$', xscl='log')
		fig_C, ax_C = my.get_fig('T', '$C/L^2$', xscl='log', yscl='log')
		fig_M2, ax_M2 = my.get_fig('T', '$(M/L^2)^2$')
		
		T_arr = np.power(10, np.linspace(T_min_log10, T_max_log10, N_T))
		
		for n in N_s:
			T_arr = proc_N(n, Nt, my_seed, ax_C=ax_C, ax_E=ax_E, ax_OP=ax_OP, ax_M2=ax_M2, to_recomp=to_recomp, T_min_log10=T_min_log10, T_max_log10=T_max_log10, N_T=N_T)
		
		ax_E.plot(T_arr, -np.tanh(1/T_arr)*2, '--', label='$-2 th(1/T)$')
		T_simulated_ind = (min(T_arr) < T_table) & (T_table < max(T_arr))
		ax_E.plot(T_table[T_simulated_ind], E_table[T_simulated_ind], '.', label='Onsager')

		ax_C.legend()
		ax_E.legend()
		ax_OP.legend()
	
	else:
		print('ERROR: mode=%s is not supported' % (mode))

	plt.show()

if(__name__ == "__main__"):
	main()

