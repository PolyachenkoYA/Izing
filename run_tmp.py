import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy
import sys
import scipy.interpolate
import filecmp

import mylib as my
import table_data
import izing

# ========================== general params ==================
dim = 2
z_neib = dim * 2
N_species = 3

dE_avg = 6   # J
print_scale_k = 1e5
print_scale_flux = 1e2
OP_min_default = {}
OP_max_default = {}
OP_scale = {}
dOP_step = {}
OP_std_overestimate = {}
OP_hist_edges_default = {}
OP_possible_jumps = {}

mu_shift = -np.log(N_species - 1)
mu_shift = 0

OP_C_id = {}
OP_C_id['M'] = 0
OP_C_id['CS'] = 1

OP_step = {}
OP_step['M'] = 1
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

# ========================== recompile ==================
if(__name__ == "__main__"):
	to_recompile = True
	if(to_recompile):
		# try:
			# which_cmake_res = my.run_it('which cmake', check=True, verbose=False)
			# cmake_exists = True
		# except:
			# cmake_exists = False
		if(my.check_for_exe('cmake') is None):
			print('cmake not found. Attempting to import existing library', file=sys.stderr)
		else:
			compile_modes = {'yp1065' : 'della', 'ypolyach' : 'local'}
			username = my.get_username()
			compile_mode = compile_modes[username]
			
			filebase = 'lattice_gas'
			if(compile_mode == 'della'):
				build_dir = 'build'
			else:
				build_dir = 'cmake-build-release'
			path_to_so = os.path.join(filebase, build_dir)
			N_dirs_down = path_to_so.count('/') + 1
			path_back = '/'.join(['..'] * N_dirs_down)
			
			os.chdir(path_to_so)
			# my.run_it(r'echo | gcc -E -Wp,-v -')
			# my.run_it(r'find /usr/include -name "*gsl_rng.h"', verbose=True)
			# my.run_it(r'find /usr/include -name "*gsl_rng.h"', verbose=True)
			# my.run_it('ls /usr/include', verbose=True)
			# my.run_it('ls /usr/include/gsl | grep rng', verbose=True)
			
			if(compile_mode == 'della'):
				compile_results = my.run_it('make %s.so -j 9' % (filebase), check=True).split('\n')
			else:
				compile_results = my.run_it('cmake --build . --target %s.so -j 9' % (filebase), check=True).split('\n')
			
			N_recompile_log_lines = len(compile_results)
			
			assert(N_recompile_log_lines > 0), 'ERROR: nothing printed by make'
			if(compile_mode == 'della'):
				nothing_done_in_recompile = (compile_results[0] == '[100%] Built target lattice_gas.so') and (N_recompile_log_lines <= 2)
			else:
				nothing_done_in_recompile = np.any(np.array([('ninja: no work to do.' in s) for s in compile_results]))
			os.chdir(path_back)
			if(nothing_done_in_recompile):
				print('Nothing done, keeping the old .so lib')
			else:
				py_suffix = my.run_it('python3-config --extension-suffix', check=True, verbose=False)[:-1]   # [:-1] to remove "\n"
				src = '%s/%s.so%s' % (path_to_so, filebase, py_suffix)
				dst_suff = '_tmp'
				#dst_suff = ''
				dst = './%s%s.so' % (filebase, dst_suff)
				
				#print(compile_results)
				#exit()
				
				to_copy = (not filecmp.cmp(src, dst, shallow=False)) if(os.path.isfile(dst)) else True
				if(to_copy):
					# TODO: ask the user only after checking tasks with 'squeue --me'
					if(input('About to copy "%s" to "%s". This may cause disruption of active running tasks.\nProceed? (y/[n]) -> ' % (src, dst)) == 'y'):
					#if(True):
						my.run_it('cp %s %s' % (src, dst))
						print('recompiled %s' % (filebase))
					else:
						print('Not copying the new .so, using the old version')
				else:
					print('"%s" and "%s" seem to be the same, not copying anything' % (src, dst))
				#input('ok')
	
	#import lattice_gas
	import lattice_gas_tmp as lattice_gas
	
	move_modes = lattice_gas.get_move_modes()
	
	#print(move_modes)
	#exit()

# ========================== functions ==================

def get_ThL_lbl(e, mu, init_composition, L, swap_type_move):
	return r'$\epsilon_{11,12,22}/T = ' + my.join_lbls([e[1,1], e[1,2], e[2,2]], ',') + \
				'$;\n$' + ((r'\phi_{1,2} = ' + my.join_lbls(init_composition[1:], ',')) if(swap_type_move) \
						else (r'\mu_{1,2}/T = ' + my.join_lbls(mu[1:], ','))) + '$; L = ' + str(L)

def set_OP_defaults(L2):
	OP_min_default['M'] = - 1
	OP_max_default['M'] = L2 + 1
	OP_scale['M'] = L2
	dOP_step['M'] = OP_step['M'] / OP_scale['M']
	OP_std_overestimate['M'] = 2 * L2
	OP_possible_jumps['M'] = np.array([-1, 0, 1])
	OP_peak_guess['M'] = 150
	
	OP_min_default['CS'] = -1
	OP_max_default['CS'] = L2 + 1
	OP_scale['CS'] = 1
	dOP_step['CS'] = OP_step['CS'] / OP_scale['CS']
	OP_std_overestimate['CS'] = L2
	OP_possible_jumps['CS'] = np.arange(-L2 // 2 - 1, L2 // 2 + 1)
	OP_peak_guess['CS'] = L2 // 2

def eFull_from_eCut(e_cut):
	assert(len(e_cut) == N_species * (N_species - 1) // 2), 'ERROR: invalid number of components in e_cut:' + str(e_cut)
	e_full = np.zeros((N_species, N_species))
	e_full[1,1] = e_cut[0]
	e_full[1,2] = e_cut[1]
	e_full[2,2] = e_cut[2]
	e_full[2,1] = e_full[1,2]
	
	return e_full

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
	fig, ax, _ = my.get_fig(r'$%s$%s' % (lbl, units_inv_str), r'$\rho$%s' % (units_str), r'$\rho(%s)$' % (lbl))

	ax.bar(k_centers, k_hist / k_lens, yerr=d_k_hist / k_lens, width=k_lens, align='center')

def flip_OP(OP, L2, interface_mode):
	if(interface_mode == 'M'):
		return -np.flip(OP)
	elif(interface_mode == 'CS'):
		return L2 - np.flip(OP)

def get_sigmoid_fit(PB, d_PB, OP, sgminv_fnc, d_sgminv_fnc=None, sgminv_dx=1e-5, fit_w=None, bad_points_mult=1.1):
	ok_inds = (PB < 1) & (d_PB > 0)
	N_OP = len(OP)
	PB_sigmoid = np.empty(N_OP)
	d_PB_sigmoid = np.empty(N_OP)
	
	if(d_sgminv_fnc is None):
		d_sgminv_fnc = lambda x, y: (sgminv_fnc(x + sgminv_dx) - sgminv_fnc(x - sgminv_dx)) / (2 * sgminv_dx)
	PB_sigmoid[ok_inds] = sgminv_fnc(PB[ok_inds]) #-np.log(1 / PB[ok_inds] - 1)
	#d_PB_sigmoid[ok_inds] = d_PB[ok_inds] / (PB[ok_inds] * (1 - PB[ok_inds]))
	d_PB_sigmoid[ok_inds] = d_PB[ok_inds] * d_sgminv_fnc(PB[ok_inds], PB_sigmoid[ok_inds])
	PB_sigmoid[~ok_inds] = max(PB_sigmoid[ok_inds]) * bad_points_mult
	d_PB_sigmoid[~ok_inds] = max(d_PB_sigmoid[ok_inds]) * bad_points_mult
	
	OP0_guess =  OP[ok_inds][np.argmin(abs( PB_sigmoid[ok_inds] - sgminv_fnc(1/2)))]
	if(fit_w is None):
		fit_w = abs(OP[0] - OP0_guess) / 5
	linfit_inds = (OP0_guess - fit_w < OP) & (OP < OP0_guess + fit_w) & ok_inds
	N_min_points_for_fit = 2
	if(np.sum(linfit_inds) < N_min_points_for_fit):
		dOP = abs(OP - OP0_guess)
		linfit_inds = (dOP <= np.sort(dOP)[N_min_points_for_fit - 1])
	
	linfit = np.polyfit(OP[linfit_inds], PB_sigmoid[linfit_inds], 1, w=1/d_PB_sigmoid[linfit_inds])
	OP0 = - linfit[1] / linfit[0]
	
	return PB_sigmoid, d_PB_sigmoid, linfit, linfit_inds, OP0

def plot_sgm_fit(ax, ax_log, ax_sgm, x_lbl, clr, \
				OP, PB, sgm_fnc, d_PB=None, PB_sgm=None, d_PB_sgm=None, \
				linfit_sgm=None, linfit_sgm_inds=None, \
				OP0_sgm=None, d_OP0_sgm=None, \
				N_fine_points=-10, linestyle='--'):
	OP0_sgm_str = x_lbl + ' = ' + (my.f2s(OP0_sgm) if(d_OP0_sgm is None) else my.errorbar_str(OP0_sgm, d_OP0_sgm))
	#ax.errorbar(OP, PB, yerr=d_PB, fmt='.', label='data', color=clr)
	if(linfit_sgm_inds is not None):
		OP_near_OP0sgm = OP[linfit_sgm_inds]
		N_fine_points_near_OP0sgm = N_fine_points if(N_fine_points > 0) else int(-(max(OP_near_OP0sgm) - min(OP_near_OP0sgm)) / min(OP_near_OP0sgm[1:] - OP_near_OP0sgm[:-1]) * N_fine_points)
		OP_near_OP0sgm_fine = np.linspace(min(OP_near_OP0sgm), max(OP_near_OP0sgm), N_fine_points_near_OP0sgm)
		if(linfit_sgm is not None):
			#sigmoid_points_near_OP0sgm = 1 / (1 + np.exp((OP0_sgm - OP_near_OP0sgm_fine) * linfit_sgm[0]))
			#sigmoid_points_near_OP0sgm = sgm_fnc((OP0_sgm - OP_near_OP0sgm_fine) * linfit_sgm[0])
			sigmoid_points_near_OP0sgm = sgm_fnc(np.polyval(linfit_sgm, OP_near_OP0sgm_fine))
			plot_kwargs = {'label' : '$' + OP0_sgm_str + '$'}
			if(clr is not None):
				plot_kwargs['color'] = clr
			#ax.plot(OP_near_OP0sgm_fine, sigmoid_points_near_OP0sgm, linestyle, label='$' + OP0_sgm_str + '$', color=clr)
			#ax_log.plot(OP_near_OP0sgm_fine, sigmoid_points_near_OP0sgm, linestyle, label='$' + OP0_sgm_str + '$', color=clr)
			ax.plot(OP_near_OP0sgm_fine, sigmoid_points_near_OP0sgm, linestyle, **plot_kwargs)
			ax_log.plot(OP_near_OP0sgm_fine, sigmoid_points_near_OP0sgm, linestyle, **plot_kwargs)

	if(PB_sgm is not None):
		plot_kwargs = {'yerr': d_PB_sgm, 'fmt': '.', 'label' : 'data'}
		if(clr is not None):
			plot_kwargs['color'] = clr
		#ax_sgm.errorbar(OP, PB_sgm, yerr=d_PB_sgm, fmt='.', label='data', color=clr)
		ax_sgm.errorbar(OP, PB_sgm, **plot_kwargs)
		if(linfit_sgm_inds is not None):
			plot_kwargs = {'label' : '$' + OP0_sgm_str + '$'}
			if(clr is not None):
				plot_kwargs['color'] = clr
			#ax_sgm.plot(OP_near_OP0sgm_fine, np.polyval(linfit_sgm, OP_near_OP0sgm_fine), label='$' + OP0_sgm_str + '$', color=clr)
			ax_sgm.plot(OP_near_OP0sgm_fine, np.polyval(linfit_sgm, OP_near_OP0sgm_fine), **plot_kwargs)

def plot_PB_AB(ThL_lbl, x_lbl, y_lbl, interface_mode, OP, PB, d_PB=None, \
				PB_sgm=None, d_PB_sgm=None, linfit_sgm=None, linfit_sgm_inds=None, OP0_sgm=None, d_OP0_sgm=None, \
				PB_erfinv=None, d_PB_erfinv=None, linfit_erfinv=None, linfit_erfinv_inds=None, OP0_erfinv=None, d_OP0_erfinv=None, \
				clr=None, N_fine_points=-10, to_plot_sgm=False, to_plot_erfinv=True):
	fig_PB_log, ax_PB_log, _ = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; ' + ThL_lbl, yscl='log')
	fig_PB, ax_PB, _ = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; ' + ThL_lbl)
	
	fig_PB_sgm, ax_PB_sgm, _ = my.get_fig(y_lbl, r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + x_lbl + ')$; ' + ThL_lbl) if(to_plot_sgm) else tuple([None] * 3)
	fig_PB_erfinv, ax_PB_erfinv, _ = my.get_fig(y_lbl, r'$erf^{-1}(2 P_B - 1)$', title=r'$P_B(' + x_lbl + ')$; ' + ThL_lbl) if(to_plot_erfinv) else tuple([None] * 3)
	
	mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sgm, ax_PB_erfinv, \
				OP, PB[:-1], PB_sgm, PB_erfinv, interface_mode)
	
	#ax_PB.errorbar(OP, PB, yerr=d_PB, fmt='.', label='data', color=clr)
	#ax_PB_log.errorbar(OP, PB, yerr=d_PB, fmt='.', label='data', color=clr)
	plot_kwargs = {'yerr' : d_PB, 'fmt' : '.', 'label' : 'data'}
	if(clr is not None):
		plot_kwargs['color'] = clr
	ax_PB.errorbar(OP, PB, **plot_kwargs)
	ax_PB_log.errorbar(OP, PB, **plot_kwargs)
	
	if(to_plot_sgm):
		plot_sgm_fit(ax_PB, ax_PB_log, ax_PB_sgm, x_lbl + '_{1/2, \sigma}', clr, \
					OP, PB, lambda x: 1 / (1 + np.exp(-x)), \
					d_PB=d_PB, PB_sgm=PB_sgm, d_PB_sgm=d_PB_sgm, \
					linfit_sgm=linfit_sgm, linfit_sgm_inds=linfit_sgm_inds, \
					OP0_sgm=OP0_sgm, d_OP0_sgm=d_OP0_sgm, \
					N_fine_points=N_fine_points, linestyle=':')
	
	if(to_plot_erfinv):
		plot_sgm_fit(ax_PB, ax_PB_log, ax_PB_erfinv, x_lbl + '_{1/2, erf}', clr, \
					OP, PB, lambda x: (scipy.special.erf(x) + 1) / 2, \
					d_PB=d_PB, PB_sgm=PB_erfinv, d_PB_sgm=d_PB_erfinv, \
					linfit_sgm=linfit_erfinv, linfit_sgm_inds=linfit_erfinv_inds, \
					OP0_sgm=OP0_erfinv, d_OP0_sgm=d_OP0_erfinv, \
					N_fine_points=N_fine_points, linestyle='--')
	
	my.add_legend(fig_PB, ax_PB)
	my.add_legend(fig_PB_log, ax_PB_log)
	if(to_plot_sgm):
		my.add_legend(fig_PB_sgm, ax_PB_sgm)
	if(to_plot_erfinv):
		my.add_legend(fig_PB_erfinv, ax_PB_erfinv)

def mark_PB_plot(ax, ax_log, ax_sgm, ax_erfinv, OP, PB, PB_sgm, PB_erfinv, interface_mode):
	ax_log.plot([min(OP), max(OP)], [1/2] * 2, '--', label='$P = 1/2$')
	ax.plot([min(OP), max(OP)], [1/2] * 2, '--', label='$P = 1/2$')
	if(ax_sgm is not None):
		ax_sgm.plot([min(OP), max(OP)], [0] * 2, '--', label='$P = 1/2$')
	if(ax_erfinv is not None):
		ax_erfinv.plot([min(OP), max(OP)], [0] * 2, '--', label='$P = 1/2$')
	
	if(interface_mode == 'M'):
		ax_log.plot([0] * 2, [min(PB), 1], '--', label='$m = 0$')
		ax.plot([0] * 2, [0, 1], '--', label='$m = 0$')
		if(ax_sgm is not None):
			ax_sgm.plot([0] * 2, [min(PB_sgm), max(PB_sgm)], '--', label='$m = 0$')
		if(ax_erfinv is not None):
			ax_erfinv.plot([0] * 2, [min(PB_erfinv), max(PB_erfinv)], '--', label='$m = 0$')

def center_crd(x, L):
	r2 = np.empty(L)
	for i in range(L):
		r2[i] = np.std(np.mod(x + i, L))

	return np.argmin(r2), r2

def draw_state_crds(x_crds, y_crds, to_show=False, ax=None, title=None):
	if(ax is None):
		fig, ax, _ = my.get_fig('x', 'y', title=title)

	ax.scatter(x_crds, y_crds)

	if(to_show):
		plt.show()

def draw_state(state, to_show=False, ax=None, title=None):
	L = state.shape[0]
	assert(state.shape[1] == L)

	if(ax is None):
		fig, ax, _ = my.get_fig('x', 'y', title=title)

	ax.imshow(state)

	if(to_show):
		plt.show()

def center_cluster(inds, L):
	L2 = L*L
	N = len(inds)

	state = np.zeros(L2)
	state[inds] = 1
	state = state.reshape((L, L))
	x_crds = np.where(np.any(state, axis=0))[0]
	y_crds = np.where(np.any(state, axis=1))[0]

	x_shift_to1image, _ = center_crd(x_crds, L)
	y_shift_to1image, _ = center_crd(y_crds, L)

	crds = np.empty((N, 2))

	crds[:, 0] = np.mod(inds % L + x_shift_to1image, L)
	crds[:, 1] = np.mod(inds // L + y_shift_to1image, L)

	x_cm = np.mean(crds[:, 0])
	y_cm = np.mean(crds[:, 1])

	crds[:, 0] = crds[:, 0] - x_cm
	crds[:, 1] = crds[:, 1] - y_cm

	return crds, x_shift_to1image, y_shift_to1image, x_cm, y_cm

def center_state_by_cluster(state, cluster_inds, \
							x_shift_to1image=None, y_shift_to1image=None, \
							x_cm=None, y_cm=None):
	L = state.shape[0]
	assert(state.shape[1] == L)
	L2 = L*L
	N = len(cluster_inds)

	cluster_centered0_crds = None
	if(x_shift_to1image is None):
		cluster_centered0_crds, x_shift_to1image, y_shift_to1image, x_cm, y_cm = \
			center_cluster(cluster_inds, L)

	species_crds = [[]] * N_species
	for i in range(N_species):
		specie_inds = np.where(state.flatten() == i)[0]
		species_crds[i] = np.empty((len(specie_inds), 2))
		species_crds[i][:, 0] = np.mod(specie_inds % L + x_shift_to1image - x_cm + L/2, L)
		species_crds[i][:, 1] = np.mod(specie_inds // L + y_shift_to1image - y_cm + L/2, L)
		#draw_state_crds(species_crds[i][:, 0], species_crds[i][:, 1], to_show=True)

	return species_crds, cluster_centered0_crds

# def keep_new_npz_file(filepath_olds, filepath_new):
	# filepath_olds = [fp for fp in filepath_olds if(os.path.isfile(fp))]
	# old_npzs_present = np.array([os.path.isfile(fp) for fp in filepath_olds])
	# if(np.any(old_npzs_present)):
		# diff_npzs = [filepath_olds[0]]
		# for i, tp in enumerate(filepath_olds[1:]):
			# if(not filecmp.cmp(diff_npzs[0], tp)):
				# diff_npzs.append(tp)
		# N_diff_npzs = len(diff_npzs)

		# #print(filepath_new)
		# #print(diff_npzs)
		# #input('ok2')

		# if(filepath_new in filepath_olds):
			# print('WARNING')
		
		# if(N_diff_npzs == 1):
			# i_npz_new = 0
		# else:
			# print('following npzs all correspond to a single new-style npz name but have different contents:')
			# print('\n'.join([(str(i) + ') ' + fp) for i,fp in enumerate(diff_npzs)]))
			# print('choose the file to keep and write to')
			# print(filepath_new)
			# print('(all other version will be deleted)')
			# i_npz_new = int(input('->'))
			# assert((i_npz_new >= 0) and (i_npz_new < N_diff_npzs))
			# for i, fp in enumerate(filepath_olds):
				# if(old_npzs_present[i] and (fp != diff_npzs[i_npz_new])):
					# os.remove(fp)
			# #for i in range(N_diff_npzs):
			# #	if(i != i_npz_new):
			# #		os.remove(diff_npzs[i])
		
		# my.run_it('mv %s %s' % (diff_npzs[i_npz_new], filepath_new))

	# #input('ok')

	# return filepath_new

def keep_new_npz_file(filepath_olds, filepath_new):
	filepath_olds = [fp for fp in filepath_olds if(os.path.isfile(fp))]
	old_npzs_present = np.array([os.path.isfile(fp) for fp in filepath_olds])
	N_present = np.sum(old_npzs_present)
	if(N_present > 0):
		if(filepath_new in filepath_olds):
			print('WARNING: %s already present, deleting all other corresponding npz-s')
			i_npz_new = filepath_olds.index(filepath_new)
		else:
			if(N_present == 1):
				i_npz_new = 0
			else:
				print('following npzs all correspond to a single new-style npz name:')
				print('\n'.join([(str(i) + ') ' + fp) for i,fp in enumerate(filepath_olds)]))
				print('choose the file to keep and write to')
				print(filepath_new)
				print('(all other version will be deleted)')
				i_npz_new = int(input('->'))
				assert((i_npz_new >= 0) and (i_npz_new < N_present))
		
		for i in range(N_present):
			if(old_npzs_present[i] and (i != i_npz_new)):
				os.remove(filepath_olds[i])
		
		my.run_it('mv %s %s' % (filepath_olds[i_npz_new], filepath_new))

	#input('ok')

	return filepath_new

def proc_order_parameter_FFS(MC_move_mode, L, e, mu, flux0, d_flux0, probs, \
							d_probs, OP_interfaces, N_init_states, \
							interface_mode, states_parent_inds=None, \
							OP_data=None, time_data=None, Nt=None, Nt_OP=None, \
							OP_hist_edges=None, memory_time=1, \
							to_plot_time_evol=False, to_plot_hists=False, stride=1, \
							x_lbl='', y_lbl='', Ms_alpha=0.5, OP_match_BF_to=None, \
							states=None, OP_optim_minstep=None, cluster_map_dx=1.41, \
							to_plot=False, N_fourier=5, to_recomp=0, \
							cluster_map_dr=0.5, npz_basename=None, to_save_npz=True, \
							init_composition=None, to_do_hists=True, \
							Dtop_est_timestride=1, Dtop_PBthr=[0.2,  0.8], \
							Dtop_Nruns=0):
	L2 = L**2
	ln_k_AB = np.log(flux0 * 1) + np.sum(np.log(probs))   # [flux0 * 1] = 1, because [flux] = 1/time = 1/step
	d_ln_k_AB = np.sqrt((d_flux0 / flux0)**2 + np.sum((d_probs / probs)**2))
	Temp = 4 / e[1, 1]
	swap_type_move = (MC_move_mode in [move_modes['swap'], move_modes['long_swap']])
	to_plot_hists = to_plot and to_plot_hists
	to_plot_time_evol = to_plot and to_plot_time_evol
	
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
	#Nc_0 = table_data.Nc_reference_data[str(Temp)] if(str(Temp) in table_data.Nc_reference_data) else None
	P_B_sigmoid, d_P_B_sigmoid, linfit_sigmoid, linfit_sigmoid_inds, OP0_sigmoid = \
		get_sigmoid_fit(P_B[:-1], d_P_B[:-1], OP_interfaces_scaled[:-1], \
						sgminv_fnc=lambda x: -np.log(1/x - 1), \
						d_sgminv_fnc=lambda x, y: 1/(x*(1-x)))
	
	P_B_erfinv, d_P_B_erfinv, linfit_erfinv, linfit_erfinv_inds, OP0_erfinv = \
		get_sigmoid_fit(P_B[:-1], d_P_B[:-1], OP_interfaces_scaled[:-1], \
						sgminv_fnc=lambda x: scipy.special.erfinv(2 * x - 1), \
						d_sgminv_fnc=lambda x, y: np.sqrt(np.pi) * np.exp(y**2))
	ZeldovichG = linfit_erfinv[0] / np.sqrt(np.pi)
	
	OP_closest_to_OP0_ind = np.argmin(np.abs(P_B - 0.5))
	
	# =========== cluster init_states ==============
	OP_init_states = []
	maxclust_ind = []
	max_cluster_1st_ind_id = []
	max_cluster_inds = []
	OP_init_states_OP0exactly_inds = []
	for i in range(N_OP_interfaces):
		OP_init_states.append(np.empty(N_init_states[i], dtype=int))
		maxclust_ind.append(np.empty(N_init_states[i], dtype=int))
		max_cluster_1st_ind_id.append(np.empty(N_init_states[i], dtype=int))
		max_cluster_inds.append([])
		for j in range(N_init_states[i]):
			if(interface_mode == 'M'):
				OP_init_states[i][j] = np.sum(states[i][j, :, :])
			elif(interface_mode == 'CS'):
				(cluster_element_inds, cluster_sizes, cluster_types) = lattice_gas.cluster_state(states[i][j, :, :].flatten())
				
				maxclust_ind[i][j] = np.argmax(cluster_sizes)
				OP_init_states[i][j] = cluster_sizes[maxclust_ind[i][j]]
				max_cluster_1st_ind_id[i][j] = np.sum(cluster_sizes[:maxclust_ind[i][j]])
				max_cluster_inds[i].append(cluster_element_inds[max_cluster_1st_ind_id[i][j] : max_cluster_1st_ind_id[i][j] + cluster_sizes[maxclust_ind[i][j]]])
		
		OP_init_states_OP0exactly_inds.append(np.array([j for j in range(N_init_states[i]) if(OP_init_states[i][j] == OP_interfaces[i])], dtype=int))
	
	if(states_parent_inds is not None):
		# ======= back-track end-states to OP0-states ======
		OP0_children_inds = {}
		OPend_parent_inds = np.empty(N_init_states[-1], dtype=int)
		for i in range(N_init_states[-1]):
			parent_ind = i
			for j in range(N_OP_interfaces - 2, OP_closest_to_OP0_ind - 1, -1):
				parent_ind = states_parent_inds[j][parent_ind]
			
			OPend_parent_inds[i] = parent_ind
			if(parent_ind in OP0_children_inds):
				OP0_children_inds[parent_ind].append(i)
			else:
				OP0_children_inds[parent_ind] = [i]
		OP0_parent_inds = np.array(list(OP0_children_inds.keys()), dtype=int)
		OP0_parent_Nchildren = np.array([len(OP0_children_inds[parent_ind]) for parent_ind in OP0_parent_inds], dtype=int)
		
		OP0_parent_inds_OP0exactly, OP0_parent_inds_OP0exactly_inds, _ = \
			np.intersect1d(OP0_parent_inds, OP_init_states_OP0exactly_inds[OP_closest_to_OP0_ind], return_indices=True)
		if(len(OP0_parent_inds_OP0exactly) == 0):
			print(r'WARNING: no states that {have OP=OPinterface_closest_to_OP0} and {have clihdren at state-B} found at the interface-set-states at OPinterf = %d with OP_interfaces = %s and OP0 = %s' % \
					(OP_interfaces[OP_closest_to_OP0_ind], str(OP_interfaces), my.f2s(OP0_erfinv)))
		else:
			OP0_parent_inds_OP0exactly_Nchildren = OP0_parent_Nchildren[OP0_parent_inds_OP0exactly_inds]
	# =========== estimate Dtop ==============
	if(Dtop_Nruns > 0):
		# ======= estimate D* at the top (P_B = 1/2) ======
		if(states_parent_inds is None):
			print('WARNING: Give Dtop_Nruns = %d > 0 to estimate Dtop, but no states_parent_inds is provided. Will pick OP0 states at random')
		
		P_B_erfinv_interp = scipy.interpolate.interp1d(OP_interfaces_scaled[:-1], P_B_erfinv, fill_value='extrapolate', bounds_error=False)
		#Dtop_OPmin = min(OP_interfaces_scaled[:-1]) if(min(P_B[:-1]) > Dtop_PBthr[0]) else 
		Dtop_OPmin = int(np.floor(scipy.optimize.root_scalar(\
						lambda x: P_B_erfinv_interp(x) - scipy.special.erfinv(2 * Dtop_PBthr[0] - 1), \
						x0=OP_interfaces_scaled[0], x1=OP_interfaces_scaled[1]).root) + 0.1)
		Dtop_OPmax = int(np.ceil(scipy.optimize.root_scalar(\
						lambda x: P_B_erfinv_interp(x) - scipy.special.erfinv(2 * Dtop_PBthr[1] - 1), \
						x0=OP_interfaces_scaled[-2], x1=OP_interfaces_scaled[-3]).root) + 0.1)
		# The interval is [), so the exit will happen is the system goes to (...)U[...)
		
		to_use_random_OPexactly_init_states = (states_parent_inds is None)
		if(not to_use_random_OPexactly_init_states):
			to_use_random_OPexactly_init_states = (len(OP0_parent_inds_OP0exactly) == 0)
		parent_state_IDs = np.random.choice(OP_init_states_OP0exactly_inds[OP_closest_to_OP0_ind], Dtop_Nruns) \
							if(to_use_random_OPexactly_init_states)  else \
							np.random.choice(OP0_parent_inds_OP0exactly, Dtop_Nruns, \
											p = OP0_parent_inds_OP0exactly_Nchildren / np.sum(OP0_parent_inds_OP0exactly_Nchildren))
		parent_state_IDs_unique, parent_state_IDs_lens = np.unique(parent_state_IDs, return_counts=True)
		for i in range(len(parent_state_IDs_unique)):
			# parent_state_ID[i] = np.random.randint(0, N_init_states[OP_closest_to_OP0_ind]) \
								# if(states_parent_inds is None) else \
								# OPend_parent_inds[np.random.randint(0, N_init_states[-1])]
			# choose a state based on the probability to end up in the end state
			
			states_Dtop, _, _, CS_Dtop, _, times_Dtop, _, _ = \
				proc_T(MC_move_mode, L, e, mu, -1, interface_mode, verbose=True, \
					timeevol_stride=Dtop_est_timestride, \
					OP_min=Dtop_OPmin, OP_max=Dtop_OPmax, \
					OP_min_save_state=Dtop_OPmin, OP_max_save_state=Dtop_OPmax, \
					stab_step=None, init_composition=init_composition, \
					to_save_npz=True, to_recomp=to_recomp, \
					to_equilibrate=False, to_post_process=False, \
					to_gen_init_state=False, to_get_timeevol=True, \
					init_state=states[OP_closest_to_OP0_ind][parent_state_IDs_unique[i], :, :].flatten(), \
					N_saved_states_max=parent_state_IDs_lens[i] + 1, \
					to_start_only_state0=1, \
					save_state_mode=3)
			# N_saved_states_max = ... + 1 because the initial state is also saved as state[0, :, :]
			# TODO: make seed work for reproducibility
			# TODO: verbose=verbose-1
			
			states_Dtop = states_Dtop[1:, :, :]
			CS_Dtop = CS_Dtop[1:]
			N_steps = len(CS_Dtop)
			
			restart_timesteps = np.where((CS_Dtop >= Dtop_OPmax) | (CS_Dtop < Dtop_OPmin))[0]
			assert(len(restart_timesteps) == parent_state_IDs_lens[i]), 'ERROR: len(restart_timesteps) = %d that must be == N_saved_states_max = parent_state_IDs_lens[i], but it is not' % (len(restart_timesteps), parent_state_IDs_lens[i])
			#CS_Dtop_shifted = [CS_Dtop[0 : restart_timesteps[0]+1]]
			CS_Dtop_shifted = [np.append(OP_interfaces[OP_closest_to_OP0_ind], CS_Dtop[0 : restart_timesteps[0]+1])]
			for j in range(1, parent_state_IDs_lens[i]):
				#print(restart_timesteps[j-1], restart_timesteps[j])
				#CS_Dtop_shifted.append(CS_Dtop[restart_timesteps[j-1]+1 : restart_timesteps[j]+1])
				CS_Dtop_shifted.append(np.append(OP_interfaces[OP_closest_to_OP0_ind], CS_Dtop[restart_timesteps[j-1]+1 : restart_timesteps[j]+1]))
			
			#print(states_Dtop.shape)
			
			fig, ax, _ = my.get_fig('t', 'CS')
			ax.plot(np.arange(N_steps), CS_Dtop)
			ax.plot([0, N_steps], [Dtop_OPmax]*2, '--')
			ax.plot([0, N_steps], [Dtop_OPmin - 1]*2, '--')
			
			fig2, ax2, _ = my.get_fig('t', 'CS')
			#draw_state(states[OP_closest_to_OP0_ind][parent_state_IDs_unique[i], :, :])
			#print(parent_state_IDs_lens[i])
			for j in range(parent_state_IDs_lens[i]):
				ax2.plot(np.arange(len(CS_Dtop_shifted[j]))+1, CS_Dtop_shifted[j])
				
				#draw_state(states_Dtop[j, :, :])
			
			plt.show()
	
	OP_optimize_f_interp_vals = 1 - np.log(P_B) / np.log(P_B[0])
	d_OP_optimize_f_interp_vals = d_P_B / P_B / np.log(P_B[0])
	#M_optimize_f_interp_fit = lambda OP: (1 - np.log(P_B_opt_fnc(OP)) / np.log(P_B[0]))
	OP_optimize_f_interp_interp1d = scipy.interpolate.interp1d(OP_interfaces_scaled, OP_optimize_f_interp_vals, fill_value='extrapolate')
	#OP_optimize_f_desired = np.linspace(0, 1, N_OP_interfaces)
	OP_optimize_f_desired = np.concatenate((np.linspace(0, OP_optimize_f_interp_interp1d(OP0_erfinv * 1.05), N_OP_interfaces - 1), np.array([1])))
	
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
	if(OP_optim_minstep is None):
		OP_optim_minstep = OP_step[interface_mode]
	
	OP_optimize_new_OP_interfaces = my.refine_interfaces(OP_optimize_new_OP_interfaces, OP_optim_minstep)
	N_new_interfaces = len(OP_optimize_new_OP_interfaces)
	#print('N_M_i =', N_new_interfaces, '\nM_new:\n', OP_optimize_new_OP_interfaces, '\nM_new_original:\n', OP_optimize_new_OP_interfaces_original)
	print('OP_new (using OP_step_min = %f):\n' % OP_optim_minstep, OP_optimize_new_OP_interfaces + (L2 if(interface_mode == 'M') else 0))
	
	if(OP_hist_edges is not None):
		OP_hist_lens = (OP_hist_edges[1:] - OP_hist_edges[:-1])
		OP_hist_centers = (OP_hist_edges[1:] + OP_hist_edges[:-1]) / 2
		N_OP_hist = len(OP_hist_centers)
	else:
		OP_hist_centers = None
		OP_hist_lens = None
	
	if(to_plot):
		ThL_lbl = get_ThL_lbl(e, mu, init_composition, L, swap_type_move)
		
		fig_OP0_Nchildren, ax_OP0_Nchildren, _ = \
				my.get_fig('ID/$N_{N^*-states}$ (sorted)', \
							'$N_{children} / N_{total}$', \
							title='$N_{children}$ distr; ' + ThL_lbl, \
							yscl='log')  # , xscl='log'
		ax_OP0_Nchildren.bar((np.arange(len(OP0_parent_Nchildren))+1) / N_init_states[OP_closest_to_OP0_ind], \
							OP0_parent_Nchildren[np.argsort(-OP0_parent_Nchildren)] / N_init_states[-1], \
							width=0.75/N_init_states[OP_closest_to_OP0_ind])
		
		plot_PB_AB(ThL_lbl, feature_label[interface_mode], title[interface_mode], \
					interface_mode, OP_interfaces_scaled[:-1], P_B[:-1], d_PB=d_P_B[:-1], \
					PB_sgm=P_B_sigmoid, d_PB_sgm=d_P_B_sigmoid, \
					linfit_sgm=linfit_sigmoid, linfit_sgm_inds=linfit_sigmoid_inds, \
					OP0_sgm=OP0_sigmoid, \
					PB_erfinv=P_B_erfinv, d_PB_erfinv=d_P_B_erfinv, \
					linfit_erfinv=linfit_erfinv, linfit_erfinv_inds=linfit_erfinv_inds, \
					OP0_erfinv=OP0_erfinv, \
					clr=my.get_my_color(1))
	
	if(to_do_hists):
		state_Rdens_centers, state_centered_Rdens_total, d_state_centered_Rdens = \
			tuple([None] * 3)
		#input(str(states is not None))
		if(states is not None):
			npz_states_filepath = os.path.join(npz_basename + '_FFSab_states.npz')
			
			if((not os.path.isfile(npz_states_filepath)) or to_recomp):
				# ============ process ===========
				OP_init_states = []
				OP_init_hist = np.empty((N_OP_hist, N_OP_interfaces))
				d_OP_init_hist = np.empty((N_OP_hist, N_OP_interfaces))
				cluster_centered_crds = []
				state_centered_crds = []
				cluster_centered_crds_per_interface = [[]] * N_OP_interfaces
				state_centered_crds_per_interface = [[]] * N_OP_interfaces
				for i in range(N_OP_interfaces):
					OP_init_states.append(np.empty(N_init_states[i], dtype=int))
					cluster_centered_crds.append([[]] * N_init_states[i])
					state_centered_crds.append([[]] * N_init_states[i])
					for j in range(N_init_states[i]):
						if(interface_mode == 'M'):
							OP_init_states[i][j] = np.sum(states[i][j, :, :])
						elif(interface_mode == 'CS'):
							#(cluster_element_inds, cluster_sizes, cluster_types) = lattice_gas.cluster_state(states[i][j, :, :].flatten())
							#OP_init_states[i][j] = max(cluster_sizes)
							#max_cluster_id = np.argmax(cluster_sizes)
							#OP_init_states[i][j] = cluster_sizes[OP_init_states_maxclust_inds[i][j]]
							
							#max_cluster_1st_ind_id = np.sum(cluster_sizes[:OP_init_states_maxclust_inds[i][j]])
							#max_cluster_inds = cluster_element_inds[max_cluster_1st_ind_id : max_cluster_1st_ind_id + cluster_sizes[OP_init_states_maxclust_inds[i][j]]]
							
							#cluster_centered_crds[i][j], _, _, _, _ = \
							#	center_cluster(max_cluster_inds, L)
							
							# if(j == 0):
								# draw_state(states[i][j, :, :], to_show=True, ax=None, title='CS = ' + str(OP_interfaces[i]))
							
							state_centered_crds[i][j], cluster_centered_crds[i][j] = \
								center_state_by_cluster(states[i][j, :, :], max_cluster_inds[i][j])
					
					#cluster_centered_crds_per_interface_local = tuple([cluster_centered_crds[i][j] for j in range(N_init_states[i]) if(OP_init_states[i][j] == OP_interfaces[i])])
					cluster_centered_crds_per_interface_local = tuple([cluster_centered_crds[i][j] for j in OP_init_states_OP0exactly_inds[i]])
					OP_init_states_OP0exactly_inds
					#if(len(cluster_centered_crds_per_interface_local) < len(cluster_centered_crds[i]) * 0.5):
					if(len(cluster_centered_crds_per_interface_local) < 2):
						print('WARNING: Interface OP[%d] = %d has too few (%d) states with OP = %d.\nOP-s are: %s\nInterfaces are:\n%s' % \
								(i, OP_interfaces[i], len(cluster_centered_crds_per_interface_local), OP_interfaces[i], str(OP_init_states[i]), str(OP_interfaces)))
						print('Using all the interfaces (and not only interfaces at OP = %d) for constructing clusters' % OP_interfaces[i])
						#input('Proceed? [press Enter or Ctrl+C] ->')
						cluster_centered_crds_per_interface_local = tuple(cluster_centered_crds[i])
					
					cluster_centered_crds_per_interface[i] = np.concatenate(cluster_centered_crds_per_interface_local, axis=0)
					state_centered_crds_per_interface[i] = [[]] * N_species
					for k in range(N_species):
						state_centered_crds_per_interface[i][k] = \
							np.concatenate(tuple([state_centered_crds[i][j][k] for j in range(N_init_states[i])]), axis=0)
					
					assert(OP_hist_edges is not None), 'ERROR: no "OP_hist_edges" provided but "OP_init_states" is asked to be analyzed (_states != None)'
					OP_init_hist[:, i], _ = np.histogram(OP_init_states[i], bins=OP_hist_edges * OP_scale[interface_mode])
					OP_init_hist[:, i] = OP_init_hist[:, i] / N_init_states[i]
					d_OP_init_hist[:, i] = np.sqrt(OP_init_hist[:, i] * (1 - OP_init_hist[:, i]) / N_init_states[i])
				
				cluster_centered_crds_all = np.concatenate(tuple(cluster_centered_crds_per_interface), axis=0)
				def stepped_linspace(x, dx, margin=1e-3):
					mn = min(x) - dx * margin
					mx = max(x) + dx * margin
					return np.linspace(mn, mx, int((mx - mn) / dx + 0.5))
				
				# ==== cluster ====
				cluster_map_edges = [stepped_linspace(cluster_centered_crds_all[:, 0], cluster_map_dx), \
									 stepped_linspace(cluster_centered_crds_all[:, 1], cluster_map_dx)]
				cluster_Rdens_edges = stepped_linspace(np.linalg.norm(cluster_centered_crds_all, axis=1), cluster_map_dr)
				cluster_map_centers = [(cluster_map_edges[0][1:] + cluster_map_edges[0][:-1]) / 2, \
										(cluster_map_edges[1][1:] + cluster_map_edges[1][:-1]) / 2]
				cluster_Rdens_centers = (cluster_Rdens_edges[1:] + cluster_Rdens_edges[:-1]) / 2
				
				# ==== state ====
				state_map_edges = [stepped_linspace([-L/2, L/2], cluster_map_dx), \
									 stepped_linspace([-L/2, L/2], cluster_map_dx)]
				state_Rdens_edges = stepped_linspace([0, L/2], cluster_map_dr)
				state_map_centers = [(state_map_edges[0][1:] + state_map_edges[0][:-1]) / 2, \
										(state_map_edges[1][1:] + state_map_edges[1][:-1]) / 2]
				state_Rdens_centers = (state_Rdens_edges[1:] + state_Rdens_edges[:-1]) / 2
				
				cluster_centered_map_total = [[]] * N_OP_interfaces
				cluster_centered_maps = [[]] * N_OP_interfaces
				cluster_centered_Rdens_total = [[]] * N_OP_interfaces
				cluster_centered_Rdens_s = [[]] * N_OP_interfaces
				d_cluster_centered_map_total = [[]] * N_OP_interfaces
				d_cluster_centered_Rdens_total = [[]] * N_OP_interfaces
				state_centered_map_total = [[]] * N_OP_interfaces
				state_centered_Rdens_total = [[]] * N_OP_interfaces
				state_centered_maps = [[]] * N_OP_interfaces
				state_centered_Rdens_s = [[]] * N_OP_interfaces
				d_state_centered_map = [[]] * N_OP_interfaces
				d_state_centered_Rdens = [[]] * N_OP_interfaces
				rho_fourier2D = np.empty((N_OP_interfaces, 2, N_fourier))
				for i in range(N_OP_interfaces):
					# ==== cluster ====
					Rs_local = np.linalg.norm(cluster_centered_crds_per_interface[i], axis=1)
					
					cluster_centered_map_total[i], _, _ = \
						np.histogram2d(cluster_centered_crds_per_interface[i][:, 0], \
										cluster_centered_crds_per_interface[i][:, 1], \
										bins=cluster_map_edges)
					cluster_centered_Rdens_total[i], _ = \
						np.histogram(Rs_local, \
									bins=cluster_Rdens_edges)
					
					rho_fourier2D_local = np.empty((2, N_init_states[i]))
					for k in range(N_fourier):
						for j in range(N_init_states[i]):
							theta_local = (k + 1) * np.arctan2(cluster_centered_crds[i][j][:, 1], cluster_centered_crds[i][j][:, 0])
							rho_fourier2D_local[0, j] = np.mean(np.cos(theta_local))
							rho_fourier2D_local[1, j] = np.mean(np.sin(theta_local))
						rho_fourier2D[i, :, k] = np.mean(np.abs(rho_fourier2D_local), axis=1)
					
					cluster_centered_maps[i] = np.empty((cluster_centered_map_total[i].shape[0], cluster_centered_map_total[i].shape[1], N_init_states[i]))
					cluster_centered_Rdens_s[i] = np.empty((cluster_centered_Rdens_total[i].shape[0], N_init_states[i]))
					for j in range(N_init_states[i]):
						cluster_centered_maps[i][:, :, j], _, _ = \
							np.histogram2d(cluster_centered_crds[i][j][:, 0], \
										cluster_centered_crds[i][j][:, 1], \
										bins=cluster_map_edges)
						
						cluster_centered_Rdens_s[i][:, j], _ = \
							np.histogram(np.linalg.norm(cluster_centered_crds[i][j], axis=1), \
										bins=cluster_Rdens_edges)
					
					d_cluster_centered_map_total[i] = np.std(cluster_centered_maps[i], axis=2)
					d_cluster_centered_Rdens_total[i] = np.std(cluster_centered_Rdens_s[i], axis=1)
					
					cluster_centered_map_norm = N_init_states[i] * cluster_map_dx**2
					cluster_centered_map_total[i] = cluster_centered_map_total[i] / cluster_centered_map_norm
					d_cluster_centered_map_total[i] = d_cluster_centered_map_total[i] / cluster_centered_map_norm * np.sqrt(N_init_states[i])
					
					cluster_centered_Rdens_norm = N_init_states[i] * (np.pi * (cluster_Rdens_edges[1:]**2 - cluster_Rdens_edges[:-1]**2))
					cluster_centered_Rdens_total[i] = cluster_centered_Rdens_total[i] / cluster_centered_Rdens_norm
					d_cluster_centered_Rdens_total[i] = d_cluster_centered_Rdens_total[i] / cluster_centered_Rdens_norm * np.sqrt(N_init_states[i])
					
					# ==== state ====
					state_centered_map_total[i] = [[]] * N_species
					state_centered_Rdens_total[i] = [[]] * N_species
					state_centered_maps[i] = [[]] * N_species
					state_centered_Rdens_s[i] = [[]] * N_species
					d_state_centered_map[i] = [[]] * N_species
					d_state_centered_Rdens[i] = [[]] * N_species
					for k in range(N_species):
						state_centered_map_total[i][k], _, _ = \
							np.histogram2d(state_centered_crds_per_interface[i][k][:, 0] - L/2, \
											 state_centered_crds_per_interface[i][k][:, 1] - L/2, \
											 bins=state_map_edges)
						
						state_centered_Rdens_total[i][k], _ = \
							np.histogram(np.linalg.norm(state_centered_crds_per_interface[i][k] - L/2, axis=1), \
										 bins=state_Rdens_edges)
						
						state_centered_maps[i][k] = np.empty((state_centered_map_total[i][k].shape[0], state_centered_map_total[i][k].shape[1], N_init_states[i]))
						state_centered_Rdens_s[i][k] = np.empty((state_centered_Rdens_total[i][k].shape[0], N_init_states[i]))
						for j in range(N_init_states[i]):
							state_centered_maps[i][k][:, :, j], _, _ = \
								np.histogram2d(state_centered_crds[i][j][k][:, 0], \
											state_centered_crds[i][j][k][:, 1], \
											bins=state_map_edges)
							
							state_centered_Rdens_s[i][k][:, j], _ = \
								np.histogram(np.linalg.norm(state_centered_crds[i][j][k], axis=1), \
											bins=state_Rdens_edges)
						
						#print(state_centered_Rdens_s[i][k].shape)
						d_state_centered_map[i][k] = np.std(state_centered_maps[i][k], axis=2)
						d_state_centered_Rdens[i][k] = np.std(state_centered_Rdens_s[i][k], axis=1)
						#print(np.std(state_centered_Rdens_s[i][k], axis=1).shape)
						
						state_centered_map_norm = N_init_states[i] * cluster_map_dx**2
						state_centered_map_total[i][k] = state_centered_map_total[i][k] / state_centered_map_norm
						d_state_centered_map[i][k] = d_state_centered_map[i][k] / state_centered_map_norm * np.sqrt(N_init_states[i])
						
						state_centered_Rdens_norm = N_init_states[i] * (np.pi * (state_Rdens_edges[1:]**2 - state_Rdens_edges[:-1]**2))
						state_centered_Rdens_total[i][k] = state_centered_Rdens_total[i][k] / state_centered_Rdens_norm
						d_state_centered_Rdens[i][k] = d_state_centered_Rdens[i][k] / state_centered_Rdens_norm * np.sqrt(N_init_states[i])
				
				if(to_save_npz):
					print('writing', npz_states_filepath)
					np.savez(npz_states_filepath, \
							OP_hist_centers=OP_hist_centers,\
							OP_init_hist=OP_init_hist, \
							d_OP_init_hist=d_OP_init_hist, \
							OP_hist_lens=OP_hist_lens, \
							cluster_Rdens_centers=cluster_Rdens_centers, \
							cluster_centered_Rdens_total=cluster_centered_Rdens_total, \
							d_cluster_centered_Rdens_total=d_cluster_centered_Rdens_total, \
							cluster_centered_map_total=cluster_centered_map_total, \
							cluster_map_centers=cluster_map_centers, \
							state_Rdens_centers=state_Rdens_centers, \
							state_centered_Rdens_total=state_centered_Rdens_total, \
							d_state_centered_Rdens=d_state_centered_Rdens, \
							rho_fourier2D=rho_fourier2D)
			
			else:
				print(npz_states_filepath, 'loading')
				npz_data = np.load(npz_states_filepath, allow_pickle=True)
				
				OP_hist_centers = npz_data['OP_hist_centers']
				OP_init_hist = npz_data['OP_init_hist']
				d_OP_init_hist = npz_data['d_OP_init_hist']
				OP_hist_lens = npz_data['OP_hist_lens']
				cluster_Rdens_centers = npz_data['cluster_Rdens_centers']
				cluster_centered_Rdens_total = npz_data['cluster_centered_Rdens_total']
				d_cluster_centered_Rdens_total = npz_data['d_cluster_centered_Rdens_total']
				cluster_centered_map_total = npz_data['cluster_centered_map_total']
				cluster_map_centers = npz_data['cluster_map_centers']
				state_Rdens_centers = npz_data['state_Rdens_centers']
				state_centered_Rdens_total = npz_data['state_centered_Rdens_total']
				d_state_centered_Rdens = npz_data['d_state_centered_Rdens']
				rho_fourier2D = npz_data['rho_fourier2D']
			
			if(to_plot_hists):
				# ============ plot ===========
				ThL_lbl = get_ThL_lbl(e, mu, init_composition, L, swap_type_move)
				
				# ==== cluster ====
				fig_OPinit_hists, ax_OPinit_hists, _ = my.get_fig(y_lbl, r'$p_i(' + x_lbl + ')$', title=r'$p(' + x_lbl + ')$; ' + ThL_lbl, yscl='log')
				fig_cluster_Rdens, ax_cluster_Rdens, _ = my.get_fig('r', r'$\rho$', title=r'$\rho(r)$; ' + ThL_lbl)
				fig_cluster_Rdens_log, ax_cluster_Rdens_log, _ = my.get_fig('r', r'$\rho$', title=r'$\rho(r)$; ' + ThL_lbl, yscl='logit')
				fig_cluster_map = [[]] * N_OP_interfaces
				ax_cluster_map = [[]] * N_OP_interfaces
				
				# ==== state ====
				fig_state_map = [[]] * N_species
				ax_state_map = [[]] * N_species
				fig_state_Rdens = [[]] * N_species
				ax_state_Rdens = [[]] * N_species
				fig_state_Rdens_log = [[]] * N_species
				ax_state_Rdens_log = [[]] * N_species
				for k in range(N_species):
					fig_state_Rdens[k], ax_state_Rdens[k], _ = my.get_fig('r', r'$\rho_%d$' % k, title=(r'$\rho_%d(r)$; ' % k) + ThL_lbl)
					fig_state_Rdens_log[k], ax_state_Rdens_log[k], _ = my.get_fig('r', r'$\rho_%d$' % k, title=(r'$\rho_%d(r)$; ' % k) + ThL_lbl, yscl='logit')
					fig_state_map[k] = [[]] * N_OP_interfaces
					ax_state_map[k] = [[]] * N_OP_interfaces
				
				for i in range(N_OP_interfaces):
					ax_OPinit_hists.bar(OP_hist_centers, OP_init_hist[:, i], yerr=d_OP_init_hist[:, i], width=OP_hist_lens, align='center', label='OP = ' + str(OP_interfaces[i]), alpha=Ms_alpha)
					
					# ==== cluster ====
					cluster_lbl = r'$OP[%d] = %d$; ' % (i, OP_interfaces[i])
					ax_cluster_Rdens.errorbar(cluster_Rdens_centers, cluster_centered_Rdens_total[i], yerr=d_cluster_centered_Rdens_total[i], label=cluster_lbl)
					ax_cluster_Rdens_log.errorbar(cluster_Rdens_centers, cluster_centered_Rdens_total[i], yerr=d_cluster_centered_Rdens_total[i], label=cluster_lbl)
					
					fig_cluster_map[i], ax_cluster_map[i], fig_id = my.get_fig('x', 'y', title=cluster_lbl + ThL_lbl)
					im = ax_cluster_map[i].imshow(cluster_centered_map_total[i], \
											extent = [min(cluster_map_centers[1]), max(cluster_map_centers[1]), \
													  min(cluster_map_centers[0]), max(cluster_map_centers[0])], \
											interpolation ='bilinear', origin ='lower', aspect='auto')
					plt.figure(fig_id)
					cbar = plt.colorbar(im)
					
					# ==== state ====
					for k in range(N_species):
						ax_state_Rdens[k].errorbar(state_Rdens_centers, state_centered_Rdens_total[i][k], yerr=d_state_centered_Rdens[i][k], label=cluster_lbl)
						ax_state_Rdens_log[k].errorbar(state_Rdens_centers, state_centered_Rdens_total[i][k], yerr=d_state_centered_Rdens[i][k], label=cluster_lbl)
				
				my.add_legend(fig_OPinit_hists, ax_OPinit_hists)
				my.add_legend(fig_cluster_Rdens, ax_cluster_Rdens)
				my.add_legend(fig_cluster_Rdens_log, ax_cluster_Rdens_log)
				for k in range(N_species):
					my.add_legend(fig_state_Rdens[k], ax_state_Rdens[k])
					my.add_legend(fig_state_Rdens_log[k], ax_state_Rdens_log[k])
	
	rho = None
	d_rho = None
	
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
		times = [time_data[ : Nt_totals[0]]]
		
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
			times.append(time_data[Nt_totals[i-1] : Nt_totals[i]])
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
			#print(OP[i].shape)
			OP_hist, _ = np.histogram(OP[i], bins=OP_hist_edges, weights=times[i])
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
		F[OP_hist_ok_inds_FFS] = -np.log(rho[OP_hist_ok_inds_FFS] * OP_hist_lens[OP_hist_ok_inds_FFS])
		d_F[OP_hist_ok_inds_FFS] = d_rho[OP_hist_ok_inds_FFS] / rho[OP_hist_ok_inds_FFS]
		
		BF_inds = OP_hist_centers <= OP_match_BF_to / OP_scale[interface_mode]
		# print(BF_inds)
		# print(OP_hist_lens[BF_inds])
		# print(rho_s[BF_inds, 0])
		# input(str(OP_hist_centers))
		F_BF = -np.log(rho_s[BF_inds, 0] * OP_hist_lens[BF_inds])
		d_F_BF = d_rho_s[BF_inds, 0] / rho_s[BF_inds, 0]
		
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
		rho[state_A_FFS_inds] = np.exp(-F_BF[state_A_BF_inds]) / OP_hist_lens[state_A_FFS_inds]
		d_rho[state_A_FFS_inds] = rho[state_A_FFS_inds] * d_F_BF[state_A_BF_inds]
		
		F = -np.log(rho * OP_hist_lens)
		#F_BF = F_BF - min(F)
		#F = F - min(F)
		F_BF = F_BF - F[0]
		F = F - F[0]
		d_F = d_rho / rho
		
		ThL_lbl = get_ThL_lbl(e, mu, init_composition, L, swap_type_move)
		
		if(to_plot_time_evol):
			fig_OP, ax_OP, _ = my.get_fig('step', y_lbl, title=x_lbl + '(step); ' + ThL_lbl)
			
			for i in range(N_OP_interfaces):
				ax_OP.plot(timesteps[i][::stride], OP[i][::stride], label='data, i=%d' % (i))
			
			my.add_legend(fig_OP, ax_OP)
		
		if(to_plot_hists):
			fig_OPhists, ax_OPhists, _ = my.get_fig(y_lbl, r'$\rho_i(' + x_lbl + ') \cdot P(i | A)$', title=r'$\rho(' + x_lbl + ')$; ' + ThL_lbl, yscl='log')
			for i in range(N_OP_interfaces - 1):
				ax_OPhists.bar(OP_hist_centers, rho_for_sum[:, i], yerr=d_rho_for_sum[:, i], width=OP_hist_lens, align='center', label=str(i), alpha=Ms_alpha)
			
			for i in range(N_OP_interfaces):
				ax_OPhists.plot([OP_interfaces_scaled[i]] * 2, [min(rho), max(rho)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			my.add_legend(fig_OPhists, ax_OPhists)
			
			fig_OPjumps_hists, ax_OPjumps_hists, _ = my.get_fig('d' + x_lbl, 'probability', title=x_lbl + ' jumps distribution; ' + ThL_lbl, yscl='log')
			for i in range(N_OP_interfaces):
				ax_OPjumps_hists.plot(OP_possible_jumps[interface_mode], OP_jumps_hists[:, i], label='$' + x_lbl + ' \in (' + str((OP_min_default[interface_mode]) if(i == 0) else OP_interfaces_scaled[i-1]) + ';' + str(OP_interfaces_scaled[i]) + ']$')
			my.add_legend(fig_OPjumps_hists, ax_OPjumps_hists)
			
			fig_F, ax_F, _ = my.get_fig(y_lbl, r'$F(' + x_lbl + r') = -T \ln(\rho(%s) \cdot d%s(%s))$' % (x_lbl, x_lbl, x_lbl), title=r'$F(' + x_lbl + ')$; ' + ThL_lbl)
			ax_F.errorbar(OP_hist_centers[BF_inds], F_BF, yerr=d_F_BF, fmt='.', label='$F_{BF}$')
			ax_F.errorbar(OP_hist_centers, F, yerr=d_F, fmt='.', label='$F_{total}$')
			for i in range(N_OP_interfaces):
				ax_F.plot([OP_interfaces_scaled[i]] * 2, [min(F), max(F)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			my.add_legend(fig_F, ax_F)
	
	if(to_plot_hists):
		plot_PB_AB(ThL_lbl, x_lbl, y_lbl, \
					interface_mode, OP_interfaces_scaled[:-1], P_B[:-1], d_PB=d_P_B[:-1], \
					PB_sgm=P_B_sigmoid, d_PB_sgm=d_P_B_sigmoid, \
					linfit_sgm=linfit_sigmoid, linfit_sgm_inds=linfit_sigmoid_inds, \
					OP0_sgm=OP0_sigmoid, \
					PB_erfinv=P_B_erfinv, d_PB_erfinv=d_P_B_erfinv, \
					linfit_erfinv=linfit_erfinv, linfit_erfinv_inds=linfit_erfinv_inds, \
					OP0_erfinv=OP0_erfinv, \
					clr=my.get_my_color(1))
		
		fig_fl, ax_fl, _ = my.get_fig(y_lbl, r'$1 - \ln[P_B(' + x_lbl + ')] / \ln[P_B(0)] = f(\lambda_i)$', title=r'$f(\lambda_i)$; ' + ThL_lbl)
		fig_fl_i, ax_fl_i, _ = my.get_fig(r'$index(' + x_lbl + '_i)$', r'$1 - \ln[P_B(' + x_lbl + '_i)] / \ln[P_B(0)] = f(\lambda_i)$', title=r'$f(\lambda_i)$; ' + ThL_lbl)
		
		m_fine = np.linspace(min(OP_interfaces_scaled), max(OP_interfaces_scaled), int((max(OP_interfaces_scaled) - min(OP_interfaces_scaled)) / min(OP_interfaces_scaled[1:] - OP_interfaces_scaled[:-1]) * 10))
		ax_fl.errorbar(OP_interfaces_scaled, OP_optimize_f_interp_vals, yerr=d_OP_optimize_f_interp_vals, fmt='.', label=r'$f(\lambda)_i$')
		ax_fl_i.errorbar(np.arange(N_OP_interfaces), OP_optimize_f_interp_vals, yerr=d_OP_optimize_f_interp_vals, fmt='.', label=r'$f(\lambda)_i$')
		#ax_fl.plot(m_fine[:-1], M_optimize_f_interp_fit(m_fine[:-1]), label=r'$f_{fit}(\lambda)$')
		ax_fl.plot(m_fine, OP_optimize_f_interp_interp1d(m_fine), label=r'$f_{interp}(\lambda)$')
		for i in range(N_new_interfaces):
			ax_fl.plot([OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode]] * 2, [0, OP_optimize_f_interp_interp1d(OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode])], '--', label=('new interfaces' if(i == 0) else None), color=my.get_my_color(3))
			ax_fl.plot([min(OP_interfaces_scaled), OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode]], [OP_optimize_f_interp_interp1d(OP_optimize_new_OP_interfaces[i] / OP_scale[interface_mode])] * 2, '--', label=None, color=my.get_my_color(3))
		#ax_fl.plot([min(OP_interfaces_scaled), max(OP_interfaces_scaled)], [0, 1], '--', label=r'$interpolation$')
		ax_fl_i.plot([0, N_OP_interfaces - 1], [0, 1], '--', label=r'$ideal$')
		
		my.add_legend(fig_fl, ax_fl)
		my.add_legend(fig_fl_i, ax_fl_i)
	
	return ln_k_AB, d_ln_k_AB, P_B, d_P_B, \
			P_B_sigmoid, d_P_B_sigmoid, linfit_sigmoid, linfit_sigmoid_inds, OP0_sigmoid, \
			P_B_erfinv, d_P_B_erfinv, linfit_erfinv, linfit_erfinv_inds, OP0_erfinv, ZeldovichG, \
			rho, d_rho, OP_hist_centers, OP_hist_lens, \
			state_Rdens_centers, state_centered_Rdens_total, d_state_centered_Rdens, \
			cluster_centered_map_total, cluster_map_centers, rho_fourier2D, \
			ax_OP, ax_OPhists, ax_F, ax_fl, ax_fl_i

def proc_FFS_AB(MC_move_mode, L, e, mu, N_init_states, OP_interfaces, interface_mode, \
				stab_step=-5, verbose=None, to_get_timeevol=False, \
				OP_sample_BF_to=None, OP_match_BF_to=None,
				to_plot_time_evol=False, to_plot_hists=False, to_do_hists=True, \
				to_plot=False, timeevol_stride=-3000, N_fourier=5, \
				init_gen_mode=-2, Ms_alpha=0.5, OP_optim_minstep=8, \
				init_composition=None, to_recomp=0, to_save_npz=True, \
				to_post_proc=True, Dtop_Nruns=0):
	
	swap_type_move = MC_move_mode in [move_modes["swap"], move_modes["long_swap"]]
	
	#traj_basepath = os.path.join(my.git_root_path(), 'izing_npzs')
	traj_basepath = '/scratch/gpfs/yp1065/Izing/npzs'
	#traj_basepath = '/scratch/gpfs/yp1065/Izing/nvt'
	
	traj_basename, traj_basename_olds = \
		izing.get_FFS_AB_npzTrajBasename(MC_move_mode, L, e, \
							('_phi' + my.join_lbls(init_composition[1:], '_')) \
								if(swap_type_move) \
								else ('_muT' + my.join_lbls(mu[1:], '_')), \
							OP_interfaces, N_init_states, \
							stab_step, OP_sample_BF_to, OP_match_BF_to, \
							timeevol_stride if(to_get_timeevol) else None, \
							init_gen_mode, bool(to_get_timeevol), \
							N_fourier, lattice_gas.get_seed())
	
	FFStraj_suff = '_FFStraj.npz'
	traj_filepath_olds = [os.path.join(traj_basepath, tb + FFStraj_suff) for tb in traj_basename_olds]
	traj_filepath_new = os.path.join(traj_basepath, traj_basename + FFStraj_suff)
	
	traj_filepath = keep_new_npz_file(traj_filepath_olds, traj_filepath_new)
	
	#print(traj_filepath, os.path.isfile(traj_filepath))
	#print(to_recomp, N_init_states)
	#input('ok')
	
	if((not os.path.isfile(traj_filepath)) or (to_recomp > 1)):
		init_state = get_fixed_composition_random_state(L, init_composition, N_main_min_warning=50) \
						if(swap_type_move) else None
		
		# py::tuple run_FFS(int move_mode, int L, py::array_t<double> e, py::array_t<double> mu,
		#			  pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces,
		#			  int to_remember_timeevol, int init_gen_mode, int interface_mode,
		#			  std::optional<int> _verbose)
		# return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M, biggest_cluster_sizes, time);
		(_states, _states_parent_inds, probs, d_probs, Nt, Nt_OP, flux0, d_flux0, _E, _M, _CS, times) = \
			lattice_gas.run_FFS(MC_move_mode, L, e.flatten(), mu, N_init_states, \
								OP_interfaces, stab_step=stab_step, \
								verbose=verbose, \
								to_remember_timeevol=to_get_timeevol, \
								init_gen_mode=init_gen_mode, \
								interface_mode=OP_C_id[interface_mode], \
								init_state=None if(init_state is None) else init_state.flatten())
		
		if(to_save_npz):
			np.savez(traj_filepath, states=_states, probs=probs, \
					states_parent_inds=_states_parent_inds, 
					d_probs=d_probs, Nt=Nt, Nt_OP=Nt_OP, \
					flux0=flux0, d_flux0=d_flux0, \
					E=_E, M=_M, CS=_CS, times=times)
			
			print(traj_filepath, 'written')
	else:
		print('loading', traj_filepath)
		npz_data = np.load(traj_filepath, allow_pickle=True)
		
		_states = npz_data['states']
		_states_parent_inds = (npz_data['states_parent_inds'] if('states_parent_inds' in npz_data.files) else None)
		probs = npz_data['probs']
		d_probs = npz_data['d_probs']
		Nt = npz_data['Nt']
		Nt_OP = npz_data['Nt_OP']
		flux0 = npz_data['flux0']
		d_flux0 = npz_data['d_flux0']
		_E = npz_data['E']
		_M = npz_data['M']
		_CS = npz_data['CS']
		times = npz_data['times']
	
	if(to_post_proc):
		L2 = L**2
		N_OP_interfaces = len(OP_interfaces)
		if(to_get_timeevol):
			Nt_total = max(len(_M), len(_CS))
			if(timeevol_stride < 0):
				timeevol_stride = np.int_(- Nt_total / timeevol_stride)
		
		states = [_states[ : N_init_states[0] * L2].reshape((N_init_states[0], L, L))]
		for i in range(1, N_OP_interfaces):
			states.append(_states[np.sum(N_init_states[:i]) * L2 : np.sum(N_init_states[:(i+1)]) * L2].reshape((N_init_states[i], L, L)))
		del _states
		
		if(_states_parent_inds is None):
			print('WARNING: _states_parent_inds is not present in', traj_filepath)
			states_parent_inds = None
		else:
			states_parent_inds = [_states_parent_inds[ : N_init_states[1]]]
			for i in range(2, N_OP_interfaces):
				states_parent_inds.append(_states_parent_inds[np.sum(N_init_states[1:i]) : np.sum(N_init_states[1:(i+1)])])
			del _states_parent_inds
		
		data = {}
		data['M'] = _M
		data['CS'] = _CS
		
		ln_k_AB, d_ln_k_AB, P_B, d_P_B, \
			P_B_sigmoid, d_P_B_sigmoid, linfit_sigmoid, linfit_sigmoid_inds, OP0_sigmoid, \
			P_B_erfinv, d_P_B_erfinv, linfit_erfinv, linfit_erfinv_inds, OP0_erfinv, ZeldovichG, \
			rho, d_rho, OP_hist_centers, OP_hist_lens, \
			state_Rdens_centers, state_centered_Rdens_total, d_state_centered_Rdens, \
			cluster_centered_map_total, cluster_map_centers, rho_fourier2D, \
			ax_OP, ax_OPhists, ax_F, ax_fl, ax_fl_i = \
				proc_order_parameter_FFS(MC_move_mode, L, e, mu, flux0, d_flux0, probs, d_probs, OP_interfaces, N_init_states, \
							interface_mode, states_parent_inds=states_parent_inds, OP_match_BF_to=OP_match_BF_to, \
							OP_data=(data[interface_mode] / OP_scale[interface_mode] if(to_get_timeevol) else None), \
							time_data = times, Nt_OP=Nt_OP, Nt=Nt, N_fourier=N_fourier, to_plot=to_plot, \
							OP_hist_edges=get_OP_hist_edges(interface_mode, OP_max=OP_interfaces[-1]), \
							memory_time=max(1, OP_std_overestimate[interface_mode] / OP_step[interface_mode]),
							to_plot_time_evol=to_plot_time_evol, to_plot_hists=to_plot_hists, stride=timeevol_stride, \
							x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], Ms_alpha=Ms_alpha, \
							states=states, OP_optim_minstep=OP_optim_minstep, Dtop_Nruns=Dtop_Nruns, \
							npz_basename=os.path.join(traj_basepath, traj_basename), \
							to_save_npz=to_save_npz, to_recomp=to_recomp, \
							init_composition=init_composition, to_do_hists=to_do_hists)
		
		return probs, d_probs, ln_k_AB, d_ln_k_AB, flux0, d_flux0, rho, d_rho, \
				OP_hist_centers, OP_hist_lens, P_B, d_P_B, \
				P_B_sigmoid, d_P_B_sigmoid, linfit_sigmoid, linfit_sigmoid_inds, OP0_sigmoid, \
				P_B_erfinv, d_P_B_erfinv, linfit_erfinv, linfit_erfinv_inds, OP0_erfinv, ZeldovichG, \
				state_Rdens_centers, state_centered_Rdens_total, d_state_centered_Rdens, \
				cluster_centered_map_total, cluster_map_centers, rho_fourier2D

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

def proc_order_parameter_BF(MC_move_mode, L, e, mu, states, m, E, stab_step, \
							dOP_step, OP_hist_edges, OP_peak_guess, OP_OP_fit2_width, \
							x_lbl=None, y_lbl=None, verbose=None, to_estimate_k=False, \
							hA=None, OP_A=None, OP_B=None, to_save_npz=False, \
							to_plot_time_evol=True, to_plot_F=True, to_plot_ETS=False, \
							stride=1, OP_jumps_hist_edges=None, init_composition=None, \
							possible_jumps=None, means_only=False, to_recomp=0, \
							npz_basename=None, to_animate=False):

	k_AB, d_k_AB, k_BA, d_k_BA, k_bc_AB, k_bc_BA, ax_OP, ax_OP_hist, ax_F, \
		F, d_F, OP_hist_centers, OP_hist_lens, rho_interp1d, d_rho_interp1d, \
		OP_mean, OP_std, d_OP_mean, E_avg, S, phi_Lmeans = \
		tuple([None] * 21)

	L2 = L**2
	Nt_OP = len(m)
	steps = np.arange(Nt_OP) * stride
	stab_ind = (steps > stab_step)
	Nt_stab = np.sum(stab_ind)
	assert(Nt_stab > 1), 'ERROR: the system may not have reached equlibrium, stab_step = %d, max-present-step = %d' % (stab_step, max(steps))
	Nt_states = states.shape[0]
	swap_type_move = (MC_move_mode in [move_modes['swap'], move_modes['long_swap']])

	# npz_basepath = os.path.join(my.git_root_path(), 'izing_npzs')
	# npz_basename = 'L' + str(L) + '_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + \
					# (('_muT' + my.join_lbls(mu[1:], '_')) if(init_composition is None) \
						# else ('_phi' + my.join_lbls(init_composition[1:], '_'))) + \
					# '_stride' + str(stride) + \
					# '_OP' + str(OP_A) + '_' + str(OP_B) + '_stab' + str(stab_step) + \
					# '_Nstates' + str(Nt_states) + '_ID' + str(lattice_gas.get_seed())

	states_timesteps = np.arange(Nt_states) * stride
	stab_states_inds = (states_timesteps > stab_step)
	N_stab_states = np.sum(stab_states_inds)

	phi_filepath = os.path.join(npz_basename + '_phi.npz')
	if((not os.path.isfile(phi_filepath)) or to_recomp):
		phi_Lmeans = np.zeros((N_species, Nt_states))
		for i in range(N_species - 1):
			#phi_Lmeans[i, :] = np.mean((states == i).reshape((Nt_states, L, L)), axis=(1,2))
			phi_Lmeans[i, :] = np.mean(states == i, axis=(1,2))
		phi_Lmeans[N_species - 1, :] = 1 - np.sum(phi_Lmeans, axis=0)
		if(to_save_npz):
			print('writing', phi_filepath)
			np.savez(phi_filepath, states_timesteps=states_timesteps, phi_Lmeans=phi_Lmeans, stab_step=stab_step)
	else:
		print('loading', phi_filepath)
		npz_data = np.load(phi_filepath, allow_pickle=True)

		states_timesteps = npz_data['states_timesteps']
		phi_Lmeans = npz_data['phi_Lmeans']
		stab_step = npz_data['stab_step']

	# metastable_inds = (~stab_states_inds) & (states_timesteps > 5 * L2)
	# N_metastable_states = np.sum(metastable_inds)
	# phi_Lmeans_stab = phi_Lmeans[:, metastable_inds].reshape((N_species, N_metastable_states))
	# phi_LTmeans = np.mean(phi_Lmeans_stab, axis=1)
	# d_phi_LTmeans = (np.std(phi_Lmeans_stab, axis=1) * np.sqrt(N_metastable_states / (N_metastable_states-1)))# / np.sqrt(N_stab_states)
	phi_Lmeans_stab = phi_Lmeans[:, stab_states_inds].reshape((N_species, N_stab_states))
	phi_LTmeans = np.mean(phi_Lmeans_stab, axis=1)
	d_phi_LTmeans = np.std(phi_Lmeans_stab, axis=1)

	ThL_lbl = get_ThL_lbl(e, mu, init_composition, L, swap_type_move)

	if(to_animate):
		my.animate_2D(states, 'x', 'y', 's(x, y)', yx_lims = [0, L-1, 0, L-1], fps=100)
		# for it in range(Nt_states):
			# draw_state(states[it, :, :], to_show=True)

	if(to_plot_time_evol):
		fig_phi, ax_phi, _ = my.get_fig('step', r'$\varphi$', title=r'$\vec{\varphi}$(step); ' + ThL_lbl)

		for i in range(N_species):
			#print()

			ax_phi.plot(states_timesteps, phi_Lmeans[i, :], \
					label=(r'$<\varphi_%d> = %s$' % (i, my.errorbar_str(phi_LTmeans[i], d_phi_LTmeans[i]))))
			ax_phi.plot([stab_step, max(states_timesteps)], [phi_LTmeans[i]] * 2, '--', \
					label=None, color=my.get_my_color(i))
		ax_phi.plot([stab_step] * 2, [0, 1], '--', label='equilibr')

		my.add_legend(fig_phi, ax_phi)

	if(not means_only):
		F_filepath = os.path.join(npz_basename + '_F.npz')

		if((not os.path.isfile(F_filepath)) or to_recomp):
			OP_stab = m[stab_ind]

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

			F = -np.log(rho * OP_hist_lens)
			F = F - F[0]
			d_F = d_rho / rho

			E_avg = np.empty(N_m_points)
			for i in range(N_m_points):
				E_for_avg = E[(OP_hist_edges[i] < m) & (m < OP_hist_edges[i+1])] * L2
				#E_avg[i] = np.average(E_for_avg, weights=np.exp(- E_for_avg))
				E_avg[i] = np.mean(E_for_avg) if(len(E_for_avg) > 0) else (e[1,1] + abs(mu[1])) * L2
			E_avg = E_avg - E_avg[0]   # we can choose the E constant
			S = (E_avg - F)
			S = S - S[0] # S[m=+-1] = 0 because S = log(N), and N(m=+-1)=1

			if(to_save_npz):
				print('writing', F_filepath)
				np.savez(F_filepath, OP_hist_centers=OP_hist_centers, \
						rho=rho, d_rho=d_rho, OP_hist_lens=OP_hist_lens, \
						possible_jumps=possible_jumps, \
						OP_jumps_hist=OP_jumps_hist, F=F, d_F=d_F, \
						E_avg=E_avg, S=S, \
						rho_interp1d=rho_interp1d,
						d_rho_interp1d=d_rho_interp1d, \
						OP_mean=OP_mean, OP_std=OP_std, d_OP_mean=d_OP_mean)
		else:
			print('loading', F_filepath)
			npz_data = np.load(F_filepath, allow_pickle=True)

			OP_hist_centers = npz_data['OP_hist_centers']
			rho = npz_data['rho']
			d_rho = npz_data['d_rho']
			OP_hist_lens = npz_data['OP_hist_lens']
			possible_jumps = npz_data['possible_jumps']
			OP_jumps_hist = npz_data['OP_jumps_hist']
			F = npz_data['F']
			d_F = npz_data['d_F']
			E_avg = npz_data['E_avg']
			S = npz_data['S']
			rho_interp1d = npz_data['rho_interp1d'].item()
			d_rho_interp1d = npz_data['d_rho_interp1d'].item()
			OP_mean = npz_data['OP_mean']
			OP_std = npz_data['OP_std']
			d_OP_mean = npz_data['d_OP_mean']

		if(to_estimate_k):
			k_filepath = os.path.join(npz_basename + '_kAB.npz')
			if((not os.path.isfile(k_filepath)) or to_recomp):
				assert(hA is not None), 'ERROR: k estimation is requested in BF, but no hA provided'
				#k_AB_ratio = np.sum(hA > 0.5) / np.sum(hA < 0.5)
				hA_jump = hA[1:] - hA[:-1]   # jump > 0 => {0->1} => from B to A => k_BA ==> k_ratio = k_BA / k_AB
				jump_inds = np.where(hA_jump != 0)[0]

				N_AB_jumps = np.sum(hA_jump < 0)
				if(N_AB_jumps > 0):
					k_AB = N_AB_jumps / np.sum(hA > 0.5)
					d_k_AB = k_AB / np.sqrt(N_AB_jumps)
				else:
					k_AB, d_k_AB = 0, 0

				N_BA_jumps = np.sum(hA_jump > 0)
				if(N_BA_jumps > 0):
					k_BA = N_BA_jumps / np.sum(hA < 0.5)
					d_k_BA = k_BA / np.sqrt(N_BA_jumps)
				else:
					k_BA, d_k_BA = 0, 0

				if(to_save_npz):
					print('writing', k_filepath)
					np.savez(k_filepath, hA=hA, hA_jump=hA_jump, \
							jump_inds=jump_inds, N_AB_jumps=N_AB_jumps, \
							k_AB=k_AB, d_k_AB=d_k_AB, \
							k_BA=k_BA, d_k_BA=d_k_BA)
			else:
				print('loading', k_filepath)
				npz_data = np.load(k_filepath, allow_pickle=True)

				hA = npz_data['hA']
				hA_jump = npz_data['hA_jump']
				jump_inds = npz_data['jump_inds']
				N_AB_jumps = npz_data['N_AB_jumps']
				k_AB = npz_data['k_AB']
				d_k_AB = npz_data['d_k_AB']
				k_BA = npz_data['k_BA']
				d_k_BA = npz_data['d_k_BA']

			OP_fit2_OPmin_ind = np.argmax(OP_hist_centers > OP_peak_guess - OP_OP_fit2_width)
			OP_fit2_OPmax_ind = np.argmax(OP_hist_centers > OP_peak_guess + OP_OP_fit2_width)
			assert(OP_fit2_OPmin_ind > 0), 'issues finding analytical border'
			assert(OP_fit2_OPmax_ind > 0), 'issues finding analytical border'
			OP_fit2_inds = (OP_fit2_OPmin_ind <= np.arange(len(OP_hist_centers))) & (OP_fit2_OPmax_ind >= np.arange(len(OP_hist_centers))) & (F < max(F))
			N_good_points_near_OPpeak = np.sum(OP_fit2_inds)
			if(N_good_points_near_OPpeak > 2):
				OP_fit2 = np.polyfit(OP_hist_centers[OP_fit2_inds], F[OP_fit2_inds], 2, w = 1/d_F[OP_fit2_inds])
				OP_peak = -OP_fit2[1] / (2 * OP_fit2[0])
				m_peak_ind = np.argmin(abs(OP_hist_centers - OP_peak))
				F_max = np.polyval(OP_fit2, OP_peak)
				#m_peak_ind = np.argmin(OP_hist_centers - OP_peak)
				bc_Z_AB = exp_integrate(OP_hist_centers[ : OP_fit2_OPmin_ind + 1], -F[ : OP_fit2_OPmin_ind + 1]) + exp2_integrate(- OP_fit2, OP_hist_centers[OP_fit2_OPmin_ind])
				bc_Z_BA = exp_integrate(OP_hist_centers[OP_fit2_OPmax_ind : ], -F[OP_fit2_OPmax_ind : ]) + abs(exp2_integrate(- OP_fit2, OP_hist_centers[OP_fit2_OPmax_ind]))
				k_bc_AB = (np.mean(abs(OP_hist_centers[m_peak_ind + 1] - OP_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max) / bc_Z_AB)
				k_bc_BA = (np.mean(abs(OP_hist_centers[m_peak_ind + 1] - OP_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max) / bc_Z_BA)
			else:
				print('WARNING: too few (%d) points with good data found near guessed OP_peak (%d)' % (N_good_points_near_OPpeak, OP_peak_guess))
				k_bc_AB, k_bc_BA = 0, 0

				# OP_fit2, OP_peak, m_peak_ind, F_max, bc_Z_AB, bc_Z_BA, \
				# k_bc_AB, k_bc_BA = \
					# tuple([np.nan] * 8)

			if(k_AB > 0):
				get_log_errors(np.log(k_AB), d_k_AB / k_AB, lbl='k_AB_BF', print_scale=print_scale_k)
			else:
				print('k_AB = 0')

			if(k_BA > 0):
				get_log_errors(np.log(k_BA), d_k_BA / k_BA, lbl='k_BA_BF', print_scale=print_scale_k)
			else:
				print('k_BA = 0')

		if(to_plot_time_evol):
			fig_OP, ax_OP, _ = my.get_fig('step', y_lbl, title='$' + x_lbl + '$(step); ' + ThL_lbl)
			ax_OP.plot(steps, m, label='data')
			ax_OP.plot([stab_step] * 2, [min(m), max(m)], '--', label='equilibr')
			ax_OP.plot([stab_step, max(steps)], [OP_mean] * 2, '--', label=('$<' + x_lbl + '> = ' + my.errorbar_str(OP_mean, d_OP_mean) + '$'))
			if(to_estimate_k):
				ax_OP.plot([0, max(steps)], [OP_A] * 2, '--', label='state A, B', color=my.get_my_color(1))
				ax_OP.plot([0, max(steps)], [OP_B] * 2, '--', label=None, color=my.get_my_color(1))
			my.add_legend(fig_OP, ax_OP)

			if(to_estimate_k):
				fig_hA, ax_hA, _ = my.get_fig('step', '$h_A$', title='$h_{A, ' + x_lbl + '}$(step); ' + ThL_lbl)
				ax_hA.plot(steps, hA, label='$h_A$')
				ax_hA.plot(steps[jump_inds+1], hA_jump[jump_inds], '.', label='$h_A$ jumps')
				my.add_legend(fig_hA, ax_hA)

			#print('k_' + x_lbl + '_AB    =    (' + my.errorbar_str(k_AB, d_k_AB) + ')    (1/step);    dk/k =', my.f2s(d_k_AB / k_AB))
			#print('k_' + x_lbl + '_BA    =    (' + my.errorbar_str(k_BA, d_k_BA) + ')    (1/step);    dk/k =', my.f2s(d_k_BA / k_BA))

		if(to_plot_F):
			fig_OP_hist, ax_OP_hist, _ = my.get_fig(y_lbl, r'$\rho(' + x_lbl + ')$', title=r'$\rho(' + x_lbl + ')$; ' + ThL_lbl)
			ax_OP_hist.bar(OP_hist_centers, rho, yerr=d_rho, width=OP_hist_lens, label=r'$\rho(' + x_lbl + ')$', align='center')
			if(to_estimate_k):
				ax_OP_hist.plot([OP_A] * 2, [0, max(rho)], '--', label='state A,B', color=my.get_my_color(1))
				ax_OP_hist.plot([OP_B] * 2, [0, max(rho)], '--', label=None, color=my.get_my_color(1))
			my.add_legend(fig_OP_hist, ax_OP_hist)

			fig_OP_jumps_hist, ax_OP_jumps_hist, _ = my.get_fig('d' + x_lbl, 'probability', title=x_lbl + ' jumps distribution; ' + ThL_lbl, yscl='log')
			ax_OP_jumps_hist.plot(possible_jumps, OP_jumps_hist, '.')

			fig_F, ax_F, _ = my.get_fig(y_lbl, r'$F/T$', title=r'$F(' + x_lbl + ')/T$; ' + ThL_lbl)
			ax_F.errorbar(OP_hist_centers, F, yerr=d_F, fmt='.', label='$F(' + x_lbl + ')/T$')
			if(to_estimate_k):
				if(N_good_points_near_OPpeak > 2):
					ax_F.plot([OP_A] * 2, [min(F), max(F)], '--', label='state A, B', color=my.get_my_color(1))
					ax_F.plot([OP_B] * 2, [min(F), max(F)], '--', label=None, color=my.get_my_color(1))
					ax_F.plot(OP_hist_centers[OP_fit2_inds], np.polyval(OP_fit2, OP_hist_centers[OP_fit2_inds]), label='fit2; $' + x_lbl + '_0 = ' + my.f2s(OP_peak) + '$')

			if(to_plot_ETS):
				ax_F.plot(OP_hist_centers, E_avg, '.', label='<E>(' + x_lbl + ')/T')
				ax_F.plot(OP_hist_centers, S, '.', label='$T \cdot S(' + x_lbl + ')$')
			my.add_legend(fig_F, ax_F)

			if(to_estimate_k):
				print('k_' + x_lbl + '_bc_AB   =   ' + my.f2s(k_bc_AB * print_scale_k) + '   (%e/step);' % (1/print_scale_k))
				print('k_' + x_lbl + '_bc_BA   =   ' + my.f2s(k_bc_BA * print_scale_k) + '   (%e/step);' % (1/print_scale_k))

	return F, d_F, OP_hist_centers, OP_hist_lens, rho_interp1d, d_rho_interp1d, \
			k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, \
			OP_mean, OP_std, d_OP_mean, \
			E_avg, S, phi_Lmeans, phi_LTmeans, d_phi_LTmeans, \
			ax_OP, ax_OP_hist, ax_F, \
			fig_OP, fig_OP_hist, fig_F

def plot_correlation(x, y, x_lbl, y_lbl):
	fig, ax, _ = my.get_fig(x_lbl, y_lbl)
	R = scipy.stats.pearsonr(x, y)[0]
	ax.plot(x, y, '.', label='R = ' + my.f2s(R))
	my.add_legend(fig, ax)

	return ax, R

def get_fixed_composition_random_state(L, composition, N_main_min_warning=0):
	L2 = L**2
	N_main_component = int(L2 * composition[1])
	N_minor_component = int(L2 * composition[2])
	#N_background = L2 - N_main_component - N_minor_component
	if(N_main_min_warning > 0):
		if(N_main_min_warning > N_main_component):
			print('WARNING:   N_main = %d < %d' % (N_main_component, N_main_min_warning))

	#init_state = np.zeros((L, L), dtype=int)   # background = 0
	#main_inds = np.unique(np.minimum(np.arange(N_main_component) * (L2 // (N_main_component - 1)), L2 - 1))
	#minor_inds = np.minimum(np.arange(N_minor_component) * (L2 // (N_minor_component - 1)), L2 - 2) + 1
	#minor_inds[np.isin(minor_inds, main_inds)] += 1
	#minor_inds = np.unique(minor_inds % L2)

	init_state = np.zeros(L2, dtype=int)   # background = 0
	init_state[ : N_main_component] = 1
	init_state[N_main_component : (N_main_component + N_minor_component)] = 2

	np.random.shuffle(init_state)

	assert(np.sum(init_state == 1) == N_main_component)
	assert(np.sum(init_state == 2) == N_minor_component)

	return init_state.reshape((L, L))

def proc_T(MC_move_mode, L, e, mu, Nt, interface_mode, verbose=None, \
			N_spins_up_init=None, to_get_timeevol=True, \
			to_plot_timeevol=False, to_plot_F=False, to_plot_ETS=False, \
			to_plot_correlations=False, N_saved_states_max=-1, \
			to_estimate_k=False, timeevol_stride=-3000, \
			OP_min=None, OP_max=None, OP_A=None, OP_B=None, \
			OP_min_save_state=None, OP_max_save_state=None, \
			means_only=False, stab_step=-10, init_composition=None, \
			to_save_npz=True, to_recomp=0, R_clust_init=None, \
			to_animate=False, to_equilibrate=None, to_post_process=True, \
			init_state=None, to_gen_init_state=None, stab_step_BF_run=-1, \
			save_state_mode=1, to_start_only_state0=0):
	'''
		save_state_mode
			1 == lattice_gas.save_state_mode_Inside
			2 == lattice_gas.save_state_mode_Influx
	'''
	L2 = L**2
	swap_type_move = MC_move_mode in [move_modes["swap"], move_modes["long_swap"]]
	if(verbose is None):
		verbose = lattice_gas.get_verbose()
	
	if(to_get_timeevol):
		if(stab_step is not None):
			if(stab_step < 0):
				stab_step = int(min(L2, Nt / 2) * (-stab_step))
			# Each spin has a good chance to flip during this time
			# This does not guarantee the global stable state, but it is sufficient to be sure-enough we are in the optimum of the local metastable-state.
			# The approximation for a global ebulibration would be '1/k_AB', where k_AB is the rate constant. But it's hard to estimate on that stage.
	
	if(OP_min is None):
		OP_min = OP_min_default[interface_mode]
	if(OP_max is None):
		OP_max = OP_max_default[interface_mode]
	
	if(N_spins_up_init is None):
		if(interface_mode == 'M'):
			N_spins_up_init = OP_min + 1
		elif(interface_mode == 'CS'):
			N_spins_up_init = OP_min + 1
	if(interface_mode == 'M'):
		OP_init = N_spins_up_init
	elif(interface_mode == 'CS'):
		OP_init = N_spins_up_init
	
	if(timeevol_stride < 0):
		timeevol_stride = np.int_(- Nt / timeevol_stride)
	
	if(to_gen_init_state is None):
		to_gen_init_state = MC_move_mode in [move_modes["swap"], move_modes["long_swap"]]
	if((init_state is not None) and to_gen_init_state):
		print('WARNING: to_get_init_state = True and init_state provided. Given init state will be overwritten')
		input('Proceed? (any-key or Ctrl+C)')
	if(to_equilibrate is None):
		to_equilibrate = (R_clust_init is None)
	# ==========
	#traj_basepath = os.path.join(my.git_root_path(), 'izing_npzs')
	traj_basepath = '/scratch/gpfs/yp1065/Izing/npzs'
	traj_basename, traj_basename_olds = \
		izing.get_BF_npzTrajBasename(MC_move_mode, L, e, \
						('_phi' + my.join_lbls(init_composition[1:], '_')) \
								if(swap_type_move) \
								else ('_muT' + my.join_lbls(mu[1:], '_')), \
						Nt, N_saved_states_max, R_clust_init, stab_step, \
						OP_A, OP_B, OP_min, OP_max, \
						OP_min_save_state, OP_max_save_state, \
						timeevol_stride, to_equilibrate, to_get_timeevol, \
						lattice_gas.get_seed())
			
	
	BFtraj_suff = '_BFtraj.npz'
	traj_filepath_olds = [os.path.join(traj_basepath, tb + BFtraj_suff) for tb in traj_basename_olds]
	traj_filepath_new = os.path.join(traj_basepath, traj_basename + BFtraj_suff)
	
	traj_filepath = keep_new_npz_file(traj_filepath_olds, traj_filepath_new)
	# ==========
	# traj_basename = 'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
					# '_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + \
					# (('_phi' + my.join_lbls(init_composition[1:], '_')) if(to_gen_init_state) \
						# else ('_muT' + my.join_lbls(mu[1:], '_'))) + \
					# '_Nt' + str(Nt) + \
					# '_NstatesMax' + str(N_saved_states_max) + \
					# ('_stride' + str(timeevol_stride) if(to_get_timeevol) else '') + \
					# '_OPab' + str(OP_A) + '_' + str(OP_B) + \
					# '_OPminx' + str(OP_min) + '_' + str(OP_max) + \
					# '_OPminxSave' + str(OP_min_save_state) + '_' + str(OP_max_save_state) + \
					# '_timeData' + str(to_get_timeevol) + \
					# '_Rinit' + str(R_clust_init) + \
					# '_ID' + str(lattice_gas.get_seed())
	# traj_filepath = os.path.join(traj_basepath, traj_basename + '_traj.npz')
	
	if((not os.path.isfile(traj_filepath)) or to_recomp):
		
		if(to_gen_init_state):
			init_state = get_fixed_composition_random_state(L, init_composition)
		if(R_clust_init is not None):
			y_crds, x_crds = tuple(np.mgrid[0:L, 0:L])
			init_clust_inds = (x_crds - (L-1)/2)**2 + (y_crds - (L-1)/2)**2 < R_clust_init**2
			init_state[y_crds[init_clust_inds], x_crds[init_clust_inds]] = 1
			
			#draw_state(init_state, to_show=True)
		
		'''
			py::tuple run_bruteforce(int move_mode, int L, py::array_t<double> e, py::array_t<double> mu, long Nt_max,
						 long N_saved_states_max, long save_states_stride, long stab_step,
						 std::optional<int> _N_spins_up_init, std::optional<int> _to_remember_timeevol,
						 std::optional<int> _OP_A, std::optional<int> _OP_B,
						 std::optional<int> _OP_min_save_state, std::optional<int> _OP_max_save_state,
						 std::optional<int> _OP_min, std::optional<int> _OP_max,
						 std::optional<int> _interface_mode, int save_state_mode,
						 std::optional< pybind11::array_t<int> > _init_state,
						 int to_use_smart_swap, int to_equilibrate,
						 std::optional<int> _verbose)
		'''
		(states, E, M, CS, hA, times, k_AB_launches, time_total) = \
			lattice_gas.run_bruteforce(MC_move_mode, L, e.flatten(), mu, Nt,
				N_saved_states_max=N_saved_states_max, \
				save_states_stride=timeevol_stride, \
				N_spins_up_init=N_spins_up_init, \
				OP_A=OP_A, OP_B=OP_B, \
				OP_min=OP_min, OP_max=OP_max, \
				OP_min_save_state=OP_min_save_state, \
				OP_max_save_state=OP_max_save_state, \
				to_remember_timeevol=to_get_timeevol, \
				interface_mode=OP_C_id[interface_mode], \
				init_state=None if(init_state is None) else init_state.flatten(), \
				to_equilibrate=to_equilibrate, \
				save_state_mode=save_state_mode, \
				to_start_only_state0=to_start_only_state0, \
				verbose=verbose)
		
		if(to_save_npz):
			print('writing', traj_filepath)
			np.savez(traj_filepath, states=states, \
					E=E, M=M, CS=CS, hA=hA, times=times, \
					k_AB_launches=k_AB_launches, time_total=time_total)
	else:
		print('loading', traj_filepath)
		npz_data = np.load(traj_filepath, allow_pickle=True)
		
		states = npz_data['states']
		E = npz_data['E']
		M = npz_data['M']
		CS = npz_data['CS']
		hA = npz_data['hA']
		times = npz_data['times']
		k_AB_launches = npz_data['k_AB_launches']
		time_total = npz_data['time_total']
	
	Nt_states = len(states) // L2
	states = states.reshape((Nt_states, L, L))
	
	k_AB_BFcount = k_AB_launches / time_total
	d_k_AB_BFcount = (k_AB_BFcount / np.sqrt(k_AB_launches)) if(k_AB_BFcount > 0) else 1
	if(verbose > 0):
		print('k_AB_BFcount = %s' % (my.errorbar_str(k_AB_BFcount, d_k_AB_BFcount) if(k_AB_BFcount > 0) else '0'))
	
	if(to_post_process):
		F = None
		d_F = None
		hist_centers = None
		hist_lens = None
		rho_interp1d = None
		d_rho_interp1d = None
		k_bc_AB = None
		k_bc_BA = None
		k_AB = None
		d_k_AB = None
		k_BA = None
		d_k_BA = None
		OP_avg = None
		d_OP_avg = None
		E_avg = None
		if(to_get_timeevol):
			data = {}
			hist_edges = {}
			OP_jumps_hist_edges = np.concatenate(([OP_possible_jumps[interface_mode][0] - OP_step[interface_mode] / 2], \
											(OP_possible_jumps[interface_mode][1:] + OP_possible_jumps[interface_mode][:-1]) / 2, \
											[OP_possible_jumps[interface_mode][-1] + OP_step[interface_mode] / 2]))
			
			data['M'] = M
			hist_edges['M'] = get_OP_hist_edges(interface_mode, OP_min if(interface_mode == 'M') else None, OP_max if(interface_mode == 'M') else None)
			del M
			
			data['CS'] = CS
			hist_edges['CS'] = get_OP_hist_edges(interface_mode, OP_min if(interface_mode == 'CS') else None, OP_max if(interface_mode == 'CS') else None)
			
			N_features = len(hist_edges.keys())
			
			F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, \
				k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_avg, \
					OP_std, d_OP_avg, E_avg, S_E, \
					phi_Lmeans, phi_LTmeans, d_phi_LTmeans, \
					ax_time, ax_hist, ax_F, \
					fig_time, fig_hist, fig_F = \
						proc_order_parameter_BF(MC_move_mode, L, e, mu, states, data[interface_mode] / OP_scale[interface_mode], E / L2, stab_step, \
										dOP_step[interface_mode], hist_edges[interface_mode], int((OP_A + OP_B) / 2 + 0.5), OP_fit2_width[interface_mode], \
										x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], verbose=verbose, to_estimate_k=to_estimate_k, hA=hA, \
										to_plot_time_evol=to_plot_timeevol, to_plot_F=to_plot_F, to_plot_ETS=to_plot_ETS, stride=timeevol_stride, \
										OP_A=int(OP_A / OP_scale[interface_mode] + 0.1), OP_B=int(OP_B / OP_scale[interface_mode] + 0.1), OP_jumps_hist_edges=OP_jumps_hist_edges, \
										possible_jumps=OP_possible_jumps[interface_mode], means_only=means_only, to_save_npz=to_save_npz, \
										init_composition=init_composition, to_recomp=to_recomp, npz_basename=os.path.join(traj_basepath, traj_basename), \
										to_animate=to_animate)
			
			if(not means_only):
				BC_cluster_size = L2 / np.pi
				#C = ((OP_std['E'] * L2)**2) / L2   # heat capacity per spin
				
				if(to_plot_F):
					F_limits = [min(F), max(F)]
					hist_limits = [min(rho_interp1d(hist_centers)), max(rho_interp1d(hist_centers))]
					ax_F.plot([OP_init / OP_scale[interface_mode]] * 2, F_limits, '--', label='init state')
					ax_hist.plot([OP_init / OP_scale[interface_mode]] * 2, hist_limits, '--', label='init state')
					if(interface_mode == 'CS'):
						ax_hist.plot([BC_cluster_size] * 2, hist_limits, '--', label='$L_{BC} = L^2 / \pi$')
						my.add_legend(fig_hist, ax_hist)
						ax_F.plot([BC_cluster_size] * 2, F_limits, '--', label='$S_{BC} = L^2 / \pi$')
					my.add_legend(fig_F, ax_F)
				
				# if(to_plot_correlations):
					# feature_labels_ordered = list(hist_edges.keys())
					# for i1 in range(N_features):
						# for i2 in range(i1 + 1, N_features):
							# k1 = feature_labels_ordered[i1]
							# k2 = feature_labels_ordered[i2]
							# plot_correlation(data[k1][::timeevol_stride] / OP_scale[k1], data[k2][::timeevol_stride] / OP_scale[k2], \
											# feature_label[k1], feature_label[k2])
		
		return F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, \
					k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_avg, d_OP_avg, \
					phi_Lmeans, phi_LTmeans, d_phi_LTmeans, \
					E_avg, k_AB_BFcount, d_k_AB_BFcount, k_AB_launches#, C
	else:
		return states, E, M, CS, hA, times, k_AB_launches, time_total

def run_phi01_pair(MC_move_mode, L, e, mu, Nt, interface_mode, timeevol_stride, \
					stab_step, to_plot_timeevol=False, to_save_npz=False, \
					init_composition_0=None, init_composition_1=None, \
					to_recomp=0, verbose=1):
	L2 = L**2

	restricted_MC_modes = [move_modes["swap"], move_modes["long_swap"]]
	assert(MC_move_mode not in restricted_MC_modes), 'ERROR: MC_move_mode = %d in restricted modes %s for phi01_pair run' % (MC_move_mode, str(restricted_MC_modes))

	_, _, _, _, _, _, \
		_, _, _, _, _, _, _, _, \
		phi_0_evol, phi_0, d_phi_0, \
		_, _, _, _, = \
			proc_T(MC_move_mode, L, e, mu, Nt, interface_mode, \
				OP_A=-1, OP_B=L2+1, N_spins_up_init=0, \
				to_get_timeevol=True, to_plot_timeevol=to_plot_timeevol, \
				OP_min=-1, OP_max=L2 + 1, stab_step=stab_step, \
				timeevol_stride=timeevol_stride, N_saved_states_max=0, \
				means_only=True, init_composition=init_composition_0, \
				to_save_npz=to_save_npz, to_recomp=to_recomp, \
				verbose=verbose)   # phi_0 is a state-vector enriched in phi0

	_, _, _, _, _, _, \
		_, _, _, _, _, _, _, _, \
		phi_1_evol, phi_1, d_phi_1, \
		_, _, _, _, = \
			proc_T(MC_move_mode, L, e, mu, Nt, interface_mode, \
				OP_A=-1, OP_B=L2+1, N_spins_up_init=L2, \
				to_get_timeevol=True, to_plot_timeevol=to_plot_timeevol, \
				OP_min=-1, OP_max=L2 + 1, stab_step=stab_step, \
				timeevol_stride=timeevol_stride, N_saved_states_max=0, \
				means_only=True, init_composition=init_composition_1, \
				to_save_npz=to_save_npz, to_recomp=to_recomp, \
				verbose=verbose)

	return phi_0, d_phi_0, phi_1, d_phi_1, phi_0_evol, phi_1_evol

def map_phase_difference(MC_move_mode, L, e, mu_vecs, Nt, interface_mode, \
						to_plot_timeevol=False, timeevol_stride=-3000, \
						to_plot_debug=False, stab_step=-10, \
						init_composition_0=None, init_composition_1=None):
	L2 = L**2
	#mu_vecs = [np.linspace(mu_bounds[i, 0], mu_bounds[i, 1], mu_Ns[i]) for i in range(N_species - 1)]

	mu_Ns = [len(mu_v) for mu_v in mu_vecs]
	N_total = np.prod(mu_Ns)

	phi_0 = np.empty((3, N_total))
	d_phi_0 = np.empty((3, N_total))
	phi_1 = np.empty((3, N_total))
	d_phi_1 = np.empty((3, N_total))
	dist = np.empty(N_total)
	mu_points = np.empty((2, N_total))
	for i1 in range(mu_Ns[0]):
		for i2 in range(mu_Ns[1]):
			i = i2 + i1 * mu_Ns[1]
			mu_points[:, i] = [mu_vecs[0][i1], mu_vecs[1][i2]]
			mu_full = np.array([0] + list(mu_points[:, i]))

			phi_0[:, i], d_phi_0[:, i], \
				phi_1[:, i], d_phi_1[:, i], _, _ = \
					run_phi01_pair(MC_move_mode, L, e, mu_full, Nt, \
								interface_mode, timeevol_stride, stab_step, \
								to_plot_timeevol=to_plot_debug)
	#dist = np.sqrt(np.sum((phi_0 - phi_1)**2 / (d_phi_0**2 + d_phi_1**2), axis=0))
	dist = np.sqrt(np.sum((phi_0 - phi_1)**2, axis=0))

	resampled_mu1_grid, resampled_mu2_grid = \
		np.mgrid[min(mu_vecs[0]):max(mu_vecs[0]):int((max(mu_vecs[0]) - min(mu_vecs[0])) / min(abs(mu_vecs[0][1:] - mu_vecs[0][:-1])))*5*1j, \
				min(mu_vecs[1]):max(mu_vecs[1]):int((max(mu_vecs[1]) - min(mu_vecs[1])) / min(abs(mu_vecs[1][1:] - mu_vecs[1][:-1])))*5*1j]

	resampled_dist = \
		scipy.interpolate.griddata(np.array([mu_points[0, :], mu_points[1, :]]).T, \
									dist, \
									(resampled_mu1_grid, resampled_mu2_grid), \
									method='linear')

	ThL_lbl = r'$\epsilon_{11,12,22}/T = ' + my.join_lbls([e[1,1], e[1,2], e[2,2]], ',')
	fig_map, ax_map, fig_map_id = my.get_fig(r'$\mu_1/T$', r'$\mu_2/T$', title=r'$|\vec{\phi}_0 - \vec{\phi}_1| (\mu_1, \mu_2)$; $' + ThL_lbl + '$')
	im_map = ax_map.imshow(resampled_dist.T, \
						extent = [min(mu_vecs[0]), max(mu_vecs[0]), \
									min(mu_vecs[1]), max(mu_vecs[1])], \
						interpolation ='bilinear', origin ='lower', aspect='auto')
	plt.figure(fig_map_id)
	cbar = plt.colorbar(im_map)


def phi_to_x(p, d_p):
	x = p[2] / p[1]
	d_x = x * np.sqrt(np.sum((d_p[1:] / p[1:])**2))
	return x, d_x

def cost_fnc(MC_move_mode, L, e, mu, Nt, interface_mode, phi0_target, phi1_target, \
			timeevol_stride, N_runs, stab_step, verbose=0, stab_step_FFS=-5, \
			to_plot_timeevol=False, to_plot_debug=False, cost_mode=0, seeds=None):
	L2 = L**2
	if(len(mu) == 2):
		mu_full = np.zeros(N_species)
		mu_full[1:] = mu
	else:
		mu_full = np.copy(mu)

	if(len(e.shape) == 1):
		e_full = eFull_from_eCut(e)
	else:
		e_full = np.copy(e)

	# phi_0, d_phi_0, phi_1, d_phi_1 = \
		# run_phi01_pair(MC_move_mode, L, e_full, mu_full, Nt, interface_mode, \
					# timeevol_stride, stab_step, \
					# to_plot_timeevol=to_plot_timeevol)
	phi_0, d_phi_0, phi_1, d_phi_1, \
		phi_0_evol, d_phi_0_evol, phi_1_evol, d_phi_1_evol = \
			run_many(MC_move_mode, L, e_full, mu_full, N_runs, interface_mode, \
				Nt_per_BF_run=Nt, stab_step=stab_step_FFS, \
				mode='BF_2sides', timeevol_stride=timeevol_stride, \
				to_plot_time_evol=to_plot_timeevol, seeds=seeds, \
				verbose=max(verbose - 1, 0))

	Nt_states = phi_0_evol.shape[1]
	d_phi_0[d_phi_0 == 0] = 1
	d_phi_1[d_phi_1 == 0] = 1
	d_phi_0_evol[d_phi_0_evol == 0] = 1
	d_phi_1_evol[d_phi_1_evol == 0] = 1
	steps = np.arange(Nt_states) * timeevol_stride
	#stab_inds = (steps < stab_step) & (steps > 5 * L2)
	stab_inds = steps > stab_step

	reg = 2

	if(cost_mode == 0):
		err =   [np.mean(((phi_0 - phi0_target) / d_phi_0)**2), \
				np.mean(((phi_1 - phi1_target) / d_phi_1)**2)]
	elif(cost_mode == 1):
		err =   [np.mean(((phi_0 - phi0_target) / d_phi_0)**2), \
				np.mean(((phi_1 - phi1_target) / d_phi_1)**2), \
				np.mean(np.array([np.mean(((phi_0_evol[i, stab_inds] - phi_0[i]) / d_phi_0_evol[i, stab_inds])**2) for i in range(N_species)])), \
				np.mean(np.array([np.mean(((phi_1_evol[i, stab_inds] - phi_1[i]) / d_phi_1_evol[i, stab_inds])**2) for i in range(N_species)]))]
	elif(cost_mode == 2):
		x0, d_x0 = phi_to_x(phi_0, d_phi_0)
		x1, d_x1 = phi_to_x(phi_1, d_phi_1)
		x0_target = phi0_target[2] / phi0_target[1]
		x1_target = phi1_target[2] / phi1_target[1]
		err = [((x0 - x0_target) / d_x0)**2, ((x1 - x1_target) / d_x1)**2]
	elif(cost_mode in [3, 4]):
		x0, d_x0 = phi_to_x(phi_0, d_phi_0)
		x1, d_x1 = phi_to_x(phi_1, d_phi_1)
		x0_target = phi0_target[2] / phi0_target[1]
		x1_target = phi1_target[2] / phi1_target[1]
		bug_num = 1e7
		if(cost_mode == 3):
			err = [my.eps_err(x0, x0_target)**2, my.eps_err(x1, x1_target)**2 * 500, (phi_0[0] < 0.97) * bug_num, (0.97 > phi_1[1]) * bug_num]
		else:
			err = [my.eps_err(x0, x0_target)**2, my.eps_err(x1, x1_target)**2 * 5000, np.any(phi_0_evol[0, :] < 0.9) * bug_num, np.any(phi_1_evol[1, :] < 0.9) * bug_num]

	if(verbose):
		print('================================== cost report ====================')
		print('stab_step =', stab_step)
		print('Temp = ', -4 / e_full[1,1])
		print('e', e)
		print('mu', mu)
		#print('mu_chi', mu + (z_neib / 2) * np.array([e_full[1,1], e_full[2,2]]))
		print('phi0 = ', my.v2s(phi_0), '+-', my.v2s(d_phi_0), '; phi0_target =', my.v2s(phi0_target), '; err =', my.v2s((phi_0 - phi0_target) / d_phi_0))
		print('phi1 = ', my.v2s(phi_1), '+-', my.v2s(d_phi_1), '; phi1_target =', my.v2s(phi1_target), '; err =', my.v2s((phi_1 - phi1_target) / d_phi_1))
		if(cost_mode in [2, 3, 4]):
			print('x0 =', my.f2s(x0), '+-', my.f2s(d_x0), '; x0_target =', x0_target, '; err =', my.f2s((x0 - x0_target) / d_x0))
			print('x1 =', my.f2s(x1), '+-', my.f2s(d_x1), '; x1_target =', x1_target, '; err =', my.f2s((x1 - x1_target) / d_x1))
		print('cost_arr =', my.v2s(err), '; COST = ', np.sum(np.array(err)))

		#plot_avg_phi_evol(phi_0, d_phi_0, phi_0_evol, d_phi_0_evol, phi0_target, stab_step, timeevol_stride, r'$\vec{\varphi}_0$')
		#plot_avg_phi_evol(phi_1, d_phi_1, phi_1_evol, d_phi_1_evol, phi1_target, stab_step, timeevol_stride, r'$\vec{\varphi}_1$')
		#plt.show()

	err = np.sum(np.array(err))

	if(to_plot_debug):
		plt.show()

	return err

'''
(T = 0.4664)
e/T = -2.68010292 -1.34005146 -1.71526587
mu/T:
5.09661988, 4.51121567 - strict 30000
5.19394261, 4.81840104 - 30000
5.14511172, 4.92238326 - 100000
'''

def find_optimal_mu(MC_move_mode, L, e0, mu0, Temp0, phi0_target, phi1_target, Nt, \
				interface_mode, N_runs, cost_mode, to_plot_timeevol=False, \
				timeevol_stride=-3000, to_plot_debug=False, stab_step=-10, \
				opt_mode=0, verbose=0, mu_step=None, Temp_step=0.02, \
				stab_step_FFS=-5, seeds=None):
	L2 = L**2

	assert(phi0_target[0] > phi0_target[1]), 'ERROR: phi0_target is supposed to have 0-component majority, but phi0 = ' + str(phi0_target)
	assert(phi1_target[1] > phi1_target[0]), 'ERROR: phi1_target is supposed to have 0-component majority, but phi1 = ' + str(phi1_target)
	if(stab_step < 0):
		stab_step = int(min(L2, Nt / 2) * (-stab_step))

	e0 = np.array([e0[1,1], e0[1,2], e0[2,2]])
	mu0 = np.array([mu0[1], mu0[2]])

	if(timeevol_stride < 0):
		timeevol_stride = np.int_(- Nt / timeevol_stride)

	if(opt_mode == 0):
		x0 = np.copy(mu0)
		bounds=((5.1, 5.5), (4, 10))
		bounds=((4.8, 5.35), (3, 6))
		x_opt = \
			scipy.optimize.minimize(\
				lambda x: \
					cost_fnc(MC_move_mode, L, e0, x, Nt, interface_mode, \
							phi0_target, phi1_target, \
							timeevol_stride, N_runs, \
							stab_step, verbose=verbose, \
							cost_mode=cost_mode, \
							stab_step_FFS=stab_step_FFS, \
							seeds=seeds), \
				x0, method="Nelder-Mead", \
				)  #, bounds=bounds, method='SLSQP'
	elif(opt_mode == 1):
		x0 = np.array(list(e0) + list(mu0))
		x_opt = \
			scipy.optimize.minimize(\
				lambda x: \
					cost_fnc(MC_move_mode, L, x[:3], x[3:], Nt, interface_mode, \
							phi0_target, phi1_target, timeevol_stride, \
							N_runs, stab_step, verbose=verbose, \
							cost_mode=cost_mode, \
							stab_step_FFS=stab_step_FFS, \
							seeds=seeds), \
				x0, \
				bounds=((-3,-2), (-2,-1), (-2, -1), (5.1, 5.5), (4, 10)))  #, method='SLSQP'
	elif(opt_mode == 2):
		x0 = np.array([Temp0] + list(mu0 * Temp0))
		#mu_step = 0.05

		cost_lambda = \
			lambda x: \
				cost_fnc(MC_move_mode, L, e0 * Temp0 / x[0], x[1:] / x[0], Nt, interface_mode, \
						phi0_target, phi1_target, timeevol_stride, N_runs, \
						stab_step, verbose=verbose, cost_mode=cost_mode, \
						stab_step_FFS=stab_step_FFS, seeds=seeds)
		if(mu_step is None):
			x_opt = scipy.optimize.minimize(cost_lambda, x0, method="Nelder-Mead")
		else:
			x_step = np.array([Temp_step, mu_step, mu_step])
			jac_fnc = lambda x : (cost_lambda(x + x_step/2) - cost_lambda(x - x_step/2)) / x_step
			x_opt = \
				scipy.optimize.minimize(cost_lambda, x0, \
										bounds=((0.7, 0.9), (3.05, 3.2), (3, 6)), \
										jac=jac_fnc)

	print(x_opt)

	return x_opt

def run_many(MC_move_mode, L, e, mu, N_runs, interface_mode, \
			OP_A=None, OP_B=None, N_spins_up_init=None, N_saved_states_max=-1, \
			OP_interfaces_AB=None, OP_interfaces_BA=None, \
			OP_sample_BF_A_to=None, OP_sample_BF_B_to=None, \
			OP_match_BF_A_to=None, OP_match_BF_B_to=None, \
			Nt_per_BF_run=None, verbose=None, to_plot_time_evol=False, \
			timeevol_stride=-3000, to_plot_k_distr=False, N_k_bins=10, \
			mode='BF', N_init_states_AB=None, N_init_states_BA=None, \
			init_gen_mode=-2, to_plot_committer=None, to_get_timeevol=None, \
			target_states0=None, target_states1=None, stab_step=-5,
			init_composition=None, to_save_npz=True, to_recomp=0, \
			R_clust_init=None, seeds=None, interfacesIDs_to_plot_dens='main', \
			N_fourier=5, to_plot=True):
	if(verbose is None):
		verbose = lattice_gas.get_verbose()
	L2 = L**2
	
	old_seed = lattice_gas.get_seed()
	if(seeds is None):
		seeds = [old_seed]
	N_seeds = len(seeds)
	if(N_seeds > 1):
		assert(N_seeds == N_runs), 'ERROR: N_seeds = %d != N_runs = %d' % (N_seeds, N_runs)
	else:
		seeds = np.arange(N_runs) + seeds[0]
	
	if(to_get_timeevol is None):
		to_get_timeevol = ('BF' == mode)
	swap_type_move = (MC_move_mode in [move_modes['swap'], move_modes['long_swap']])
	
	if('BF' in mode):
		assert(Nt_per_BF_run is not None), 'ERROR: BF mode but no Nt_per_run provided'
		# if(OP_A is None):
			# assert(len(OP_interfaces_AB) > 0), 'ERROR: nor OP_A neither OP_interfaces_AB were provided for run_many(BF)'
			# OP_A = OP_interfaces_AB[0]
		if(stab_step < 0):
			stab_step = int(min(L2, Nt_per_BF_run / 2) * (-stab_step))
		if(timeevol_stride < 0):
			timeevol_stride = np.int_(- Nt_per_BF_run / timeevol_stride)
	
	elif('FFS' in mode):
		assert(N_init_states_AB is not None), 'ERROR: FFS_AB mode but no "N_init_states_AB" provided'
		assert(OP_interfaces_AB is not None), 'ERROR: FFS_AB mode but no "OP_interfaces_AB" provided'
		if(to_get_timeevol):
			assert(timeevol_stride > 0), 'ERROR: timeevol_stride = %d <= 0 for FFS-style mode' % (timeevol_stride)
		
		N_OP_interfaces_AB = len(OP_interfaces_AB)
		OP_AB = OP_interfaces_AB / OP_scale[interface_mode]
		if(mode == 'FFS_AB'):
			OP = np.copy(OP_AB)
	
	ln_k_AB_data = np.empty(N_runs)
	ln_k_BA_data = np.empty(N_runs)
	k_AB_BFcount_N_data = np.empty(N_runs)
	rho_fncs = [[]] * N_runs
	d_rho_fncs = [[]] * N_runs
	if(mode == 'BF'):
		ln_k_bc_AB_data = np.empty(N_runs)
		ln_k_bc_BA_data = np.empty(N_runs)
	
	elif(mode == 'BF_2sides'):
		Nt_states = Nt_per_BF_run // timeevol_stride
		phi_0_data = np.empty((N_species, N_runs))
		phi_1_data = np.empty((N_species, N_runs))
		phi_0_evol_data = np.empty((N_species, Nt_states, N_runs))
		phi_1_evol_data = np.empty((N_species, Nt_states, N_runs))
	
	elif('FFS' in mode):
		flux0_AB_data = np.empty(N_runs)
		probs_AB_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		PB_AB_data = np.empty((N_OP_interfaces_AB, N_runs))
		d_PB_AB_data = np.empty((N_OP_interfaces_AB, N_runs))
		PB_sigmoid_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		d_PB_sigmoid_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		PB_linfit_sigmoid_data = np.empty((2, N_runs))
		PB_linfit_sigmoid_inds_data = np.empty((N_OP_interfaces_AB - 1, N_runs), dtype=bool)
		OP0_sigmoid_AB_data = np.empty(N_runs)
		PB_erfinv_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		d_PB_erfinv_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		PB_linfit_erfinv_data = np.empty((2, N_runs))
		PB_linfit_erfinv_inds_data = np.empty((N_OP_interfaces_AB - 1, N_runs), dtype=bool)
		OP0_erfinv_AB_data = np.empty(N_runs)
		ZeldovichG_AB_data = np.empty(N_runs)
		cluster_centered_map_total_interps = [[]] * N_runs
		cluster_map_XYlims = np.zeros(2)
		rho_fourier2D_data = np.empty((N_OP_interfaces_AB, 2, N_fourier, N_runs))
	
	npz_basepath = os.path.join(my.git_root_path(), 'izing_npzs')
	npz_basename = 'MCmoves' + str(MC_move_mode) + '_L' + str(L) + \
					'_eT' + my.join_lbls([e[1,1], e[1,2], e[2,2]], '_') + \
					(('_phi' + my.join_lbls(init_composition[1:], '_')) if(swap_type_move) \
						else ('_muT' + my.join_lbls(mu[1:], '_')))
	
	for i in range(N_runs):
		lattice_gas.init_rand(seeds[i])
		if(mode == 'BF'):
			npz_BF_filename = npz_basename + '_OPab' + str(OP_A) + '_' + str(OP_B) + \
								'_stride' + str(timeevol_stride) + \
								'_Nstates' + str(N_saved_states_max) + \
								'_ID' + str(lattice_gas.get_seed()) + '_manyBF'
			npz_BF_filepath = os.path.join(npz_basepath, npz_BF_filename + '.npz')
			
			if((not os.path.isfile(npz_BF_filepath)) or to_recomp):
				F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, \
					rho_fncs[i], d_rho_fncs[i], k_bc_AB_BF, k_bc_BA_BF, \
					k_AB_BF, _, k_BA_BF, _, _, _, _, _, _, _, k_AB_BFcount, _, _ = \
						proc_T(MC_move_mode, L, e, mu, Nt_per_BF_run, interface_mode, \
								OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
								verbose=verbose, to_plot_time_evol=to_plot_time_evol, \
								to_plot_F=False, to_plot_correlations=False, to_plot_ETS=False, \
								timeevol_stride=timeevol_stride, to_estimate_k=True,
								to_get_timeevol=True, N_saved_states_max=N_saved_states_max,
								init_composition=init_composition, \
								R_clust_init=R_clust_init, \
								to_save_npz=to_save_npz, to_recomp=(to_recomp > 1))
				
				if(to_save_npz):
					print('writing', npz_BF_filepath)
					np.savez(npz_BF_filepath, F_new=F_new, d_F_new=d_F_new, \
							OP_hist_centers_new=OP_hist_centers_new, \
							OP_hist_lens_new=OP_hist_lens_new, \
							rho_fncs=rho_fncs[i], \
							d_rho_fncs=d_rho_fncs[i], \
							k_bc_AB_BF=k_bc_AB_BF, k_bc_BA_BF=k_bc_BA_BF, \
							k_AB_BF=k_AB_BF, k_BA_BF=k_BA_BF, \
							k_AB_BFcount=k_AB_BFcount)
			
			else:
				print('loading', npz_BF_filepath)
				npz_data = np.load(npz_BF_filepath, allow_pickle=True)
				
				F_new = npz_data['F_new']
				d_F_new = npz_data['d_F_new']
				OP_hist_centers_new = npz_data['OP_hist_centers_new']
				OP_hist_lens_new = npz_data['OP_hist_lens_new']
				rho_fncs[i] = npz_data['rho_fncs']
				d_rho_fncs[i] = npz_data['d_rho_fncs']
				k_bc_AB_BF = npz_data['k_bc_AB_BF']
				k_bc_BA_BF = npz_data['k_bc_BA_BF']
				k_AB_BF = npz_data['k_AB_BF']
				k_BA_BF = npz_data['k_BA_BF']
				k_AB_BFcount = npz_data['k_AB_BFcount']
			
			ln_k_bc_AB_data[i] = np.log(k_bc_AB_BF * 1)   # proc_T returns 'k', not 'log(k)'; *1 for units [1/step]
			ln_k_bc_BA_data[i] = np.log(k_bc_BA_BF * 1)
			ln_k_AB_data[i] = np.log(k_AB_BF * 1)
			ln_k_BA_data[i] = np.log(k_BA_BF * 1)
		if(mode == 'BF_2sides'):
			npz_BF2s_filename = npz_basename + '_stride' + str(timeevol_stride) + \
								'_Nt' + str(Nt_per_BF_run) + \
								'_ID' + str(lattice_gas.get_seed()) + \
								'_BF2sides'
			npz_BF2s_filepath = os.path.join(npz_basepath, npz_BF2s_filename + '.npz')
			
			if((not os.path.isfile(npz_BF2s_filepath)) or to_recomp):
				phi_0_data[:, i], _, phi_1_data[:, i], _, \
					phi_0_evol_data[:, :, i], phi_1_evol_data[:, :, i], = \
						run_phi01_pair(MC_move_mode, L, e, mu, Nt_per_BF_run, interface_mode, \
									timeevol_stride, stab_step, verbose=verbose, \
									to_plot_timeevol=to_plot_time_evol)
				
				if(to_save_npz):
					print('writing', npz_BF2s_filepath)
					np.savez(npz_BF2s_filepath, \
							phi_0_data=phi_0_data[:, i], \
							phi_1_data=phi_1_data[:, i], \
							phi_0_evol_data=phi_0_evol_data[:, :, i], \
							phi_1_evol_data=phi_1_evol_data[:, :, i])
			else:
				print('loading', npz_BF2s_filepath)
				npz_data = np.load(npz_BF2s_filepath, allow_pickle=True)
				
				phi_0_data[:, i] = npz_data['phi_0_data']
				phi_1_data[:, i] = npz_data['phi_1_data']
				phi_0_evol_data[:, :, i] = npz_data['phi_0_evol_data']
				phi_1_evol_data[:, :, i] = npz_data['phi_1_evol_data']
		
		elif(mode  == 'BF_AB'):
			npz_BFAB_filename = npz_basename + '_stride' + str(timeevol_stride) + \
								'_Nt' + str(Nt_per_BF_run) + \
								'_NtSavedStates' + str(N_saved_states_max) + \
								'_ID' + str(lattice_gas.get_seed()) + \
								'_BFAB'
			npz_BFAB_filepath = os.path.join(npz_basepath, npz_BFAB_filename + '.npz')
			
			if((not os.path.isfile(npz_BFAB_filepath)) or to_recomp):
				F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, \
					rho_fncs[i], d_rho_fncs[i], _, _, _, _, _, _, _, _, _, _, _, \
						_, k_AB_BFcount, _, k_AB_BFcount_N_data[i] = \
							proc_T(MC_move_mode, L, e, mu, Nt_per_BF_run, interface_mode, \
								OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
								verbose=verbose, to_plot_time_evol=to_plot_time_evol, \
								timeevol_stride=timeevol_stride, to_estimate_k=False,
								to_get_timeevol=to_get_timeevol, OP_max=OP_B, \
								N_saved_states_max=N_saved_states_max, \
								init_composition=init_composition, \
								R_clust_init=R_clust_init, \
								to_save_npz=to_save_npz, to_recomp=(to_recomp > 1))
				
				if(to_save_npz):
					print('writing', npz_BFAB_filepath)
					np.savez(npz_BFAB_filepath, \
							F_new=F_new, d_F_new=d_F_new, \
							OP_hist_centers_new=OP_hist_centers_new, \
							OP_hist_lens_new=OP_hist_lens_new, \
							rho_fncs=rho_fncs[i], \
							d_rho_fncs=d_rho_fncs[i], \
							k_AB_BFcount=k_AB_BFcount, \
							k_AB_BFcount_N_data=k_AB_BFcount_N_data[i])
			else:
				print('loading', npz_BFAB_filepath)
				npz_data = np.load(npz_BFAB_filepath, allow_pickle=True)
				
				F_new = npz_data['F_new']
				d_F_new = npz_data['d_F_new']
				OP_hist_centers_new = npz_data['OP_hist_centers_new']
				OP_hist_lens_new = npz_data['OP_hist_lens_new']
				rho_fncs[i] = npz_data['rho_fncs']
				d_rho_fncs[i] = npz_data['d_rho_fncs']
				k_AB_BFcount = npz_data['k_AB_BFcount']
				k_AB_BFcount_N_data[i] = npz_data['k_AB_BFcount_N_data']
			
			ln_k_AB_data[i] = np.log(k_AB_BFcount * 1)
		elif(mode == 'FFS_AB'):
			npz_FFSAB_filename_olds = [npz_basename + '_OPab' + str(OP_A) + '_' + str(OP_B) + \
								'_stride' + str(timeevol_stride) + \
								'_Ninit' + '_'.join([str(n) for n in N_init_states_AB]) + \
								'_OPs' + '_'.join([str(p) for p in OP_interfaces_AB]) + \
								'_ID' + str(lattice_gas.get_seed()) + \
								'_FFSAB', \
								npz_basename + '_OPab' + str(OP_A) + '_' + str(OP_B) + \
								'_stride' + str(timeevol_stride) + \
								'_Ninit' + '_'.join([str(n) for n in N_init_states_AB[:2]]) + \
								'_OPs' + '_'.join([str(p) for p in OP_interfaces_AB]) + \
								'_Nfourier' + str(N_fourier) + \
								'_ID' + str(lattice_gas.get_seed()) + \
								'_FFSAB', \
								npz_basename + '_OPab' + str(OP_A) + '_' + str(OP_B) + \
								'_stride' + str(table_data.timeevol_stride_dict[my.f2s(e[1,1], n=3)]) + \
								'_Ninit' + '_'.join([str(n) for n in N_init_states_AB[:2]]) + \
								'_OPs' + '_'.join([str(p) for p in np.append(OP_interfaces_AB[[0, len(OP_interfaces_AB)-1]], len(OP_interfaces_AB))]) + \
								'_Nfourier' + str(N_fourier) + \
								'_ID' + str(lattice_gas.get_seed()) + \
								'_FFSAB']
			
			npz_FFSAB_filename_new = npz_basename + \
								(('_stride' + str(timeevol_stride)) if(to_get_timeevol) else '') + \
								'_Ninit' + '_'.join([str(n) for n in N_init_states_AB[:2]]) + \
								'_OPs' + '_'.join([str(p) for p in np.append(OP_interfaces_AB[[0, len(OP_interfaces_AB)-1]], len(OP_interfaces_AB))]) + \
								'_Nfourier' + str(N_fourier) + \
								'_ID' + str(lattice_gas.get_seed()) + \
								'_FFSAB'
			
			npz_FFSAB_filepath_olds = [os.path.join(npz_basepath, fn + '.npz') for fn in npz_FFSAB_filename_olds]
			npz_FFSAB_filepath_new = os.path.join(npz_basepath, npz_FFSAB_filename_new + '.npz')
			
			npz_FFSAB_filepath = keep_new_npz_file(npz_FFSAB_filepath_olds, npz_FFSAB_filepath_new)
			
			if((not os.path.isfile(npz_FFSAB_filepath)) or (to_recomp > 0)):
				probs_AB_data[:, i], _, ln_k_AB_data[i], _, flux0_AB_data[i], _, rho_new, d_rho_new, OP_hist_centers_new, OP_hist_lens_new, \
					PB_AB_data[:, i], d_PB_AB_data[:, i], \
					PB_sigmoid_data[:, i], d_PB_sigmoid_data[:, i], PB_linfit_sigmoid_data[:, i], PB_linfit_sigmoid_inds_data[:, i], OP0_sigmoid_AB_data[i], \
					PB_erfinv_data[:, i], d_PB_erfinv_data[:, i], PB_linfit_erfinv_data[:, i], PB_linfit_erfinv_inds_data[:, i], OP0_erfinv_AB_data[i], ZeldovichG_AB_data[i], \
					state_Rdens_centers_new, state_centered_Rdens_total_new, d_state_centered_Rdens_new, \
					cluster_centered_map_total_new, cluster_map_centers_new, rho_fourier2D_data[:, :, :, i] = \
						proc_FFS_AB(MC_move_mode, L, e, mu, N_init_states_AB, \
								OP_interfaces_AB, interface_mode, \
								stab_step=stab_step, timeevol_stride=timeevol_stride, \
								OP_sample_BF_to=OP_sample_BF_A_to, \
								OP_match_BF_to=OP_match_BF_A_to, \
								init_gen_mode=init_gen_mode, \
								to_get_timeevol=to_get_timeevol, \
								init_composition=init_composition, \
								to_recomp=max([to_recomp - 1, 0]), \
								to_save_npz=to_save_npz, \
								to_do_hists=True, \
								N_fourier=N_fourier)
				
				if(to_save_npz):
					print('writing', npz_FFSAB_filepath)
					np.savez(npz_FFSAB_filepath, \
							probs_AB_data=probs_AB_data[:, i], \
							ln_k_AB_data=ln_k_AB_data[i], \
							flux0_AB_data=flux0_AB_data[i], \
							rho_new=rho_new, d_rho_new=d_rho_new, \
							OP_hist_centers_new=OP_hist_centers_new, \
							OP_hist_lens_new=OP_hist_lens_new, \
							PB_AB_data=PB_AB_data[:, i], \
							d_PB_AB_data=d_PB_AB_data[:, i], \
							PB_sigmoid_data=PB_sigmoid_data[:, i], \
							d_PB_sigmoid_data=d_PB_sigmoid_data[:, i], \
							PB_linfit_sigmoid_data=PB_linfit_sigmoid_data[:, i], \
							PB_linfit_sigmoid_inds_data=PB_linfit_sigmoid_inds_data[:, i], \
							OP0_AB_data=OP0_sigmoid_AB_data[i], \
							PB_erfinv_data=PB_erfinv_data[:, i], \
							d_PB_erfinv_data=d_PB_erfinv_data[:, i], \
							PB_linfit_erfinv_data=PB_linfit_erfinv_data[:, i], \
							PB_linfit_erfinv_inds_data=PB_linfit_erfinv_inds_data[:, i], \
							OP0_erfinv_AB_data=OP0_erfinv_AB_data[i], \
							ZeldovichG_AB_data=ZeldovichG_AB_data[i], \
							state_Rdens_centers_new=state_Rdens_centers_new, \
							state_centered_Rdens_total_new=state_centered_Rdens_total_new, \
							d_state_centered_Rdens_new=d_state_centered_Rdens_new, \
							cluster_centered_map_total_new=cluster_centered_map_total_new, \
							cluster_map_centers_new=cluster_map_centers_new, \
							rho_fourier2D_data=rho_fourier2D_data[:, :, :, i])
			else:
				print('loading', npz_FFSAB_filepath)
				npz_data = np.load(npz_FFSAB_filepath, allow_pickle=True)
				
				probs_AB_data[:, i] = npz_data['probs_AB_data']
				ln_k_AB_data[i] = npz_data['ln_k_AB_data']
				flux0_AB_data[i] = npz_data['flux0_AB_data']
				rho_new = npz_data['rho_new']
				d_rho_new = npz_data['d_rho_new']
				OP_hist_centers_new = npz_data['OP_hist_centers_new']
				OP_hist_lens_new = npz_data['OP_hist_lens_new']
				PB_AB_data[:, i] = npz_data['PB_AB_data']
				d_PB_AB_data[:, i] = npz_data['d_PB_AB_data']
				PB_sigmoid_data[:, i] = npz_data['PB_sigmoid_data']
				d_PB_sigmoid_data[:, i] = npz_data['d_PB_sigmoid_data']
				PB_linfit_sigmoid_data[:, i], _ = my.get_npz_field(npz_data, ['PB_linfit_sigmoid_data', 'PB_linfit_data'])
				PB_linfit_sigmoid_inds_data[:, i], _ = my.get_npz_field(npz_data, ['PB_linfit_sigmoid_inds_data', 'PB_linfit_inds_data'])
				OP0_sigmoid_AB_data[i], _ = my.get_npz_field(npz_data, ['OP0_sigmoid_AB_data', 'OP0_AB_data'])
				PB_erfinv_data[:, i] = npz_data['PB_erfinv_data']
				d_PB_erfinv_data[:, i] = npz_data['d_PB_erfinv_data']
				PB_linfit_erfinv_data[:, i] = npz_data['PB_linfit_erfinv_data']
				PB_linfit_erfinv_inds_data[:, i] = npz_data['PB_linfit_erfinv_inds_data']
				OP0_erfinv_AB_data[i] = npz_data['OP0_erfinv_AB_data']
				ZeldovichG_AB_data[i] = npz_data['ZeldovichG_AB_data']
				state_Rdens_centers_new = npz_data['state_Rdens_centers_new']
				state_centered_Rdens_total_new = npz_data['state_centered_Rdens_total_new']
				d_state_centered_Rdens_new = npz_data['d_state_centered_Rdens_new']
				cluster_centered_map_total_new = npz_data['cluster_centered_map_total_new']
				cluster_map_centers_new = npz_data['cluster_map_centers_new']
				rho_fourier2D_data[:, :, :, i] = npz_data['rho_fourier2D_data']
			
			if(i == 0):
				state_centered_Rdens_total_data = [[]] * N_OP_interfaces_AB
				
				for j in range(N_OP_interfaces_AB):
					state_centered_Rdens_total_data[j] = [[]] * N_species
					for k in range(N_species):
						state_centered_Rdens_total_data[j][k] = \
							np.empty((len(state_centered_Rdens_total_new[j][k]), N_runs))
			
			cluster_centered_map_total_interps[i] = [[]] * N_OP_interfaces_AB
			
			for j in range(2):
				cluster_map_XYmax_local = max(abs(cluster_map_centers_new[j]))
				if(cluster_map_XYlims[j] < cluster_map_XYmax_local):
					cluster_map_XYlims[j] = cluster_map_XYmax_local
			for j in range(N_OP_interfaces_AB):
				cluster_centered_map_total_interps[i][j] = \
					scipy.interpolate.RegularGridInterpolator(\
						(cluster_map_centers_new[0], cluster_map_centers_new[1]), \
						cluster_centered_map_total_new[j],
						bounds_error=False, fill_value=0)
				
				#print(cluster_centered_map_total_interps[i][j])
				#input('ok')
				
				for k in range(N_species):
					state_centered_Rdens_total_data[j][k][:, i] = state_centered_Rdens_total_new[j][k]
			
			if(to_get_timeevol):
				rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, rho_new, fill_value='extrapolate')
				d_rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, d_rho_new, fill_value='extrapolate')
				rho_AB = rho_fncs[i](OP_hist_centers_new)
				d_rho_AB = d_rho_fncs[i](OP_hist_centers_new)
				F_new = -np.log(rho_AB * OP_hist_lens_new)
				d_F_new = d_rho_AB / rho_AB
				F_new = F_new - F_new[0]
		
		if(verbose > 0):
			print(r'%s %lf %%' % (mode, (i+1) / (N_runs) * 100))
		else:
			print('%s %lf %%                             \r' % (mode, (i+1) / (N_runs) * 100), end='')
		
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
	
	print(mode, 'DONE                                   ')
	
	ln_k_AB, d_ln_k_AB = get_average(ln_k_AB_data)
	
	if(to_get_timeevol):
		F, d_F = get_average(F_data, axis=1)
	else:
		F = None
		d_F = None
		OP_hist_centers = None
		OP_hist_lens = None
	
	if(mode == 'BF_AB'):
		k_AB_BFcount_N, d_k_AB_BFcount_N = get_average(k_AB_BFcount_N_data)
	
	if(mode == 'BF_2sides'):
		phi_0, d_phi_0 = get_average(phi_0_data, axis=1)
		phi_1, d_phi_1 = get_average(phi_1_data, axis=1)
		phi_0_evol, d_phi_0_evol = get_average(phi_0_evol_data, axis=2)
		phi_1_evol, d_phi_1_evol = get_average(phi_1_evol_data, axis=2)
	
	elif(mode == 'BF'):
		ln_k_bc_AB, d_ln_k_bc_AB = get_average(ln_k_bc_AB_data)
		ln_k_bc_BA, d_ln_k_bc_BA = get_average(ln_k_bc_BA_data)
		ln_k_BA, d_ln_k_BA = get_average(ln_k_BA_data)
	
	elif('FFS' in mode):
		flux0_AB, d_flux0_AB = get_average(flux0_AB_data, mode='log')
		OP0_sigmoid_AB, d_OP0_sigmoid_AB = get_average(OP0_sigmoid_AB_data)
		OP0_erfinv_AB, d_OP0_erfinv_AB = get_average(OP0_erfinv_AB_data)
		ZeldovichG_AB, d_ZeldovichG_AB = get_average(ZeldovichG_AB_data)
		probs_AB, d_probs_AB = get_average(probs_AB_data, mode='log', axis=1)
		PB_AB, d_PB_AB = get_average(PB_AB_data, mode='log', axis=1)
		
		rho_fourier2D, d_rho_fourier2D = get_average(rho_fourier2D_data, axis=3)
		rho_fourier2D_Abs = np.linalg.norm(rho_fourier2D, axis=1)
		d_rho_fourier2D_Abs = np.sqrt(np.sum((d_rho_fourier2D * rho_fourier2D)**2, axis=1)) / rho_fourier2D_Abs
		# TODO: implement this for single FFS_AB
		
		#Temp = 4 / e[1,1]
		#Nc_0 = table_data.Nc_reference_data[str(Temp)] if(str(Temp) in table_data.Nc_reference_data) else None
		PB_sigmoid, d_PB_sigmoid, linfit_sigmoid_AB, linfit_sigmoid_AB_inds, _ = \
			get_sigmoid_fit(PB_AB[:-1], d_PB_AB[:-1], OP_AB[:-1], \
							sgminv_fnc=lambda x: -np.log(1/x - 1), \
							d_sgminv_fnc=lambda x, y: 1/(x*(1-x)))
		
		PB_erfinv, d_PB_erfinv, linfit_erfinv_AB, linfit_erfinv_AB_inds, _ = \
			get_sigmoid_fit(PB_AB[:-1], d_PB_AB[:-1], OP_AB[:-1], \
							sgminv_fnc=lambda x: scipy.special.erfinv(2 * x - 1), \
							d_sgminv_fnc=lambda x, y: np.exp(y**2) * np.sqrt(np.pi))
		
		#PB_erfinv = scipy.special.erfinv(2 * PB_AB[:-1] - 1)
		#d_PB_erfinv = (scipy.special.erfinv(2 * (PB_AB[:-1] + d_PB_AB[:-1]) - 1) - scipy.special.erfinv(2 * (PB_AB[:-1] - d_PB_AB[:-1]) - 1)) / 2
		
		rho_chi2 = None
		if(state_Rdens_centers_new is not None):
			Rmax_cluster_draw = max(cluster_map_XYlims)
			cluster_map_x = np.linspace(-Rmax_cluster_draw, Rmax_cluster_draw, int(Rmax_cluster_draw * 2) + 1)
			cluster_map_y = np.linspace(-Rmax_cluster_draw, Rmax_cluster_draw, int(Rmax_cluster_draw * 2) + 1)
			cluster_map_X, cluster_map_Y = np.meshgrid(cluster_map_x, cluster_map_y, indexing='ij')
			cluster_centered_map_total_data = np.empty((cluster_map_X.shape[0], cluster_map_X.shape[1], N_runs, N_OP_interfaces_AB))
			for j in range(N_OP_interfaces_AB):
				for i in range(N_runs):
					cluster_centered_map_total_data[:, :, i, j] = cluster_centered_map_total_interps[i][j]((cluster_map_X, cluster_map_Y))
			
			state_centered_Rdens_total = [[]] * N_OP_interfaces_AB
			d_state_centered_Rdens_total = [[]] * N_OP_interfaces_AB
			cluster_centered_map_total = [[]] * N_OP_interfaces_AB
			d_cluster_centered_map_total = [[]] * N_OP_interfaces_AB
			for j in range(N_OP_interfaces_AB):
				cluster_centered_map_total[j], d_cluster_centered_map_total[j] = \
					get_average(cluster_centered_map_total_data[:, :, :, j], axis=2)
				
				state_centered_Rdens_total[j] = [[]] * N_species
				d_state_centered_Rdens_total[j] = [[]] * N_species
				for k in range(N_species):
					state_centered_Rdens_total[j][k], d_state_centered_Rdens_total[j][k] = \
						get_average(state_centered_Rdens_total_data[j][k], axis=1)
					
			rho_fit_params = np.empty((N_species, 4, N_OP_interfaces_AB))
			d_rho_fit_params = np.empty((N_species, 4, N_OP_interfaces_AB))
			species_to_fit_rho_inds = np.array([0, 1], dtype=int)   # , 2
			rho_fit_fncs = []
			rho_splines = []
			rho_chi2 = np.empty((N_species, N_OP_interfaces_AB))
			R_peak = np.empty((N_species, N_OP_interfaces_AB))
			R_fit_minmax = np.empty((2, N_species, N_OP_interfaces_AB))
			for k in range(N_species):
				rho_fit_fncs.append([])
				for j in range(N_OP_interfaces_AB):
					if(k in species_to_fit_rho_inds):
						rho_fit_params[k, :, j], d_rho_fit_params[k, :, j] = \
							rho_fit_full(state_Rdens_centers_new, \
										state_centered_Rdens_total[j][k], \
										d_state_centered_Rdens_total[j][k], \
										init_composition[k], \
										OP_interfaces_AB[j], L2, \
										fit_error_name='specie-N%d' % k, \
										mode=k)
						rho_fit_fncs[k].append(lambda r, \
													pp1=rho_fit_params[k, :, j], \
													pp2=init_composition[k], \
													pp3=OP_interfaces_AB[j], \
													pp4=L2, pp5=k: \
												rho_fit_sgmfnc_template(r, pp1, pp2, pp3, pp4, mode=pp5))
						
						R_peak[k, j] = rho_fit_params[k, 2, j]
						R_fit_minmax[1, k, j] = R_peak[k, j] + rho_fit_params[k, 1, j] * 25
					else:
						rho_fit_fnc_new, R_fit_minmax[1, k, j], R_peak[k, j], rho_spline_new = \
							rho_fit_spline(state_Rdens_centers_new, \
											state_centered_Rdens_total[j][k], \
											d_state_centered_Rdens_total[j][k], \
											init_composition[k], L2)   # s=-0.8, k=5
						rho_fit_fncs[k].append(rho_fit_fnc_new)
					
					rho_const_inds = state_Rdens_centers_new >= R_fit_minmax[1, k, j]
					R_fit_inds = (state_Rdens_centers_new >= R_peak[k, j]) & (state_Rdens_centers_new < R_fit_minmax[1, k, j])
					peak_ind = np.argmin(np.abs(state_Rdens_centers_new - R_peak[k, j]))
					
					rho_inf = \
						np.average(state_centered_Rdens_total[j][k][rho_const_inds], \
								weights=1/d_state_centered_Rdens_total[j][k][rho_const_inds]**2)
					R_fit_left_ind = \
						np.argmax((state_centered_Rdens_total[j][k][R_fit_inds] - rho_inf) * \
								  (state_centered_Rdens_total[j][k][peak_ind] - rho_inf) < 0)
					R_fit_minmax[0, k, j] = state_Rdens_centers_new[R_fit_inds][R_fit_left_ind]
					rho_chi2_inds = (state_Rdens_centers_new >= R_fit_minmax[0, k, j]) & (state_Rdens_centers_new < R_fit_minmax[1, k, j])
					rho_chi2[k, j] = np.mean(((state_centered_Rdens_total[j][k][rho_chi2_inds] - rho_inf) / d_state_centered_Rdens_total[j][k][rho_chi2_inds])**2)
			
			sgmfit_species_id = 0
			assert(sgmfit_species_id in species_to_fit_rho_inds), 'ERROR: Species N%d to be used for width estimation but this specie is not included in species to be fit: %s' % (sgmfit_species_id, str(species_to_fit_rho_inds))
			sgm_logfit_Nmin = OP0_erfinv_AB
			sgm_logfit_inds = (OP_interfaces_AB > sgm_logfit_Nmin)
			sgm_logfit, sgm_logfit_cov = \
				np.polyfit(np.log(OP_interfaces_AB[sgm_logfit_inds]), \
							np.log(rho_fit_params[sgmfit_species_id, 1, sgm_logfit_inds]), \
							1, cov=True, w=1 / (d_rho_fit_params[sgmfit_species_id, 1, sgm_logfit_inds] / rho_fit_params[sgmfit_species_id, 1, sgm_logfit_inds]))
	
	if(to_plot):
		if(state_Rdens_centers_new is not None):
			# =============== Rdens ================
			fig_state_Rdens = [[]] * N_species
			ax_state_Rdens = [[]] * N_species
			fig_state_Rdens_log = [[]] * N_species
			ax_state_Rdens_log = [[]] * N_species
			ThL_lbl = get_ThL_lbl(e, mu, init_composition, L, swap_type_move)
			for k in range(N_species):
				fig_state_Rdens[k], ax_state_Rdens[k], _ = my.get_fig('r', r'$\phi_%d$' % k, title=(r'$\phi_%d(r)$; ' % k) + ThL_lbl)
				fig_state_Rdens_log[k], ax_state_Rdens_log[k], _ = my.get_fig('r', r'$\phi_%d$' % k, title=(r'$\phi_%d(r)$; ' % k) + ThL_lbl, yscl='logit')
			
			fig_rho_fourier, ax_rho_fourier, _ = my.get_fig('$n$', r'$\sim F_n[\phi]$', title=r'$F[\phi(\theta)](n)$')
			
			if(interfacesIDs_to_plot_dens == 'main'):
				interfacesIDs_to_plot_dens = np.array([0, np.argmin(np.abs(OP_interfaces_AB - OP0_erfinv_AB)), len(OP_interfaces_AB) - 1], dtype=int)
			elif(interfacesIDs_to_plot_dens == 'all'):
				interfacesIDs_to_plot_dens = np.arange(len(N_OP_interfaces_AB))
			N_interfaces_to_plot = len(interfacesIDs_to_plot_dens)
			
			fig_cluster_map = [[]] * N_interfaces_to_plot
			ax_cluster_map = [[]] * N_interfaces_to_plot
			
			ax_rho_fourier.plot([0, N_fourier], [0] * 2, '--', label='F=0')
			for i, OPind in enumerate(interfacesIDs_to_plot_dens):
				cluster_lbl = r'$OP[%d] = %d$; ' % (OPind, OP_interfaces_AB[OPind])
				
				ax_rho_fourier.errorbar(np.arange(N_fourier+1), np.append(1, rho_fourier2D_Abs[OPind, :]), \
										yerr = np.append(0, d_rho_fourier2D_Abs[OPind, :]), \
										label=cluster_lbl, color=my.get_my_color(i))
				
				for k in range(N_species):
					#chi2_lbl = r'$\chi_{%s,%s}^2 = %s$' % (my.f2s(R_fit_minmax[0, k, OPind]), my.f2s(R_fit_minmax[1, k, OPind]), my.f2s(rho_chi2[k, OPind]))
					chi2_lbl = r'$\chi^2 = %s$' % (my.f2s(rho_chi2[k, OPind]))
#										rho_fit_sgmfnc_template(state_Rdens_centers_new, rho_fit_params[k, :, OPind], init_composition[k], OP_interfaces_AB[OPind], L2, mode=k), \
					ax_state_Rdens[k].plot(state_Rdens_centers_new, \
										rho_fit_fncs[k][OPind](state_Rdens_centers_new), \
										'--', color=my.get_my_color(i), label=chi2_lbl)
					ax_state_Rdens_log[k].plot(state_Rdens_centers_new, \
										rho_fit_fncs[k][OPind](state_Rdens_centers_new), \
										'--', color=my.get_my_color(i), label=chi2_lbl)
				
				fig_cluster_map[i], ax_cluster_map[i], fig_cluster_map_id = my.get_fig('x', 'y', title=cluster_lbl + ThL_lbl)
				im_cluster_map = ax_cluster_map[i].imshow(cluster_centered_map_total[OPind], \
										extent = [min(cluster_map_y), max(cluster_map_y), \
												  min(cluster_map_x), max(cluster_map_x)], \
										interpolation ='bilinear', origin ='lower', aspect='auto')
				plt.figure(fig_cluster_map_id)
				cbar = plt.colorbar(im_cluster_map)
				
				for k in range(N_species):
					ax_state_Rdens[k].errorbar(state_Rdens_centers_new, \
											state_centered_Rdens_total[OPind][k], \
											yerr=d_state_centered_Rdens_total[OPind][k], \
											label=cluster_lbl, color=my.get_my_color(i), fmt='.-')
					ax_state_Rdens_log[k].errorbar(state_Rdens_centers_new, \
											state_centered_Rdens_total[OPind][k], \
											yerr=d_state_centered_Rdens_total[OPind][k],
											label=cluster_lbl, color=my.get_my_color(i), fmt='.-')
			
			for k in range(N_species):
				Rdens_lims = [min(state_Rdens_centers_new), max(state_Rdens_centers_new)]
				ax_state_Rdens[k].plot(Rdens_lims, [init_composition[k]] * 2, '--', \
										label='$\phi_{bulk} = %s$' % (my.f2s(init_composition[k], n=4)))
				ax_state_Rdens_log[k].plot(Rdens_lims, [init_composition[k]] * 2, '--', \
										label='$\phi_{bulk} = %s$' % (my.f2s(init_composition[k], n=4)))
			
			fig_rhoW, ax_rhoW, _ = my.get_fig(r'$N_{clust}$', r'$w_{\rho_{%d}}$' % (sgmfit_species_id), xscl='log', yscl='log')
			#fig_rhoW, ax_rhoW, _ = my.get_fig(r'$N_{clust}$', r'$w_{\rho_{%d}}$' % (sgmfit_species_id), xscl='log')
			ax_rhoW.errorbar(OP_interfaces_AB, rho_fit_params[sgmfit_species_id, 1, :],  yerr=d_rho_fit_params[sgmfit_species_id, 1, :], fmt='.', label='data')
			ax_rhoW.plot(OP_interfaces_AB, np.exp(np.polyval(sgm_logfit, np.log(OP_interfaces_AB))), \
						'--', label=r'$k = %s$' % (my.errorbar_str(sgm_logfit[0], np.sqrt(sgm_logfit_cov[0,0]))))
			ax_rhoW.plot([sgm_logfit_Nmin] * 2, [min(rho_fit_params[sgmfit_species_id, 1, :]), max(rho_fit_params[sgmfit_species_id, 1, :])], \
						'--', label=r'$N_{min} = N^* = %s$' % (my.f2s(sgm_logfit_Nmin)))
			
			my.add_legend(fig_rhoW, ax_rhoW)
			my.add_legend(fig_rho_fourier, ax_rho_fourier)
			
			for k in range(N_species):
				my.add_legend(fig_state_Rdens[k], ax_state_Rdens[k])
				my.add_legend(fig_state_Rdens_log[k], ax_state_Rdens_log[k])
		
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
			ThL_lbl = get_ThL_lbl(e, mu, init_composition, L, swap_type_move)
			plot_PB_AB(ThL_lbl, feature_label[interface_mode], title[interface_mode], \
						interface_mode, OP_AB[:-1], PB_AB[:-1], d_PB=d_PB_AB[:-1], \
						PB_sgm=PB_sigmoid, d_PB_sgm=d_PB_sigmoid, \
						linfit_sgm=linfit_sigmoid_AB, linfit_sgm_inds=linfit_sigmoid_AB_inds, \
						OP0_sgm=OP0_sigmoid_AB, d_OP0_sgm=d_OP0_sigmoid_AB, \
						PB_erfinv=PB_erfinv, d_PB_erfinv=d_PB_erfinv, \
						linfit_erfinv=linfit_erfinv_AB, linfit_erfinv_inds=linfit_erfinv_AB_inds, \
						OP0_erfinv=OP0_erfinv_AB, d_OP0_erfinv=d_OP0_erfinv_AB, \
						clr=my.get_my_color(1))
	
	lattice_gas.init_rand(old_seed)
	
	if(mode == 'BF'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, ln_k_bc_AB, d_ln_k_bc_AB, ln_k_bc_BA, d_ln_k_bc_BA
	elif(mode == 'FFS_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, probs_AB, d_probs_AB, PB_AB, d_PB_AB, OP0_erfinv_AB, d_OP0_erfinv_AB, ZeldovichG_AB, d_ZeldovichG_AB, rho_chi2
	elif(mode == 'BF_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, k_AB_BFcount_N, d_k_AB_BFcount_N
	elif(mode == 'BF_2sides'):
		return phi_0, d_phi_0, phi_1, d_phi_1, phi_0_evol, d_phi_0_evol, phi_1_evol, d_phi_1_evol
# 	if(mode == 'FFS'):
# 		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, flux0_AB, d_flux0_AB, flux0_BA, d_flux0_BA, probs_AB, d_probs_AB, probs_BA, d_probs_BA, PB_AB, d_PB_AB, PA_BA, d_PA_BA, OP0_AB, d_OP0_AB, OP0_BA, d_OP0_BA

def get_rho_inf(rho_bulk, rho_0, Ncl, L2):
	#rho_inf = (rho_bulk) / (1 - Ncl / L2)
	return (rho_bulk - rho_0 * (Ncl / L2)) / (1 - Ncl / L2)

def rho_fit_sgmfnc_template(r, prm, rho_bulk, Ncl, L2, \
						capilar_correction_min=0, \
						capilar_correction_min_sgm=-0.2, \
 						capilar_correction_max=-25, \
						capilar_correction_max_sgm=-5, \
						mode=0, A=0, \
						verbose=False):
# 						capilar_correction_min=-4, \
	rho_inf = get_rho_inf(rho_bulk, prm[0], Ncl, L2)
	
	if(verbose):
		print('Ncl =', Ncl, '; rho_inf =', rho_inf)
	
	if(capilar_correction_min == 0):
		if(mode in [0,1]):
			#capilar_correction_min = prm[2] + np.sign(prm[0] - rho_inf) * prm[1] * np.log(1/rho_inf - 1)
			capilar_correction_min = -np.abs(np.log(1/rho_inf - 1))
		elif(mode in [2]):
			capilar_correction_min = -1.8
	if(capilar_correction_min < 0):
		capilar_correction_min = prm[2] - capilar_correction_min * prm[1]
	if(capilar_correction_min_sgm < 0):
		capilar_correction_min_sgm *= -prm[1]
	if(capilar_correction_max == 0):
		capilar_correction_max = np.sqrt(L2)
		capilar_correction_max_sgm = capilar_correction_max / 100
	if(capilar_correction_max < 0):
		capilar_correction_max = prm[2] - capilar_correction_max * prm[1]
	if(capilar_correction_max_sgm < 0):
		capilar_correction_max_sgm *= -prm[1]
		
	if(mode in [0, 1]):
		return prm[0] + (rho_inf - prm[0]) / (1 + np.exp(-((r - prm[2]) / prm[1]))) + \
				prm[3] / r / (1 + np.exp(-(r - capilar_correction_min) / capilar_correction_min_sgm)) / (1 + np.exp((r - capilar_correction_max) / capilar_correction_max_sgm))
	elif(mode in [2]):
		return rho_bulk + A * np.exp(-(np.abs(r - prm[2]) / prm[1])**4) + \
			prm[3] / r / (1 + np.exp(-(r - prm[0]) / capilar_correction_min_sgm)) / (1 + np.exp((r - capilar_correction_max) / capilar_correction_max_sgm))

def rho_fit_full(R_centers, Rdens, d_Rdens, rho_bulk, Ncl, L2, mode=0, \
				sgmfit_rho_min=0.1, sgmfit_rho_max=0.9, Rdens_eps=-1e-2, \
				fit_error_name='', debug=1, strict_bounds=True):
	ok_inds = (d_Rdens > 0) & (Rdens < 1)
	fit_error_name = fit_error_name + '; rho_bulk = ' + str(rho_bulk)
	
	if(mode in [0, 1]):
		rho_fit_params = np.empty(4)
		rho_fit_params[0] = Rdens[ok_inds][0]   # argmin(r)
		rho_inf = get_rho_inf(rho_bulk, rho_fit_params[0], Ncl, L2)
		rho_linsgm_inds = (Rdens > sgmfit_rho_min) & (Rdens < sgmfit_rho_max) & ok_inds
		assert(np.any(rho_linsgm_inds)), 'ERROR: no points suited for sigmoidal fit of %s' % (fit_error_name)
		
		max_Rdens = max(Rdens[ok_inds])
		min_Rdens = min(Rdens[ok_inds])
		if(Rdens_eps < 0):
			Rdens_eps *= -min_Rdens
		Rdens_scaled = (max_Rdens - Rdens[rho_linsgm_inds] + Rdens_eps) / (max_Rdens - min_Rdens + Rdens_eps)
		rho_linsgmfit = np.polyfit(R_centers[rho_linsgm_inds], \
									-np.log(1 / Rdens_scaled - 1), \
									1, \
									w = Rdens_scaled * (1 - Rdens_scaled) / d_Rdens[rho_linsgm_inds])
		rho_fit_params[1] = 1 / abs(rho_linsgmfit[0])
		rho_fit_params[2] = -rho_linsgmfit[1] / rho_linsgmfit[0]
		
		#R_correction = rho_fit_params[2] + rho_fit_params[1] * abs(np.log(1/rho_inf - 1))
		R_correction = R_centers[ok_inds][np.argmax((Rdens[ok_inds] > rho_inf) if(rho_bulk > 0.5) else (Rdens[ok_inds] < rho_inf))]
		
		opt_fnc_full = lambda prm: np.sum(((rho_fit_sgmfnc_template(R_centers[ok_inds], prm, rho_bulk, Ncl, L2, mode=mode) - Rdens[ok_inds]) / d_Rdens[ok_inds])**2)   # , capilar_correction_min=R_correction
		#rho_fit_params[3] = 0
		prm3_opt_options = {}
		if(strict_bounds):
			rho_fit_params[3] = R_correction * \
								((max(Rdens[ok_inds]) - rho_inf) if(rho_bulk > 0.5) \
								else (min(Rdens[ok_inds]) - rho_inf))
			prm3_opt_options['bounds'] = (min(0, 2 * rho_fit_params[3]), max(0, 2 * rho_fit_params[3]))
		
		rho_fit_params[3] = scipy.optimize.minimize_scalar(lambda x: opt_fnc_full(my.subst_x1d(rho_fit_params, x, 3)), **prm3_opt_options).x
		
		if(strict_bounds):
			#[0]: (max([0, 1 - 5 * (1 - rho_fit_params[0])]), 1) if(rho_fit_params[0] > 0.5) else (0, min([1, 5 * rho_fit_params[0]])
			d_rho_fit_params_0 = d_Rdens[ok_inds][0]
			opt_bounds = ((0, min(1, rho_fit_params[0] + d_rho_fit_params_0 * 10)) if(rho_bulk > 0.5) else (max(0, rho_fit_params[0] - d_rho_fit_params_0 * 10), 1), \
						(rho_fit_params[1] / 2, rho_fit_params[1] * 2), \
						(rho_fit_params[2] / 2, rho_fit_params[2] * 2), \
						(min(0, 2 * rho_fit_params[3]), max(0, 2 * rho_fit_params[3])))
		else:
			R0_estimate = np.sqrt(Ncl / np.pi)
			opt_bounds = ((0, 1), (0.1, R0_estimate), (R0_estimate / 3, R0_estimate * 3), (-1,1))
			# [0]: desnity [0;1]
			# [1]: width. It is (0;None) but really it is < 2 so here it is
			# [2]: R0 ~ np.sqrt(Ncl / np.pi)
			# [3]: unclear, but usually |a| << 1
		
		rho_fit_optim_res = scipy.optimize.minimize(opt_fnc_full, rho_fit_params)
		if(np.any([((rho_fit_optim_res.x[i] < opt_bounds[i][0]) or (opt_bounds[i][1] < rho_fit_optim_res.x[i])) for i in range(len(rho_fit_params))])):
			rho_fit_optim_res = scipy.optimize.minimize(opt_fnc_full, rho_fit_params, bounds=opt_bounds)
		
		rho_fit_params = rho_fit_optim_res.x
		
		if(isinstance(rho_fit_optim_res.hess_inv, scipy.optimize._lbfgsb_py.LbfgsInvHessProduct)):
			d_rho_fit_params = np.sqrt(rho_fit_optim_res.hess_inv.todense().diagonal())
		else:
			d_rho_fit_params = np.sqrt(rho_fit_optim_res.hess_inv.diagonal())
	elif(mode in [2]):
		### !!!!!!!! not supported
		rho_fit_params = np.empty(4)
		
		# R_peak_left, R_peak_right, R_peak_rhomax, R_peak_avg, \
			# rho_max, peak_width, rho_inf, fit_inds, peak_inds = \
					# get_peak_features(R_centers, Rdens, rho_bulk, L2, ok_inds=ok_inds, iter=2)
		
		# rho_fit_params[2] = R_peak_rhomax
		# rho_fit_params[1] = peak_width
		# rho_fit_params[0] = R_peak_right
		
		# opt_fnc_full = lambda prm: np.sum(((rho_fit_sgmfnc_template(R_centers[fit_inds], prm, rho_bulk, Ncl, L2, capilar_correction_min=R_peak_right + peak_width/5, mode=mode, A=rho_max - rho_bulk) - Rdens[fit_inds]) / d_Rdens[fit_inds])**2)
		# #rho_fit_params[3] = 0
		# prm3_opt_options = {}
		# if(strict_bounds):
			# rho_fit_params[3] = (min(Rdens[fit_inds]) - rho_bulk) * R_peak_right
			# prm3_opt_options['bounds'] = (min(0, 2 * rho_fit_params[3]), max(0, 2 * rho_fit_params[3]))
		
		# rho_fit_params[3] = scipy.optimize.minimize_scalar(lambda x: opt_fnc_full(my.subst_x1d(rho_fit_params, x, 3)), **prm3_opt_options).x
		
		# if(strict_bounds):
			# #[0]: (rho_fit_params[0] / 1.05, rho_fit_params[0] * 1.05)
			# opt_bounds = ((rho_fit_params[0] - rho_fit_params[1]/5, rho_fit_params[0] + rho_fit_params[1]/5), \
						# (rho_fit_params[1] / 2, rho_fit_params[1] * 2), \
						# (rho_fit_params[2] - (R_centers[1] - R_centers[0]) * 1.1, rho_fit_params[2] + (R_centers[1] - R_centers[0]) * 1.1), \
						# (min(0, 2 * rho_fit_params[3]), max(0, 2 * rho_fit_params[3])))
		# else:
			# R0_estimate = np.sqrt(Ncl / np.pi)
			# opt_bounds = ((0, 1), (0.1, R0_estimate), (R0_estimate / 3, R0_estimate * 3), (-1,1))
			# # [0]: desnity [0;1]
			# # [1]: width. It is (0;None) but really it is < 2 so here it is
			# # [2]: R0 ~ np.sqrt(Ncl / np.pi)
			# # [3]: unclear, but usually |a| << 1
		
		# if(debug % 2 == 0):
			# fig, ax, _ = my.get_fig('R', r'$\phi_2$', yscl='log')
			# ax.errorbar(R_centers[fit_inds], Rdens[fit_inds], d_Rdens[fit_inds])
			# ax.errorbar(R_centers[peak_inds], Rdens[peak_inds], d_Rdens[peak_inds])
			# ax.plot([min(R_centers[fit_inds]), max(R_centers[fit_inds])], [rho_bulk]*2, '--')
			# #ax.plot([min(R_centers[fit_inds]), max(R_centers[fit_inds])], [rho_bulk]*2, '--')
			# ax.plot(R_centers[ok_inds], rho_fit_sgmfnc_template(R_centers[ok_inds], rho_fit_params, rho_bulk, Ncl, L2, capilar_correction_min=R_peak_right, mode=mode))
			# plt.show()
		
		# rho_fit_optim_res = scipy.optimize.minimize(opt_fnc_full, rho_fit_params)
		# if(np.any([((rho_fit_optim_res.x[i] < opt_bounds[i][0]) or (opt_bounds[i][1] < rho_fit_optim_res.x[i])) for i in range(len(rho_fit_params))])):
			# rho_fit_optim_res = scipy.optimize.minimize(opt_fnc_full, rho_fit_params, bounds=opt_bounds)
		
		# rho_fit_params = rho_fit_optim_res.x
		
		# if(isinstance(rho_fit_optim_res.hess_inv, scipy.optimize._lbfgsb_py.LbfgsInvHessProduct)):
			# d_rho_fit_params = np.sqrt(rho_fit_optim_res.hess_inv.todense().diagonal())
		# else:
			# d_rho_fit_params = np.sqrt(rho_fit_optim_res.hess_inv.diagonal())
		
		# if(debug % 3 == 0):
			# fig, ax, _ = my.get_fig('R', r'$\phi_2$', yscl='log')
			# ax.errorbar(R_centers[fit_inds], Rdens[fit_inds], d_Rdens[fit_inds])
			# ax.errorbar(R_centers[peak_inds], Rdens[peak_inds], d_Rdens[peak_inds])
			# ax.plot([min(R_centers[fit_inds]), max(R_centers[fit_inds])], [rho_bulk]*2, '--')
			# ax.plot([R_peak_right] * 2, [min(Rdens[fit_inds]), max(Rdens[fit_inds])], '--')
			
			# ax.plot(R_centers[ok_inds], rho_fit_sgmfnc_template(R_centers[ok_inds], rho_fit_params, rho_bulk, Ncl, L2, capilar_correction_min=R_peak_right, mode=mode))
			# plt.show()
		
		
	#print(rho_fit_params, d_rho_fit_params)
	#input('ok')
	
	return rho_fit_params, d_rho_fit_params

def get_peak_features(R_centers, Rdens, rho_bulk, L2, d_Rdens=None, \
						rho_inf=None, ok_inds=None, iter=1):
	if(ok_inds is None):
		ok_inds = (Rdens >= 0)   # all True
	if(rho_inf is None):
		rho_inf = rho_bulk
	
	Rdens_maxind = np.argmax(Rdens[ok_inds])
	R_peak_rhomax = R_centers[ok_inds][Rdens_maxind]
	rho_max = Rdens[Rdens_maxind]
	R_left_to_peak_inds = ok_inds & (R_centers <= R_peak_rhomax)
	rho_core_max = (Rdens[R_left_to_peak_inds][0] * 2) if(d_Rdens is None) \
					else (Rdens[R_left_to_peak_inds][0] + 4 * d_Rdens[R_left_to_peak_inds][0])
	R_left_peak_ind = np.where(Rdens[R_left_to_peak_inds] < rho_core_max)[0]   # find the last index with (R < Rpeak) and (rho < rho_core_max_estimate)
	if(len(R_left_peak_ind) == 0):
		R_left_peak_ind = 0
	else:
		R_left_peak_ind = min(R_left_peak_ind[-1] + 1, np.sum(R_left_to_peak_inds) - 1)
	R_peak_left = R_centers[R_left_to_peak_inds][R_left_peak_ind]
	fit_inds = ok_inds & (R_centers >= R_peak_left)
	R_right_to_peak_inds = ok_inds & (R_centers > R_peak_rhomax)
	R_right_peak_ind = max(0, np.argmax(Rdens[R_right_to_peak_inds] < rho_inf) - 1)   # first index with (R > Rpeak) and (rho < rho_inf)
	R_peak_right = R_centers[R_right_to_peak_inds][R_right_peak_ind]
	peak_inds = fit_inds & (R_centers <= R_peak_right)
	if(np.sum(peak_inds) < 2):
		fig, ax, _ = my.get_fig('R', r'$\phi_{fit}$')
		ax.errorbar(R_centers[fit_inds], Rdens[fit_inds], yerr=d_Rdens[fit_inds])
		ax.plot([min(R_centers[fit_inds]), max(R_centers[fit_inds])], [rho_inf]*2, '--')
		ax.plot([R_peak_right]*2, [min(Rdens[fit_inds]), max(Rdens[fit_inds])], '--')
		plt.show()
	if(np.all(Rdens[peak_inds] <= rho_inf)):
		fig, ax, _ = my.get_fig('R', r'$\phi_{peak}$')
		print(R_centers[peak_inds])
		ax.errorbar(R_centers[peak_inds], Rdens[peak_inds], yerr=d_Rdens[peak_inds])
		ax.plot([min(R_centers[peak_inds]), max(R_centers[peak_inds])], [rho_inf]*2, '--')
		plt.show()
	R_peak_avg = np.average(R_centers[peak_inds], weights=np.maximum(Rdens[peak_inds] - rho_inf, 0))  # w *= R_centers[fit_inds][peak_inds] because  dV = r dphi dr
	#rho_fit_params[1] = np.sqrt(np.average((Rdens[peak_inds] - R_peak_avg)**2, weights=Rdens[peak_inds]) / (1 - 1/np.sum(peak_inds)))
	peak_width = np.sqrt(np.average((Rdens[peak_inds] - R_peak_avg)**2, weights=Rdens[peak_inds])) * 0.6  # * 0.581368  # * 0.610968
	
	inside_inds = (R_centers < R_peak_right)
	rho_inf = (L2 * rho_bulk - np.trapz(Rdens[inside_inds] * 2*np.pi * R_centers[inside_inds], x=R_centers[inside_inds])) / (L2 - np.pi * R_peak_right**2)
	
	iter -= 1
	
	return (R_peak_left, R_peak_right, R_peak_rhomax, R_peak_avg, rho_max, peak_width, rho_inf, fit_inds, peak_inds) if(iter==0) \
			else get_peak_features(R_centers, Rdens, rho_bulk, L2, d_Rdens=d_Rdens, rho_inf=rho_inf, ok_inds=ok_inds, iter=iter)

def rho_fit_spline(R_centers, Rdens, d_Rdens, rho_bulk, L2, s=None, ext='const', k=3, Rmax_fit=-4, rho_inf_inter=4):
	R_peak_left, R_peak_right, R_peak_rhomax, R_peak_avg, \
		rho_max, peak_width, rho_inf, _, peak_inds = \
				get_peak_features(R_centers, Rdens, rho_bulk, L2, d_Rdens=d_Rdens, iter=rho_inf_inter)
	
	if(Rmax_fit < 0):
		Rmax_fit = R_peak_right - Rmax_fit * (R_peak_right - R_peak_left)
	fit_inds = (R_centers >= R_peak_right) & (R_centers < Rmax_fit)
	
	if(s is not None):
		if(s < 0):
			s = -s * np.sum(fit_inds)
	
	spl = scipy.interpolate.UnivariateSpline(R_centers[fit_inds], Rdens[fit_inds], w=1/d_Rdens[fit_inds], ext=ext, s=s, k=k)
	full_fnc = lambda r, \
					R_peak_right_loc=R_peak_right, \
					Rmax_fit_loc=Rmax_fit, \
					Rdens_max_loc=max(Rdens[fit_inds]), \
					spl_loc=spl, \
					rho_inf_loc=rho_inf: \
				np.piecewise(r, [r < R_peak_right_loc, (r >= R_peak_right_loc) & (r < Rmax_fit_loc), r >= Rmax_fit_loc], [Rdens_max_loc, lambda x: spl_loc(x), rho_inf_loc])
	
	return full_fnc, Rmax_fit, R_peak_rhomax, spl

def get_mu_dependence(MC_move_mode, mu_arr, L, e, N_runs, interface_mode, \
						N_init_states, OP_interfaces, init_gen_mode=-3, \
						init_composition=None, stab_step=-5, \
						to_plot=False, to_save_npy=True, to_recomp=0, \
						seeds=None):
	L2 = L**2
	N_mu = mu_arr.shape[0]
	
	ln_k_AB = np.empty(N_mu)
	d_ln_k_AB = np.empty(N_mu)
	flux0_A = np.empty(N_mu)
	d_flux0_A = np.empty(N_mu)
	OP0_AB = np.empty(N_mu)
	d_OP0_AB = np.empty(N_mu)
	prob_AB = []
	d_prob_AB = []
	PB_AB = []
	d_PB_AB = []
	k_AB_mean = np.empty(N_mu)
	k_AB_min = np.empty(N_mu)
	k_AB_max = np.empty(N_mu)
	d_k_AB = np.empty(N_mu)
	N_OP_interfaces = np.empty(N_mu, dtype=int)
	
	npy_filename = interface_mode + '_eT11_12_22_' + str(e[1,1]) + '_' + str(e[1,2]) + '_' + str(e[2,2]) \
				+ '_L' + str(L) \
				+ '_mu1T_' + ((str(min(mu_arr[:, 1])) + '_' + str(max(mu_arr[:, 1]))) if(max(mu_arr[:, 1]) > min(mu_arr[:, 1])) else str(mu_arr[0, 1])) \
				+ '_mu2T_' + ((str(min(mu_arr[:, 2])) + '_' + str(max(mu_arr[:, 2]))) if(max(mu_arr[:, 2]) > min(mu_arr[:, 2])) else str(mu_arr[0, 2])) \
				+ '_Nintrf' +  str(len(OP_interfaces[0])) \
				+ '_Nruns' + str(N_runs) + '_Ninit'	+ str(N_init_states[0]) \
				+ '_Nstates' + str(N_init_states[-1]) \
				+ '_OP0' + (str(min(OP_interfaces[0]))) + '.npz'
	
	if(os.path.isfile(npy_filename) and (not to_recomp)):
		print('loading from "' + npy_filename + '"')
		npy_data = np.load(npy_filename, allow_pickle=True)
		
		mu_arr = npy_data['mu_arr']
		k_AB_mean = npy_data['k_AB']
		k_AB_errbars = npy_data['d_k_AB']
		OP0_AB = npy_data['P_B']
		d_OP0_AB = npy_data['d_P_B']
	else:
		print('file\n"%s"\nnot found, computing' % npy_filename)
		
		for i_mu in range(N_mu):
			N_OP_interfaces[i_mu] = len(OP_interfaces[i_mu])
			_, _, _, _, ln_k_AB[i_mu], d_ln_k_AB[i_mu], flux0_A[i_mu], \
				d_flux0_A[i_mu], prob_AB_new, d_prob_AB_new, PB_AB_new, d_PB_AB_new, \
				OP0_AB[i_mu], d_OP0_AB[i_mu], _, _, _ = \
					run_many(MC_move_mode, L, e, mu_arr[i_mu, :], N_runs, interface_mode, \
						stab_step=stab_step, \
						N_init_states_AB=N_init_states, \
						OP_interfaces_AB=OP_interfaces[i_mu], \
						mode='FFS_AB', init_gen_mode=init_gen_mode, \
						init_composition=init_composition, \
						to_get_timeevol=False, to_plot_committer=False, \
						to_recomp=max([0, to_recomp - 1]), \
						to_save_npz=to_save_npy, seeds=seeds)
			# ZeldovichG_AB, d_ZeldovichG_AB, rho_chi2 are ommitted
			
			prob_AB.append(prob_AB_new)
			d_prob_AB.append(d_prob_AB_new)
			PB_AB.append(PB_AB_new)
			d_PB_AB.append(d_PB_AB_new)
			#print(ln_k_AB, d_ln_k_AB, flux0_A, d_flux0_A)
			#print(prob_AB.shape)
			#print(d_prob_AB.shape)
			#print((N_OP_interfaces - 1, N_mu))
			#input('hi')
			k_AB_mean[i_mu], k_AB_min[i_mu], k_AB_max[i_mu], d_k_AB[i_mu] = get_log_errors(ln_k_AB[i_mu], d_ln_k_AB[i_mu], lbl='k_AB_' + str(i_mu))
		
		k_AB_errbars = np.concatenate(((k_AB_mean - k_AB_min)[np.newaxis, :], (k_AB_max - k_AB_mean)[np.newaxis, :]), axis=0)
		
		if(to_save_npy):
			np.savez(npy_filename, mu_arr=mu_arr, k_AB=k_AB_mean, d_k_AB=k_AB_errbars, P_B=OP0_AB, d_P_B=d_OP0_AB)
			print('saved "' + npy_filename + '"')
	
	if(to_plot):
		TL_lbl = '$\epsilon_{11, 22}/T = ' + my.f2s(e[1,1]) + ',' + my.f2s(e[2,2]) + '$; L = ' + str(L)
		fig_k, ax_k, _ = my.get_fig('$\mu_1/T$', '$k_{AB} / L^2$ [trans/(sweep $\cdot l^2$)]', '$k_{AB}(\mu)$; ' + TL_lbl, yscl='log')
		#fig_Nc, ax_Nc, _ = my.get_fig('$\mu_1/T$', '$N_c$', '$N_c(\mu)$; ' + TL_lbl, xscl='log', yscl='log')
		fig_Nc, ax_Nc, _ = my.get_fig('$\mu_1/T$', '$N_c$', '$N_c(\mu)$; ' + TL_lbl, yscl='log')
		
		ax_k.errorbar(mu_arr[:, 1], k_AB_mean, yerr=k_AB_errbars, fmt='.', label='data')
		
		ax_Nc.errorbar(mu_arr[:, 1], OP0_AB, yerr=d_OP0_AB, fmt='.', label='data')
		
		Temp = 4 / e[1,1]
		Temp_key = str(round(Temp, 1))
		if((Temp_key in table_data.k_AB_reference_data) and (Temp_key in table_data.Nc_reference_data)):
			#mu_kAB_ref = table_data.k_AB_reference_data[Temp_key][0, :]
			mu_kAB_ref = table_data.k_AB_reference_data[Temp_key][0, :] * 2 / Temp - 2 * e[1,1] + mu_shift
			#mu_Nc_ref = table_data.Nc_reference_data[Temp_key][0, :]
			mu_Nc_ref = table_data.Nc_reference_data[Temp_key][0, :] * 2 / Temp - 2 * e[1,1] + mu_shift
			
			#k_AB_ref = table_data.k_AB_reference_data[Temp_key][1, :] / table_data.L_reference ** 2
			k_AB_ref = table_data.k_AB_reference_data[Temp_key][1, :]
			Nc_AB_ref = table_data.Nc_reference_data[Temp_key][1, :]
			
			ax_k.plot(mu_kAB_ref, k_AB_ref, '-o', label='reference')
			ax_Nc.plot(mu_Nc_ref, Nc_AB_ref, '-o', label='reference')
			
			fig_k_err, ax_k_err, _ = my.get_fig('$\mu_1/T$', '$[max(\ln(k_{my, ref})) / min(\ln(k_{my, ref})) - 1]$', '$\ln(k)_{error}(\mu)$; ' + TL_lbl, yscl='log')
			#fig_Nc_err, ax_Nc_err, _ = my.get_fig('$\mu_1/T$', '$[max(Nc_{my, ref}) / min(Nc_{my, ref}) - 1]$', '$Nc_{error}(\mu)$; ' + TL_lbl, xscl='log', yscl='log')
			fig_Nc_err, ax_Nc_err, _ = my.get_fig('$\mu_1/T$', '$[max(Nc_{my, ref}) / min(Nc_{my, ref}) - 1]$', '$Nc_{error}(\mu)$; ' + TL_lbl, yscl='log')
			
			fig_k_ratio, ax_k_ratio, _ = my.get_fig('$\mu_1/T$', '$k_{my} / k_{ref}$', '$k_{error}(\mu)$; ' + TL_lbl)
			def get_k_err_vals(k, d_ln_k, k_ref):
				r = np.exp(np.abs(np.log(k / k_ref)))
				# d(k/k0 - 1) / (k/k0) = dk/k
				# d(k0/k - 1) / (k0/k) = dk*k0/k^2 / (k0/k) / dk/k
				# So, 'd_ans / r == d_k/k'
				return r - 1, r * d_ln_k
			mu_kAB_ref_inds = np.array([np.argmin(np.abs(mu_kAB_ref - mu_my)) for mu_my in mu_arr[:, 1]])
			mu_Nc_ref_inds = np.array([np.argmin(np.abs(mu_Nc_ref - mu_my)) for mu_my in mu_arr[:, 1]])
			k_err, d_k_err = get_k_err_vals(np.log(k_AB_mean), np.abs(d_ln_k_AB / np.log(k_AB_mean)), np.log(k_AB_ref[mu_kAB_ref_inds]))
			Nc_err, d_Nc_err = get_k_err_vals(OP0_AB, d_OP0_AB / OP0_AB, Nc_AB_ref[mu_Nc_ref_inds])
			
			ax_k_err.errorbar(mu_arr[:, 1], k_err, yerr=d_k_err, fmt='.', label='data')
			my.add_legend(fig_k_err, ax_k_err)
			
			ax_k_ratio.errorbar(mu_arr[:, 1], k_AB_mean / k_AB_ref[mu_kAB_ref_inds], yerr=d_ln_k_AB * abs(k_AB_mean / k_AB_ref[mu_kAB_ref_inds]), fmt='.', label='data')
			my.add_legend(fig_k_ratio, ax_k_ratio)
			
			ax_Nc_err.errorbar(mu_arr[:, 1], Nc_err, yerr=d_Nc_err, fmt='.', label='data')
			my.add_legend(fig_Nc_err, ax_Nc_err)
		my.add_legend(fig_k, ax_k)
		my.add_legend(fig_Nc, ax_Nc)

def load_Q_target_phase_data(suff):
	phi2_target = np.load('string_radial_ternary_V3_2_phi0%s.npy' % suff).flatten()   # phi0-Q ~ phi2-My
	phi1_target = np.load('string_radial_ternary_V3_2_phi1%s.npy' % suff).flatten()
	phi0_target = 1 - phi1_target - phi2_target
	N_Q_points = len(phi0_target)
	return np.concatenate((phi0_target, phi1_target, phi2_target)).reshape((3, N_Q_points))

def plot_avg_phi_evol(phi_mean, d_phi_mean, phi_evol, d_phi_evol, target_phi, stab_step, timeevol_stride, title, time_scl='linear'):
	fig, ax, _ = my.get_fig('step', r'$\varphi$', title=title, xscl=time_scl)
	phi_evol_timesteps = np.arange(phi_evol.shape[1]) * timeevol_stride
	to_plot_steps = phi_evol_timesteps[1:]
	for i in range(N_species):
		ax.errorbar(to_plot_steps, phi_evol[i, 1:], yerr=d_phi_evol[i, 1:], fmt='.', label=str(i), color=my.get_my_color(i))
		ax.errorbar([min(to_plot_steps), max(to_plot_steps)], [phi_mean[i]] * 2, fmt='.-', yerr = [d_phi_mean[i]]*2, color=my.get_my_color(i), label = ('mean' if(i == 0) else None))
		ax.plot([min(to_plot_steps), max(to_plot_steps)], [target_phi[i]] * 2, ':', color=my.get_my_color(i), label = ('target' if(i == 0) else None))
	ax.plot([stab_step] * 2, [0, 1], '--', label='stab time')
	my.add_legend(fig, ax)

def plot_Q_target_phase_data(target_states, used_target_ids, phi_obtained, d_phi_obtained, name, cost_mode):
	if(len(phi_obtained.shape) == 1):
		phi_obtained = phi_obtained.reshape((1, len(phi_obtained)))
		d_phi_obtained = d_phi_obtained.reshape((1, len(d_phi_obtained)))

	fig_phi, ax_phi, _ = \
		my.get_fig('id', r'$\vec{\varphi}_{%s}$' % name)

	for i in range(N_species):
		ax_phi.plot(target_states[:, i], label=r'$\varphi_%s$' % i, color=my.get_my_color(i), lw=1)
		ax_phi.errorbar(used_target_ids, phi_obtained[:, i], fmt='.', yerr=d_phi_obtained[:, i], color=my.get_my_color(i), lw=1.5)
	#ax_phi.plot([try_id] * 2, [0, 1], '--', label=r'current target')
	ax_phi.plot(np.sum(target_states, axis=1), label='total', lw=1)
	my.add_legend(fig_phi, ax_phi)

	fig_x, ax_x = None, None
	if(cost_mode in [2, 3, 4]):
		fig_x, ax_x, _ = \
			my.get_fig('id', r'$x_{%s} = \varphi_2 / \varphi_1$' % name, title = '$x_{%s}$' % name)

		x_obtained = np.array([phi_to_x(phi_obtained[i, :], d_phi_obtained[i, :]) for i in range(phi_obtained.shape[0])])
		ax_x.plot(target_states[:, 2] / target_states[:, 1])
		ax_x.errorbar(used_target_ids, x_obtained[:, 0], fmt='.', yerr=x_obtained[:, 1])

		return fig_x, ax_x

	return fig_phi, ax_phi

# 					OP_sample_BF_A_to=None, OP_match_BF_A_to=None, \

def get_Tphi1_dependence(Temp_s, phi1_s, phi2, MC_move_mode_name, \
					L, e, interface_mode, OP_interfaces, \
					OP0_constraint_s=[], init_gen_mode=-2, \
					OP_sample_BF_to=None, OP_match_BF_to=None, \
					timeevol_stride=-3000, to_get_timeevol=None, \
					to_save_npz=False, to_recomp=0, N_fourier=5, \
					seeds_range=np.arange(1000, 1200), \
					N_points_OP0constr=1000, N_ID_groups=None, \
					fit_ord=2, to_plot=False, 
					npz_path='/scratch/gpfs/yp1065/Izing/npzs/', \
					npz_suff='FFStraj'):
	N_OP_interfaces = len(OP_interfaces)
	N_Temps = len(Temp_s)
	N_phi1s = len(phi1_s)
	
	ln_k = np.zeros((N_Temps, N_phi1s))
	d_ln_k = np.zeros((N_Temps, N_phi1s))
	flux0 = np.zeros((N_Temps, N_phi1s))
	d_flux0 = np.zeros((N_Temps, N_phi1s))
	OP0 = np.zeros((N_Temps, N_phi1s))
	d_OP0 = np.zeros((N_Temps, N_phi1s))
	ZeldovichG = np.zeros((N_Temps, N_phi1s))
	d_ZeldovichG = np.zeros((N_Temps, N_phi1s))
	Temps_grid = np.zeros((N_Temps, N_phi1s))
	phi1s_grid = np.zeros((N_Temps, N_phi1s))
	rho_chi2 = np.zeros((N_Temps, N_phi1s, N_species, N_OP_interfaces))
	d_rho_chi2 = np.zeros((N_Temps, N_phi1s, N_species, N_OP_interfaces))
	chi2_rho0 = np.zeros((N_Temps, N_phi1s))
	chi2_rho1 = np.zeros((N_Temps, N_phi1s))
	chi2_rho2 = np.zeros((N_Temps, N_phi1s))
	d_chi2_rho0 = np.zeros((N_Temps, N_phi1s))
	d_chi2_rho1 = np.zeros((N_Temps, N_phi1s))
	d_chi2_rho2 = np.zeros((N_Temps, N_phi1s))
	for i_Temp, Temp in enumerate(Temp_s):
		for i_phi1, phi1 in enumerate(phi1_s):
			n_FFS = table_data.nPlot_FFS_dict[MC_move_mode_name][my.f2s(Temp, n=2)][my.f2s(phi1, n=5)]
			if(n_FFS > 0):
				e_use = e / Temp
				mu = np.array([0] * N_species)
				init_composition = np.array([1 - phi1 - phi2, phi1, phi2])
				N_init_states = np.ones(N_OP_interfaces, dtype=int) * n_FFS
				N_init_states[0] *= 2
				MC_move_mode = lattice_gas.get_move_modes()[MC_move_mode_name]
				
				# It does not have to be L**2 (can be more), but it always is L**2 in practice for the cases of June2023 work
				stab_step = L**2
				
				# old_versions are not valid because no full 'N_init_states' is given
				templ, templ_old = \
					izing.get_FFS_AB_npzTrajBasename(MC_move_mode, L, e_use, \
									'_phi' + my.join_lbls(init_composition[1:], '_'), \
									OP_interfaces, N_init_states, \
									stab_step, OP_sample_BF_to, OP_match_BF_to, \
									timeevol_stride, init_gen_mode, \
									bool(to_get_timeevol), \
									N_fourier, '%d')
				templ = npz_path + templ + '_' + npz_suff + '.npz'
				#templ = npz_path + templ_old[-1] + '_' + npz_suff + '.npz'   # to 'mv' old files to new format
				#print(templ)
				
				found_IDs, found_filepaths, _ = \
					izing.find_finished_runs_npzs(templ, seeds_range, verbose=True)
				#print(found_IDs)
				#input('ok')
				
				N_found_npzs = len(found_IDs)
				if(N_ID_groups is None):
					N_ID_groups_local = int(np.sqrt(N_found_npzs) + 0.5)
				else:
					N_ID_groups_local = int(-N_found_npzs / N_ID_groups + 0.5) if(N_ID_groups < 0) else N_ID_groups
				
				if(N_ID_groups_local > 0):
					Temps_grid[i_Temp, i_phi1] = Temp
					phi1s_grid[i_Temp, i_phi1] = phi1
					
					ID_group_size = int(N_found_npzs / N_ID_groups_local + 0.5)
					ID_group_sizes = np.array([ID_group_size] * N_ID_groups_local, dtype=int)
					N_bigger_ID_groups = N_found_npzs - ID_group_size * N_ID_groups_local
					if(N_bigger_ID_groups > 0):
						ID_group_sizes[:N_bigger_ID_groups] += 1
					ID_groups = [found_IDs[: ID_group_sizes[0]]]
					for i_group in range(1, N_ID_groups_local):
						ID_groups.append(found_IDs[np.sum(ID_group_sizes[:(i_group)]) : np.sum(ID_group_sizes[:(i_group+1)])])
					
					#print(N_ID_groups_local, ID_groups)
					#input('ok')
					
					N_ID_groups_local = len(ID_groups)
					ln_k_local = np.empty(N_ID_groups_local)
					flux0_local = np.empty(N_ID_groups_local)
					OP0_local = np.empty(N_ID_groups_local)
					ZeldovichG_local = np.empty(N_ID_groups_local)
					rho_chi2_local = np.empty((N_ID_groups_local, N_species, N_OP_interfaces))
					PB_local = np.empty((N_ID_groups_local, N_OP_interfaces))
					for i_group in range(N_ID_groups_local):
						_, _, _, _, \
							ln_k_local[i_group], _, \
							flux0_local[i_group], _, \
							_, _, PB_local[i_group, :], _, \
							OP0_local[i_group], _, \
							ZeldovichG_local[i_group], _, \
							rho_chi2_local[i_group, :, :] = \
								run_many(MC_move_mode, L, e_use, mu, len(ID_groups[i_group]), \
									interface_mode, stab_step=stab_step, \
									N_init_states_AB=N_init_states, \
									OP_interfaces_AB=OP_interfaces, \
									mode='FFS_AB', init_gen_mode=init_gen_mode, \
									OP_sample_BF_A_to=OP_sample_BF_to, \
									OP_match_BF_A_to=OP_match_BF_to, \
									timeevol_stride=timeevol_stride, \
									to_get_timeevol=to_get_timeevol, \
									init_composition=init_composition, \
									to_save_npz=to_save_npz, \
									to_recomp=max([to_recomp - 1, 0]), \
									N_fourier=N_fourier, \
									seeds=ID_groups[i_group], \
									to_plot=False)
					
					ln_k[i_Temp, i_phi1], d_ln_k[i_Temp, i_phi1] = my.get_average(ln_k_local)
					flux0[i_Temp, i_phi1], d_flux0[i_Temp, i_phi1] = my.get_average(flux0_local, mode='log')
					OP0[i_Temp, i_phi1], d_OP0[i_Temp, i_phi1] = my.get_average(OP0_local)
					ZeldovichG[i_Temp, i_phi1], d_ZeldovichG[i_Temp, i_phi1] = my.get_average(ZeldovichG_local, mode='log')
					rho_chi2[i_Temp, i_phi1, :, :], d_rho_chi2[i_Temp, i_phi1, :, :] = my.get_average(rho_chi2_local, axis=0)
					PB, d_PB = my.get_average(PB_local, mode='log', axis=0)
					
					#OP0_closest_ind = np.argmin(np.abs(OP_interfaces - OP0[i_Temp, i_phi1]))
					#OP0_closest_ind = np.argmin(np.abs(np.log(1/PB - 1)))
					OP0_closest_ind = np.argmin(np.abs(PB - 0.5))
					chi2_rho0[i_Temp, i_phi1] = rho_chi2[i_Temp, i_phi1, 0, OP0_closest_ind]
					chi2_rho1[i_Temp, i_phi1] = rho_chi2[i_Temp, i_phi1, 1, OP0_closest_ind]
					chi2_rho2[i_Temp, i_phi1] = rho_chi2[i_Temp, i_phi1, 2, OP0_closest_ind]
					d_chi2_rho0[i_Temp, i_phi1] = d_rho_chi2[i_Temp, i_phi1, 0, OP0_closest_ind]
					d_chi2_rho1[i_Temp, i_phi1] = d_rho_chi2[i_Temp, i_phi1, 1, OP0_closest_ind]
					d_chi2_rho2[i_Temp, i_phi1] = d_rho_chi2[i_Temp, i_phi1, 2, OP0_closest_ind]
	
	# d_chi2_rho0, d_chi2_rho1, d_chi2_rho2 = \
		# tuple([(chi2 / 100) for chi2 in [chi2_rho0, chi2_rho1, chi2_rho2]])
	# I do not have an estimate for d_chi2 so this is necessary to make 
	
	ok_inds = (flux0.flatten() > 0)
	N_ok_points = np.sum(ok_inds)
	
	# ==== flatten all the data and extract only good poits ====
	ln_k_arr, d_ln_k_arr, OP0_arr, d_OP0_arr, ZeldovichG_arr, d_ZeldovichG_arr, \
	flux0_arr, d_flux0_arr, chi2_rho0_arr, d_chi2_rho0_arr, chi2_rho1_arr, d_chi2_rho1_arr, \
	chi2_rho2_arr, d_chi2_rho2_arr, Temps_arr, phi1s_arr = \
		tuple([arr.flatten()[ok_inds] for arr in \
			[ln_k, d_ln_k, OP0, d_OP0, ZeldovichG, d_ZeldovichG, \
			flux0, d_flux0, chi2_rho0, d_chi2_rho0, chi2_rho1, d_chi2_rho1, \
			chi2_rho2, d_chi2_rho2, Temps_grid, phi1s_grid]])
	
	# ==== turn selected data_pairs into their logs ====
	ln_flux0_arr, d_ln_flux0_arr, ln_OP0_arr, d_ln_OP0_arr, \
	ln_ZeldovichG_arr, d_ln_ZeldovichG_arr, ln_chi2_rho0_arr, d_ln_chi2_rho0_arr, \
	ln_chi2_rho1_arr, d_ln_chi2_rho1_arr, ln_chi2_rho2_arr, d_ln_chi2_rho2_arr = \
		tuple([item for data_pair in \
					[[np.log(data_pair[0]), data_pair[1] / data_pair[0]] for data_pair in \
					[[flux0_arr, d_flux0_arr], [OP0_arr, d_OP0_arr], \
					 [ZeldovichG_arr, d_ZeldovichG_arr], \
					 [chi2_rho0_arr, d_chi2_rho0_arr], \
					 [chi2_rho1_arr, d_chi2_rho1_arr], \
					 [chi2_rho2_arr, d_chi2_rho2_arr]]] \
				for item in data_pair])
	
	if(N_ok_points >= fit_ord**2):
		#chi2_rho0_fitfnc, _ = fit_Tphi1_grid(Temps_arr, phi1s_arr, ln_ZeldovichG_arr, dz=d_ln_ZeldovichG_arr, xord=fit_ord, yord=fit_ord)
		
		data_arr_pairs = [[ln_OP0_arr, d_ln_OP0_arr], \
						  [ln_k_arr, d_ln_k_arr], \
						  [ln_flux0_arr, d_ln_flux0_arr], \
						  [ln_ZeldovichG_arr, d_ln_ZeldovichG_arr], \
						  [ln_chi2_rho0_arr, d_ln_chi2_rho0_arr], \
						  [ln_chi2_rho1_arr, d_ln_chi2_rho1_arr], \
						  [ln_chi2_rho2_arr, d_ln_chi2_rho2_arr]]
		N_data_pairs = len(data_arr_pairs)
		for i in range(N_data_pairs):
			data_arr_pairs[i].append(fit_Tphi1_grid(Temps_arr, phi1s_arr, data_arr_pairs[i][0], dz=data_arr_pairs[i][1], xord=fit_ord, yord=fit_ord)[0])
			# this makes the pair [z, dz, fitfnc]
		
		N_OP0_constr = len(OP0_constraint_s)
		OP0consr_Tphi1_sets = []
		ln_OP0_fitfnc = data_arr_pairs[0][2]
		if(N_OP0_constr > 0):
			for i in range(N_OP0_constr):
				OP0consr_Tphi1_sets.append(my.find_optim_manifold(lambda x: np.abs(ln_OP0_fitfnc(x[0], x[1]) - np.log(OP0_constraint_s[i])), \
									np.array([[min(Temp_s), max(Temp_s)], [min(phi1_s), max(phi1_s)], [min(ln_OP0_arr), max(ln_OP0_arr)]]).T, \
									trial_points=N_points_OP0constr))
	
	if(to_plot):
		if(N_ok_points >= fit_ord**2):
			figs = [[]] * N_data_pairs
			axs = [[]] * N_data_pairs
			data_names = ['$\ln(N^*)$', '$\ln(k)$', '$\ln(\Phi_A)$', '$\ln(\Gamma)$', r'$\ln(\chi^2_{dip,0})$', r'$\ln(\chi^2_{dip,1})$', r'$\ln(\chi^2_{dip,2})$']
			assert(len(data_names) == N_data_pairs), 'ERROR: data_names = %s\nhas %d elements, but N_data_pairs = %d' % (str(data_names), len(data_names), N_data_pairs)
			for i in range(N_data_pairs):
				figs[i], axs[i] = plot_Tphi1_data(Temps_arr, phi1s_arr, \
						data_arr_pairs[i][0], data_arr_pairs[i][1], data_names[i], data_arr_pairs[i][2], \
						OP0_constraint_s=OP0_constraint_s, OP0consr_Tphi1_sets=OP0consr_Tphi1_sets)
				

def plot_Tphi1_data(x, y, z, dz, z_lbl, z_fitfnc, x_lbl='Temp', y_lbl='$\phi_1$', \
				N_draw_points=1000, draw_only_inside_convexhull=True, \
				points_mode='grid', to_add_legend=False, \
				OP0_constraint_s=[], OP0consr_Tphi1_sets=[]):
	fig, ax, _ = my.get_fig(x_lbl, y_lbl, zlbl=z_lbl, projection='3d')
	
	ax.errorbar(x, y, z, zerr=dz, fmt='.', label='data')
	
	if(points_mode == 'random'):
		xy_draw = np.random.random((N_draw_points, 2))
		sort_inds = np.argsort(xy_draw[:, 0] + xy_draw[:, 1])
		xy_draw[:, 0] = xy_draw[sort_inds, 0] * (max(x) - min(x)) + min(x)
		xy_draw[:, 1] = xy_draw[sort_inds, 1] * (max(y) - min(y)) + min(y)
	elif(points_mode == 'grid'):
		X, Y = np.meshgrid(np.linspace(min(x), max(x), int(np.sqrt(N_draw_points))), \
							 np.linspace(min(y), max(y), int(np.sqrt(N_draw_points))), \
							indexing='xy')
		xy_draw = np.stack((X.flatten(), Y.flatten()), axis=-1)
	
	if(draw_only_inside_convexhull):
		xy_draw = xy_draw[my.in_hull(xy_draw, np.stack((x, y), axis=-1)), :]
		#convx_hull = scipy.spatial.ConvexHull(xy_draw)
	
	z_draw = z_fitfnc(xy_draw[:, 0], xy_draw[:, 1])
	triangulation = matplotlib.tri.Triangulation(xy_draw[:, 0], xy_draw[:, 1])
	
	#ax.plot3D(xy_draw[:, 0], xy_draw[:, 1], z_draw, '.')
	surf = ax.plot_trisurf(xy_draw[:, 0], xy_draw[:, 1], z_draw, label='interp', \
						triangles=triangulation.triangles, \
						color=(0,0,0,0), edgecolor='Gray', lw=0.5)
	surf._facecolors2d = surf._facecolor3d
	surf._edgecolors2d = surf._edgecolor3d
	
	N_OP0_constr = len(OP0_constraint_s)
	if(N_OP0_constr > 0):
		fig_vsX, ax_vsX, _ = my.get_fig(x_lbl, z_lbl, title='%s(%s | $N^*$)' % (z_lbl, x_lbl))
		fig_vsY, ax_vsY, _ = my.get_fig(y_lbl, z_lbl, title='%s(%s | $N^*$)' % (z_lbl, y_lbl))
		for i in range(N_OP0_constr):
			if(OP0consr_Tphi1_sets[i].shape[0] > 0):
				ax.plot3D(OP0consr_Tphi1_sets[i][:, 0], OP0consr_Tphi1_sets[i][:, 1], \
						z_fitfnc(OP0consr_Tphi1_sets[i][:, 0], OP0consr_Tphi1_sets[i][:, 1]), \
						'.', label='$N^* = %d$' % OP0_constraint_s[i], \
						color=my.get_my_color(i+1))
				
				ax_vsX.plot(OP0consr_Tphi1_sets[i][:, 0], \
							z_fitfnc(OP0consr_Tphi1_sets[i][:, 0], OP0consr_Tphi1_sets[i][:, 1]), \
							'.', label='$N^* = %d$' % OP0_constraint_s[i], \
							color=my.get_my_color(i+1))
				
				ax_vsY.plot(OP0consr_Tphi1_sets[i][:, 1], \
							z_fitfnc(OP0consr_Tphi1_sets[i][:, 0], OP0consr_Tphi1_sets[i][:, 1]), \
							'.', label='$N^* = %d$' % OP0_constraint_s[i], \
							color=my.get_my_color(i+1))
		
		ax_vsX.set_xlim([min(x), max(x)])
		ax_vsY.set_xlim([min(y), max(y)])
		
		my.add_legend(fig_vsX, ax_vsX)
		my.add_legend(fig_vsY, ax_vsY)
	
	if(to_add_legend):
		my.add_legend(fig, ax)
	
	return fig, ax

def fit_Tphi1_grid(x, y, z, dz=None, xord=2, yord=2, to_rescale=True, mode='SmoothBivariateSpline'):
	# polynomial_features = sklearn.preprocessing.PolynomialFeatures(degree=degrees[i], include_bias=False)
	# linear_regression = sklearn.linear_model.LinearRegression()
	# pipeline = Pipeline(
		# [
			# ("polynomial_features", polynomial_features),
			# ("linear_regression", linear_regression),
		# ]
	# )
	# pipeline.fit(X[:, np.newaxis], y)
	
	if(dz is None):
		dz = np.ones(z.shape)
	
	xm = np.mean(x) if(to_rescale) else 0
	xs = np.std(x) if(to_rescale) else 1
	ym = np.mean(y) if(to_rescale) else 0
	ys = np.std(y) if(to_rescale) else 1
	if(mode == 'SmoothBivariateSpline'):
		interp = scipy.interpolate.SmoothBivariateSpline(\
						(x - xm) / xs, (y - ym) / ys, \
						z, w=1/dz, kx=xord, ky=yord)
		
		interp_fnc = lambda xin, yin: interp((xin - xm) / xs, (yin - ym) / ys, grid=False)
	elif(mode == 'Rbf'):
		interp = scipy.interpolate.Rbf((x - xm) / xs, (y - ym) / ys, z, smooth=np.std(z)/10)
		interp_fnc = lambda xin, yin: interp((xin - xm) / xs, (yin - ym) / ys)
	
	return interp_fnc, interp
	
def main():
	################################ Ising ############################
	# python run.py -mode BF_AB_L -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 35000000 -N_runs 5 -h 0.15 -interface_set_mode spaced -L 100 71 56 32 24 -to_get_timeevol 0 -OP_max 200 -Temp 1.5

	# python run.py -mode FFS_AB -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.15 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 200 -Temp 1.5
	# python run.py -mode FFS_AB_L -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.15 -interface_set_mode spaced -L 32 24 -to_get_timeevol 0 -OP_max 200 -Temp 1.5
	# python run.py -mode FFS_AB_L -N_states_FFS 100 -N_init_states_FFS 200 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 3 -h 0.15 -interface_set_mode spaced -L 100 78 56 42 32 24 18 -to_get_timeevol 0 -OP_max 200 -Temp 1.5
	# python run.py -mode FFS_AB_L -N_states_FFS 100 -N_init_states_FFS 500 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 5 -h 0.13 -interface_set_mode spaced -L 100 78 56 42 32 24 18 -to_get_timeevol 0 -OP_max 200 -Temp 1.5
	# python run.py -mode FFS_AB_L -N_states_FFS 500 -N_init_states_FFS 1000 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 15 -h 0.15 -interface_set_mode spaced -L 24 31 39 50 56 63 71 79 89 100 -to_get_timeevol 0 -OP_max 200 -Temp 1.5
	#
	# python run.py -mode FFS_AB_L -N_states_FFS 500 -N_init_states_FFS 1000 -N_OP_interfaces 15 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 20 -h 0.15 -interface_set_mode spaced -L 100 64 48 32 -to_get_timeevol 0 -OP_max 210 -Temp 1.5
	###########
	# python run.py -mode FFS_AB_h -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.13 0.12 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 200 -Temp 1.5
	# python run.py -mode FFS_AB_h -N_states_FFS 400 -N_init_states_FFS 800 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 20 -h 0.13 0.12 0.11 0.1 0.09 0.08 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 200 250 300 350 400 500 -Temp 1.5
	#
	# python run.py -mode FFS_AB_mu -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.13 0.12 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 200 250 -J 0.6666666667 -OP_interfaces_set_IDs hJ0.13_2 hJ0.12_2
	# python run.py -mode FFS_AB_mu -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.13 0.12 0.11 0.1 0.09 0.08 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 200 250 300 350 400 500 -J 0.6666666667 -OP_interfaces_set_IDs hJ0.13_2 hJ0.12_2 hJ0.11_2 hJ0.10_2 hJ0.09_2 hJ0.08_2

	###########
	# python run.py -mode FFS_AB_h -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.09 0.08 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 200 -Temp 1.9
	# python run.py -mode FFS_AB_h -N_states_FFS 400 -N_init_states_FFS 800 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 20 -h 0.09 0.08 0.07 0.06 0.05 0.04 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 300 350 400 450 500 600 -Temp 1.9
	###########
	# python run.py -mode FFS_AB_h -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.06 0.04 0.02 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 400 500 600 -Temp 2.0
	# python run.py -mode FFS_AB_h -N_states_FFS 400 -N_init_states_FFS 800 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 20 -h 0.09 0.08 0.07 0.06 0.05 0.04 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 300 350 400 450 500 600 -Temp 2.0
	#
	
	################################ Lattice ###########################
	# python run.py -mode BF_1 -interface_mode CS -Nt 15000 -h 0.13 -L 32 -to_get_timeevol 1 -chi 2.5 1.6 1.6 -to_plot_timeevol 1 -N_spins_up_init 0 -N_saved_states_max 0 -MC_move_mode flip
	# python run.py -mode BF_1 -interface_mode CS -Nt 15000 -h 0.13 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 0 -J 0.66667 -N_saved_states_max 0 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_mu_grid -interface_mode CS -Nt 45000 -mu 2.6 10000 -L 32 -to_get_timeevol 1 -chi 2.5 1.6 1.6 -to_plot_timeevol 1 -N_spins_up_init 0 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_AB -interface_mode CS -Nt 45000 -mu 2.6 10000 -L 32 -to_get_timeevol 1 -chi 2.5 1.6 1.6 -to_plot_timeevol 1 -N_spins_up_init 0 -MC_move_mode flip

	# python run.py -N_saved_states_max 0 -mode BF_AB -interface_mode CS -Nt 15000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 0 -stab_step -5 -mu 5.1 10 -e -2.68 -1.34 -1.715 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_mu_grid -interface_mode CS -Nt 15000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 0 -N_spins_up_init 0 -stab_step -5 -mu 5.1 10 -e -2.68 -1.34 -1.715 -MC_move_mode flip

	# python run.py -N_saved_states_max 0 -mode BF_AB -interface_mode CS -Nt 15000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 0 -stab_step -5 -mu 5.1 3.8 -e -2.68 -1.34 -1.715 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_muOpt -interface_mode CS -Nt 15000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 0 -stab_step -5 -mu 5.1 3.8 -e -2.68 -1.34 -1.715 -MC_move_mode flip

	# python run.py -N_saved_states_max 0 -mode BF_AB -interface_mode CS -Nt 15000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 1024 -stab_step -5 -mu 3.12 5.2 -chi 2.5 1.6 1.6 -Temp 0.8 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_2sides_muOpt -interface_mode CS -Nt 5000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 1024 -stab_step -2 -mu 3.12 5.2 -chi 2.5 1.6 1.6 -Temp 0.8 -N_runs 50 -target_phase_id0 0 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_2sides_muOpt -interface_mode CS -Nt 5000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 1024 -stab_step -2 -mu 3.12 5.2 -chi 2.5 1.6 1.6 -Temp 0.8 -N_runs 2 -target_phase_id0 0 19 -to_plot_target_phase 1 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_2sides_muOpt -interface_mode CS -Nt 5000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 0 -N_spins_up_init 1024 -stab_step -2 -mu 3.12 5.2 -chi 2.5 1.6 1.6 -Temp 0.8 -N_runs 2 -target_phase_id0 0 10 19 20 30 39 40 50 59 60 70 79 80 90 99 100 110 119 120 130 139 140 150 159 160 170 179 180 190 199 200 210 219 220 230 239 240 250 259 260 270 279 280 290 299 -to_plot_target_phase 1 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_2sides_many -interface_mode CS -Nt 5000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 1024 -stab_step -2 -mu 3.12 5.2 -chi 2.5 1.6 1.6 -Temp 0.8 -N_runs 50 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_2sides_many -interface_mode CS -Nt 200000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 1024 -stab_step -2 -mu 2.77096226 2.42018176 -chi 2.5 1.6 1.6 -Temp 0.8954028240792891 -N_runs 10 -MC_move_mode flip

	# python run.py -N_saved_states_max 0 -mode BF_2sides_muOpt -interface_mode CS -Nt 100000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 1024 -stab_step -5 -mu_chi 0 0.4 -chi 2.5 1.6 1.6 -Temp 0.4664 -N_runs 5 -target_phase_id0 0 -to_plot_target_phase 1 -cost_mode 3 -opt_mode 0 -MC_move_mode flip
	# python run.py -N_saved_states_max 0 -mode BF_2sides_many -interface_mode CS -Nt 100000 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -stab_step -5 -mu 5.14511172 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -N_runs 25 -target_phase_id0 0 -to_plot_target_phase 1 -MC_move_mode flip

	# python run.py -mode FFS_AB -N_states_FFS 50 -N_init_states_FFS 100 -N_OP_interfaces 9 -interface_mode CS -OP_0 24 -Nt 1500000 -N_runs 2 -h 0.13 -interface_set_mode spaced -L 32 -to_get_timeevol 0 -OP_max 200 -J 0.6666666667 -OP_interfaces_set_IDs hJ0.13_2 -MC_move_mode flip
	# python run.py -mode FFS_AB -L 32 -OP_interfaces_set_IDs mu1 -to_get_timeevol 0 -N_states_FFS 50 -N_init_states_FFS 100 -N_runs 2 -mu 5.14511172 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode flip

	# python run.py -mode BF_1 -interface_mode CS -Nt 15000 -h 0.13 -L 32 -to_get_timeevol 1 -to_plot_timeevol 1 -N_spins_up_init 0 -J 0.66667 -N_saved_states_max 0 -MC_move_mode flip
	# python run.py -mode FFS_AB -L 32 -OP_interfaces_set_IDs mu1 -to_get_timeevol 0 -N_states_FFS 50 -N_init_states_FFS 100 -N_runs 2 -mu 5.14511172 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap

	# python run.py -mode FFS_AB -L 32 -OP_interfaces_set_IDs mu5 -to_get_timeevol 0 -N_states_FFS 2500 -N_init_states_FFS 5000 -mu 4.9 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode flip
	# python run.py -mode FFS_AB -L 32 -OP_interfaces_set_IDs mu5 -to_get_timeevol 0 -N_states_FFS 250 -N_init_states_FFS 500 -mu 5.15 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode flip -to_recomp 2
	# python run.py -mode FFS_AB -L 128 -to_get_timeevol 0 -N_states_FFS 50 -N_init_states_FFS 100 -N_runs 2 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -init_composition 0.97 0.02 0.01 -OP_interfaces_set_IDs nvt10
	# python run.py -mode BF_1 -Nt 150000 -L 128 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.97 0.02 0.01 -OP_interfaces_set_IDs nvt10 -e -2.68010292 -1.34005146 -1.71526587
	# python run.py -mode BF_1 -Nt 800000000 -L 128 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt-1 -e -2.68010292 -1.34005146 -1.71526587 -OP_min_BF -1 -OP_max_BF 0 -timeevol_stride 16000
	# python run.py -mode FFS_AB -L 256 -to_get_timeevol 0 -N_states_FFS 5 -N_init_states_FFS 10 -N_runs 2 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -init_composition 0.977 0.013 0.01 -OP_interfaces_set_IDs nvt16
	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 15 -N_init_states_FFS 30 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -my_seeds 1 -N_runs 10
	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 250 -N_init_states_FFS 500 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -N_runs 87 -my_seeds 1 2 3 4 6 7 8 9 10 11 13 14 15 16 17 18 19 20 21 24 25 26 27 28 30 31 32 33 34 36 37 38 39 40 41 42 43 44 45 46 47 48 49 51 52 53 54 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 80 81 82 84 85 86 87 89 90 92 94 95 96 97 98 99 100 -to_recomp 1 -font_mode present

	# python run.py -mode FFS_AB_many -L 64 -to_get_timeevol 0 -N_states_FFS 15 -N_init_states_FFS 30 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.9 0.09 0.01 -OP_interfaces_set_IDs nvt18 -N_runs 80 -my_seeds 100 102 103 105 106 107 108 109 110 111 112 113 114 116 117 118 119 120 121 123 126 127 129 130 131 133 134 135 136 139 140 141 142 143 144 145 147 148 149 150 151 152 153 154 155 156 158 161 162 163 165 166 168 169 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 188 189 190 191 192 193 194 195 197 198
	# python run.py -mode FFS_AB_many -L 64 -to_get_timeevol 0 -N_states_FFS 250 -N_init_states_FFS 500 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.92 0.07 0.01 -OP_interfaces_set_IDs nvt18 -N_runs 194 -my_seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 228 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 297 298 299
	# python run.py -mode BF_1 -Nt 150000 -L 128 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt10 -e -2.68010292 -1.34005146 -1.71526587 -R_clust_init 10

	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -N_runs 20 -my_seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 -to_recomp 1 -font_mode present
	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 80 -N_init_states_FFS 160 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -N_runs 280 -my_seeds 100 101 102 103 104 106 107 108 109 110 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 206 207 208 212 213 214 215 216 217 218 219 220 221 222 224 225 226 227 228 229 230 231 232 233 234 235 236 239 240 241 243 244 245 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 292 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 -to_recomp 2 -font_mode present
	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 80 -N_init_states_FFS 160 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -N_runs 3 -my_seeds 105 111 189  -to_recomp 2 -font_mode present

	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 80 -N_init_states_FFS 160 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -N_runs 283 -my_seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 206 207 208 212 213 214 215 216 217 218 219 220 221 222 224 225 226 227 228 229 230 231 232 233 234 235 236 239 240 241 243 244 245 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 292 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 -to_recomp 1 -font_mode present
	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 15 -N_init_states_FFS 30 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.985 0.015 0.00 -OP_interfaces_set_IDs nvt16 -N_runs 246 -my_seeds 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 612 613 614 615 616 617 618 619 620 621 622 623 624 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 -to_recomp 1 -font_mode present
	# python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 15 -N_init_states_FFS 30 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.985 0.015 0.00 -OP_interfaces_set_IDs nvt16 -N_runs 5 -my_seeds 500 501 502 503 504 -to_recomp 1 -font_mode present
	
	# python run.py -mode FFS_AB_Tphi1 -Temp_s 0.8 0.9 1.0 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 0 -font_mode present -OP0_constr_s 15 20 30 50 100
	# python run.py -mode FFS_AB_Tphi1 -Temp_s 0.8 0.9 1.0 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 0 -font_mode present -OP0_constr_s 15 20 30 50 100
	# python run.py -mode FFS_AB_many -init_composition 0.975 0.015 0.01 -Temp 1.0 -N_states_FFS FFS_auto -N_init_states_FFS FFS_auto -font_mode present -MC_move_mode swap -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -OP_interfaces_set_IDs nvt -my_seeds FFS_auto -to_recomp 1
	
	# TODO: run failed IDs with longer times
	
	[                                           L,    potential_filenames,      mode,           Nt,     N_states_FFS,     N_init_states_FFS,         to_recomp,     to_get_timeevol,     verbose,     my_seeds,     N_OP_interfaces,     N_runs,     init_gen_mode,     OP_0,     OP_max,     interface_mode,     OP_min_BF,     OP_max_BF,     Nt_sample_A,     Nt_sample_B,      N_spins_up_init,       to_plot_ETS,     interface_set_mode,     timeevol_stride,     to_plot_timeevol,     N_saved_states_max,       J,       h,     OP_interfaces_set_IDs,     chi,      mu,       e,     stab_step,    Temp,    mu_chi,     to_plot_target_phase,     target_phase_id0,     target_phase_id1,     cost_mode,     opt_mode,     MC_move_mode,           init_composition,     to_show_on_screen,         to_save_npz,     R_clust_init,        to_animate,     font_mode,     N_fourier,     Temp_s,     phi1_s,      phi2,     OP0_constr_s,     N_ID_groups,       to_post_proc,     Dtop_Nruns], _ = \
		my.parse_args(sys.argv,            [ '-L', '-potential_filenames',   '-mode',        '-Nt',  '-N_states_FFS',  '-N_init_states_FFS',      '-to_recomp',  '-to_get_timeevol',  '-verbose',  '-my_seeds',  '-N_OP_interfaces',  '-N_runs',  '-init_gen_mode',  '-OP_0',  '-OP_max',  '-interface_mode',  '-OP_min_BF',  '-OP_max_BF',  '-Nt_sample_A',  '-Nt_sample_B',   '-N_spins_up_init',    '-to_plot_ETS',  '-interface_set_mode',  '-timeevol_stride',  '-to_plot_timeevol',  '-N_saved_states_max',    '-J',    '-h',  '-OP_interfaces_set_IDs',  '-chi',   '-mu',    '-e',  '-stab_step', '-Temp', '-mu_chi',  '-to_plot_target_phase',  '-target_phase_id0',  '-target_phase_id1',  '-cost_mode',  '-opt_mode',  '-MC_move_mode',        '-init_composition',  '-to_show_on_screen',      '-to_save_npz',  '-R_clust_init',     '-to_animate',  '-font_mode',  '-N_fourier',  '-Temp_s',  '-phi1_s',   '-phi2',  '-OP0_constr_s',  '-N_ID_groups',    '-to_post_proc',  '-Dtop_Nruns'], \
					  possible_arg_numbers=[['+'],                   None,       [1],       [0, 1],           [0, 1],                [0, 1],            [0, 1],              [0, 1],      [0, 1],         None,              [0, 1],     [0, 1],            [0, 1],     None,       None,             [0, 1],        [0, 1],        [0, 1],          [0, 1],          [0, 1],               [0, 1],            [0, 1],                 [0, 1],              [0, 1],               [0, 1],                 [0, 1],  [0, 1],    None,                      None,  [0, 3],    None,  [0, 3],        [0, 1],  [0, 1],      None,                   [0, 1],                 None,                 None,        [0, 1],       [0, 1],           [0, 1],                     [0, 3],                [0, 1],              [0, 1],           [0, 1],            [0, 1],        [0, 1],        [0, 1],       None,       None,    [0, 1],             None,          [0, 1],             [0, 1],         [0, 1]], \
					  default_values=      [ None,                 [None],      None, ['-1000000'],        ['-5000'],          ['FFS_auto'],             ['0'],               ['1'],       ['1'],       ['23'],              [None],     ['-1'],            ['-3'],    ['1'],     [None],             ['CS'],        [None],        [None],    ['-1000000'],    ['-1000000'],               [None],  [my.no_flags[0]],            [ 'spaced'],           ['-3000'],     [my.no_flags[0]],               ['1000'],  [None],  [None],                    [None],  [None],  [None],  [None],        ['-1'],   ['1'],    [None],         [my.no_flags[0]],                ['0'],               [None],         ['2'],        ['2'],             None,   ['0.97', '0.02', '0.01'],      [my.yes_flags[0]],  [my.yes_flags[0]],           [None],  [my.no_flags[0]],      ['work'],         ['5'],    ['1.0'],  ['0.015'],  ['0.01'],           [None],          [None],  [my.yes_flags[0]],          ['0']])
	
	Ls = np.array([int(l) for l in L], dtype=int)
	N_L = len(Ls)
	L = Ls[0]
	L2 = L**2
	
	MC_move_mode_name = MC_move_mode[0]
	MC_move_mode = move_modes[MC_move_mode_name]
	init_composition = np.array([float(ic) for ic in init_composition])
	swap_type_move = (MC_move_mode in [move_modes['swap'], move_modes['long_swap']])
	
	# ================ interaction setup ==============
	e_inputs = ~np.array([J[0] is None, chi[0] is None, potential_filenames[0] is None, e[0] is None])
	e_inputs_strs = ['J', 'chi', 'file', 'e']
	assert(np.sum(e_inputs) == 1), 'ERROR: possible e-inputs are %s, but provided are: %s. Aborting' % (str(e_inputs_strs), str(e_inputs))
	e_new = np.zeros((N_species, N_species))
	if(potential_filenames[0] is not None):
		pass
		# load potentials
	elif(J[0] is not None):
		assert(h[0] is not None), 'ERROR: J is given but no "h". Aborting'
		N_param_points = len(h)
		
		e_new[1, 1] = -4 * float(J[0])
	elif(chi[0] is not None):
		kk = 0
		for i in range(N_species - 1):
			for j in range(i + 1, N_species):
				e_new[i, j] = float(chi[kk]) * (1 / z_neib)
				kk += 1
				e_new[j, i] = e_new[i, j]
	elif(e[0] is not None):
		e_new[1,1] = float(e[0])
		e_new[1,2] = float(e[1])
		e_new[2,2] = float(e[2])
		e_new[2,1] = e_new[1,2]
	
	e = np.copy(e_new)
	for i in range(N_species):
		for j in range(N_species):
			e[i, j] = (e_new[i, j] + e_new[0,0] - e_new[0, i] - e_new[0, j])
	
	if(MC_move_mode in [move_modes['swap'], move_modes['long_swap']]):
		N_param_points = 1
		mu = np.zeros((N_param_points, N_species))
	else:
		mu_inputs = ~np.array([h[0] is None, mu[0] is None, mu_chi[0] is None])   # , potential_filenames[0] is None
		mu_inputs_strs = ['h', 'mu', 'mu_chi']    # , 'file'
		assert(np.sum(mu_inputs) == 1), 'ERROR: possible mu-inputs are %s, but provided are: %s. Aborting' % (str(mu_inputs_strs), str(mu_inputs))
		if(h[0] is not None):
			N_param_points = len(h)
			mu = np.zeros((N_param_points, N_species))
			mu[:, 1] = (2 * np.array([-float(hh) for hh in h]) - z_neib * e[1, 1] / 2)
			mu[:, 2] = 10
		elif(mu[0] is not None):
			N_param_points = len(mu) // (N_species - 1)
			assert(N_param_points * (N_species - 1) == len(mu)), 'ERROR: number of mu-s given is not divisible by %d. Aborting' % (N_species - 1)
			mu = np.array([([0] + [float(mu[i * (N_species - 1) + j]) for j in range(N_species - 1)]) for i in range(N_param_points)])   # N x N_species; mu[:, 0] = 0
		elif(mu_chi[0] is not None):
			N_param_points = len(mu_chi) // (N_species - 1)
			assert(N_param_points * (N_species - 1) == len(mu_chi)), 'ERROR: number of mu_chi-s given is not divisible by %d. Aborting' % (N_species - 1)
			mu = np.array([([e[0, 0] * z_neib / 2] + [(float(mu_chi[i * (N_species - 1) + j]) - e[j+1, j+1] * z_neib / 2) for j in range(N_species - 1)]) for i in range(N_param_points)])   # N x N_species; mu[:, 0] = 0
	
	mu_orig = np.copy(mu)
	for i in range(N_species):
		mu[:, i] = (mu_orig[:, i] - mu_orig[:, 0] + z_neib * (e[0, i] - e[0,0]))
	
	Temp = float(Temp[0])
	e /= Temp
	mu /= Temp
	
	N_mu = mu.shape[0]
	N_param_points = N_L if('L' in mode) else (N_mu if('mu' in mode) else 1)
	if(OP_interfaces_set_IDs[0] is not None):
		if((N_param_points > 1) and (len(OP_interfaces_set_IDs) == 1)):
			OP_interfaces_set_IDs = [OP_interfaces_set_IDs[0]] * N_param_points
	
	# ================ other params ==============
	set_OP_defaults(L2)
	
	to_recomp = int(to_recomp[0])
	#mode = mode[0]
	to_get_timeevol = (to_get_timeevol[0] in my.yes_flags)
	verbose = int(verbose[0])
	Nt = int(Nt[0])
	N_states_FFS = table_data.nPlot_FFS_dict[MC_move_mode_name][my.f2s(Temp, n=2)][my.f2s(init_composition[1], n=5)] if(N_states_FFS[0] == 'FFS_auto') else int(N_states_FFS[0])
	N_init_states_FFS = (2 * N_states_FFS) if(N_init_states_FFS[0] == 'FFS_auto') else int(N_init_states_FFS[0])
	N_runs = int(N_runs[0])
	init_gen_mode = int(init_gen_mode[0])
	interface_mode = interface_mode[0]
	N_spins_up_init = (None if(N_spins_up_init[0] is None) else int(N_spins_up_init[0]))
	to_plot_ETS = (to_plot_ETS[0] in my.yes_flags)
	interface_set_mode = interface_set_mode[0]
	timeevol_stride = int(timeevol_stride[0])
	to_plot_timeevol = (to_plot_timeevol[0] in my.yes_flags)
	N_saved_states_max = int(N_saved_states_max[0])
	stab_step = int(stab_step[0])
	to_plot_target_phase = (to_plot_target_phase[0] in my.yes_flags)
	target_phase_id0 = np.array([int(i) for i in target_phase_id0])
	target_phase_id1 = np.copy(target_phase_id0) if(target_phase_id1[0] is None) else np.array([int(i) for i in target_phase_id1])
	assert(len(target_phase_id0) == len(target_phase_id1))
	cost_mode = int(cost_mode[0])
	opt_mode = int(opt_mode[0])
	to_show_on_screen = (to_show_on_screen[0] in my.yes_flags)
	to_save_npz = (to_save_npz[0] in my.yes_flags)
	R_clust_init = None if(R_clust_init[0] is None) else float(R_clust_init[0])
	to_animate = (to_animate[0] in my.yes_flags)
	my.font_mode = font_mode[0]
	N_fourier = int(N_fourier[0])
	Temp_s = np.array([float(xx) for xx in Temp_s])
	phi1_s = np.array([float(xx) for xx in phi1_s])
	phi2 = float(phi2[0])
	OP0_constr_s = None if(OP0_constr_s[0] is None) else np.array(OP0_constr_s, dtype=int)
	N_ID_groups = None if(N_ID_groups[0] is None) else int(N_ID_groups[0])
	to_post_proc = (to_post_proc[0] in my.yes_flags)
	Dtop_Nruns = int(Dtop_Nruns[0])
	
	assert(interface_mode == 'CS'), 'ERROR: only CS (not M) mode is supported'
	
	OP_min_BF = OP_min_default[interface_mode] if(OP_min_BF[0] is None) else int(OP_min_BF[0])
	OP_max_BF = OP_max_default[interface_mode] if(OP_max_BF[0] is None) else int(OP_max_BF[0])
	
	if(OP_interfaces_set_IDs[0] is None):
		if(OP_0[0] is None):
			OP_0[0] = OP_min_BF
		if(OP_max[0] is None):
			OP_max[0] = OP_max_BF
	else:
		OP_0[0] = table_data.OP_interfaces_table[OP_interfaces_set_IDs[0]][0]
		OP_max[0] = table_data.OP_interfaces_table[OP_interfaces_set_IDs[0]][-1]
		if(OP_min_BF == 0):
			OP_min_BF = OP_0[0]
		if(OP_max_BF == 0):
			OP_max_BF = OP_max[0]
	
	OP_0 = np.array(([int(OP_0[0])] * N_param_points) if(len(OP_0) == 1) else [int(x) for x in OP_0], dtype=int)
	OP_max = np.array(([int(OP_max[0])] * N_param_points) if(len(OP_max) == 1) else [int(x) for x in OP_max], dtype=int)
	OP_0_handy = np.copy(OP_0)
	OP_max_handy = np.copy(OP_max)
	if(interface_mode == 'M'):
		OP_0 = OP_0 - L2
		OP_max = OP_max - L2
	
	assert(interface_mode == 'CS'), 'ERROR: interface_mode==M is temporary not supported'
	
	lattice_gas.set_verbose(verbose)
	
	dE_avg = ((np.sqrt(abs(e[1,1])) + np.sqrt(abs(e[2,2]))) / 2)**2
	Nt, N_states_FFS = \
		tuple([(int(-n * np.exp(3 - dE_avg) + 1) if(n < 0) else n) for n in [Nt, N_states_FFS]])
	
	BC_cluster_size = L2 / np.pi
	
	if(stab_step < 0):
		stab_step = int(min(L2, Nt / 2) * (-stab_step))
	if(timeevol_stride < 0):
		timeevol_stride = np.int_(- Nt / timeevol_stride)
	#print(Nt, timeevol_stride)
	#input(str(timeevol_stride))
	
	if('2sides' in mode):
		target_states0_all = load_Q_target_phase_data('bc').T
		target_states1_all = load_Q_target_phase_data('nc').T
		
		# x0 = [0.8, 3.12, 5.2]
		# (id0, id1) : T, m1/T, m2/T
		# (  0,  0) :  [0.77928023, 3.18452509, 5.16587932]
		# ( 19, 19) :  [0.81041512, 3.02352679, 5.20396601]
		
		target_states0 = np.array([target_states0_all[i, :] for i in target_phase_id0])
		target_states1 = np.array([target_states1_all[i, :] for i in target_phase_id1])
	
	if('FFS' in mode):
		N_OP_interfaces = int(N_OP_interfaces[0]) if(OP_interfaces_set_IDs is None) else len(table_data.OP_interfaces_table[OP_interfaces_set_IDs[0]])
		
		assert(not(('AB' not in mode) and (interface_mode == 'CS'))), 'ERROR: BA transitions is not supported for CS'
		OP_left = OP_min_default[interface_mode]
		OP_right = OP_max_default[interface_mode] - 1
		
		UIDs = []
		
		OP_interfaces_AB = []
		OP_match_BF_A_to = []
		OP_sample_BF_A_to = []
		for i in range(N_param_points):
			if(N_OP_interfaces > 0):
				if(OP_interfaces_set_IDs is None):
					L_local = Ls[i if('L') in mode else 0]
					mu_local_1 = mu[i if('mu' in mode) else 0, 1]
					mu_local_2 = mu[i if('mu' in mode) else 0, 2]
					UID = (interface_mode, e[1,1], e[1,2], e[2,2], L_local, mu_local_1, mu_local_2, N_OP_interfaces, OP_0_handy[i], OP_max_handy[i], interface_set_mode)
				else:
					UID = OP_interfaces_set_IDs[i]
				UIDs.append(UID)
				
				if(UIDs[i] in table_data.OP_interfaces_table):
					OP_interfaces_AB.append(table_data.OP_interfaces_table[UIDs[i]])
					assert(N_OP_interfaces == len(OP_interfaces_AB[i])), 'ERROR: OP_interfaces number of elements does not match N_OP_interfaces'
				else:
					assert(input('WARNING: no custom interfaces found for %s, using equly-spaced interfaces, proceed? (y/N)\n' % (str(UIDs[i]))) in my.yes_flags), 'exiting'
				
				if(interface_mode == 'M'):
					OP_interfaces_AB[i] = OP_interfaces_AB[i] - L_local**2
			else:
				# =========== if the necessary set in not in the table - this will work and nothing will erase its results ========
				OP_interfaces_AB.append(OP_0[i] + np.round(np.arange(abs(N_OP_interfaces)) * (OP_max[i] - OP_0[i]) / (abs(N_OP_interfaces) - 1) / OP_step[interface_mode]) * OP_step[interface_mode])
					# this gives Ms such that there are always pairs +-M[i], so flipping this does not move the intefraces, which is (is it?) good for backwords FFS (B->A)
			
			OP_match_BF_A_to.append(OP_interfaces_AB[i][1])
			OP_sample_BF_A_to.append(OP_interfaces_AB[i][2])
		
		N_init_states_AB = np.ones(N_OP_interfaces, dtype=np.intc) * N_states_FFS
		N_init_states_AB[0] = N_init_states_FFS
		
		if(my_seeds[0] == 'FFS_auto'):
			FFS_npz_template_filename, FFS_npz_template_filename_old = \
					izing.get_FFS_AB_npzTrajBasename(MC_move_mode, L, e, \
									'_phi' + my.join_lbls(init_composition[1:], '_'), \
									OP_interfaces_AB[-1], N_init_states_AB, \
									Ls[0]**2, OP_sample_BF_A_to[0], OP_match_BF_A_to[0], \
									timeevol_stride, init_gen_mode, \
									to_get_timeevol, \
									N_fourier, '%d')
			FFS_npz_template_filepath = '/scratch/gpfs/yp1065/Izing/npzs/' + FFS_npz_template_filename + '_FFStraj.npz'
			my_seeds, found_filepaths, _ = \
				izing.find_finished_runs_npzs(FFS_npz_template_filepath, np.arange(1000, 1200), verbose=True)
		
		print('N_states_FFS=%d, N_OP_interfaces=%d, interface_mode=%s\nOP_AB: %s\nN_states_FFS: %s' % \
			(N_states_FFS, N_OP_interfaces, interface_mode, str(OP_interfaces_AB), str(N_init_states_AB)))
	
	if(my_seeds not in ['FFS_auto']):
		my_seeds = np.array(my_seeds, dtype=int)
	
	my_seed = my_seeds[0]
	lattice_gas.init_rand(my_seed)
	np.random.seed(my_seed)
	#input(str(lattice_gas.get_seed()))
	if(N_runs == -1):
		N_runs = len(my_seeds)
	if(('compare' in mode) or ('many' in mode)):
		N_k_bins = int(np.round(np.sqrt(N_runs) / 2) + 1)
	
	print('Ls=%s, mode=%s, Nt=%d, N_runs=%d, init_gen_mode=%d, OP_0=%s, OP_max=%s, N_spins_up_init=%s, OP_min_BF=%d, OP_max_BF=%d, to_get_timeevol=%r, verbose=%d, my_seed=%d, to_recomp=%d, N_param_points=%d\ne: %s\nmu: %s\n' % \
		(str(Ls), mode, Nt, N_runs, init_gen_mode, str(OP_0), str(OP_max), str(N_spins_up_init), OP_min_BF, OP_max_BF, to_get_timeevol, verbose, my_seed, to_recomp, N_param_points, str(e), str(mu)))
	
	if(mode == 'BF_1'):
		proc_T(MC_move_mode, Ls[0], e, mu[0, :], Nt, interface_mode, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_timeevol=to_plot_timeevol, to_plot_F=True, to_plot_ETS=to_plot_ETS, \
				to_plot_correlations=True and to_plot_timeevol, to_get_timeevol=True, \
				OP_min=OP_min_BF, OP_max=OP_max_BF, to_estimate_k=True, \
				timeevol_stride=timeevol_stride, N_saved_states_max=N_saved_states_max, \
				stab_step=stab_step, init_composition=init_composition, \
				R_clust_init=R_clust_init, to_animate=to_animate, \
				to_save_npz=to_save_npz, to_recomp=to_recomp)
	
	elif(mode == 'BF_mu_grid'):
		mu1_vec = np.linspace(3, 3.3, 16)   # t=0.8
		mu2_vec = np.linspace(4, 6, 21)
		
		map_phase_difference(MC_move_mode, Ls[0], e, [np.array(mu1_vec), np.array(mu2_vec)], \
							Nt, interface_mode, \
							to_plot_timeevol=to_plot_timeevol, \
							stab_step=stab_step, \
							to_plot_debug=False)
	
	elif(mode == 'BF_2sides'):
		err = cost_fnc(MC_move_mode, Ls[0], e, mu[0, :], Nt, interface_mode,
					target_states0[target_phase_id0, :], target_states1[target_phase_id1, :], \
					timeevol_stride, stab_step, verbose=verbose, \
					to_plot_timeevol=True, seeds=my_seeds)
	
	elif(mode == 'BF_2sides_many'):
		phi_0, d_phi_0, phi_1, d_phi_1, \
			phi_0_evol, d_phi_0_evol, phi_1_evol, d_phi_1_evol= \
				run_many(MC_move_mode, Ls[0], e, mu[0, :], N_runs, interface_mode, \
					Nt_per_BF_run=Nt, stab_step=stab_step, \
					mode='BF_2sides', timeevol_stride=timeevol_stride, \
					to_plot_time_evol=False, to_recomp=to_recomp, \
					to_save_npz=to_save_npz, verbose=verbose, seeds=my_seeds)
		
		if(to_plot_timeevol):
			plot_avg_phi_evol(phi_0, d_phi_0, phi_0_evol, d_phi_0_evol, phi_0, stab_step, timeevol_stride, r'$\vec{\varphi}_0$', time_scl='linear')    #  * np.sqrt(N_runs - 1)
			plot_avg_phi_evol(phi_1, d_phi_0, phi_1_evol, d_phi_1_evol, phi_1, stab_step, timeevol_stride, r'$\vec{\varphi}_1$', time_scl='linear')    #  * np.sqrt(N_runs - 1)
		
		if(to_plot_target_phase):
			plot_Q_target_phase_data(target_states0_all, target_phase_id0, phi_0, d_phi_0, '0', cost_mode)
			plot_Q_target_phase_data(target_states1_all, target_phase_id1, phi_1, d_phi_1, '1', cost_mode)
	
	elif(mode == 'BF_2sides_muOpt'):
		N_targets = target_states0.shape[0]
		phi_0 = np.empty((N_targets, N_species))
		d_phi_0 = np.empty((N_targets, N_species))
		phi_1 = np.empty((N_targets, N_species))
		d_phi_1 = np.empty((N_targets, N_species))
		Nt_states = (Nt*10) // timeevol_stride
		phi_0_evol = np.empty((N_targets, N_species, Nt_states))
		d_phi_0_evol = np.empty((N_targets, N_species, Nt_states))
		phi_1_evol = np.empty((N_targets, N_species, Nt_states))
		d_phi_1_evol = np.empty((N_targets, N_species, Nt_states))
		
		Temp_opt_arr = np.empty(N_targets)
		mu_opt_arr = np.empty((2, N_targets))
		for i in range(N_targets):
			x_opt = find_optimal_mu(MC_move_mode, Ls[0], e, mu[0, :], Temp, \
						target_states0[i, :], target_states1[i, :], \
						Nt, interface_mode, N_runs, cost_mode, \
						stab_step_FFS=stab_step, seeds=my_seeds,
						opt_mode=opt_mode, verbose=verbose)
			
			if(opt_mode == 0):
				e_opt = e
				mu_opt = np.array([0] + list(x_opt.x))
			elif(opt_mode == 1):
				e_opt = eFull_from_eCut(x_opt.x[:3])
				mu_opt = np.array([0] + list(x_opt.x[3:]))
			elif(opt_mode == 2):
				Temp_opt_arr[i] = x_opt.x[0]
				mu_opt_arr[:, i] = x_opt.x[1:] / Temp_opt_arr[i]
				e_opt = e
				mu_opt = np.array([0] + list(mu_opt_arr[:, i]))
			
			phi_0[i, :], d_phi_0[i, :], phi_1[i, :], d_phi_1[i, :], \
				phi_0_evol[i, :, :], d_phi_0_evol[i, :, :], \
				phi_1_evol[i, :, :], d_phi_1_evol[i, :, :] = \
					run_many(MC_move_mode, L, e_opt, mu_opt, N_runs * 10, interface_mode, \
						Nt_per_BF_run=Nt_states * timeevol_stride, stab_step=stab_step, \
						mode='BF_2sides', timeevol_stride=timeevol_stride, \
						to_save_npz=to_save_npz, to_recomp=to_recomp, \
						to_plot_time_evol=False, verbose=max(verbose - 1, 0), \
						seeds=my_seeds)
			
			if(to_plot_timeevol):
				states_timesteps = np.arange(Nt_states) * timeevol_stride
				metastab_inds = (states_timesteps > stab_step) & (states_timesteps < (stab_step + 5 * L2))
				N_metastable_states = np.sum(metastab_inds)
				for j in range(N_species):
					phi_0[i, j] = np.mean(phi_0_evol[i, j, metastab_inds])
					d_phi_0[i, j] = np.std(phi_0_evol[i, j, metastab_inds])
					phi_1[i, j] = np.mean(phi_1_evol[i, j, metastab_inds])
					d_phi_1[i, j] = np.std(phi_1_evol[i, j, metastab_inds])
				plot_avg_phi_evol(phi_0[i, :], d_phi_0[i, :], phi_0_evol[i, :, :], d_phi_0_evol[i, :, :], target_states0[i, :], stab_step, timeevol_stride, r'$\vec{\varphi}_0$')
				plot_avg_phi_evol(phi_1[i, :], d_phi_1[i, :], phi_1_evol[i, :, :], d_phi_1_evol[i, :, :], target_states1[i, :], stab_step, timeevol_stride, r'$\vec{\varphi}_1$')
		
		TMuOpt_filename = 'TMuOpt_stab2000_time4000.npz'
		np.savez(TMuOpt_filename, Temp_opt_arr=Temp_opt_arr, mu_opt_arr=mu_opt_arr)
		print('%s written' % TMuOpt_filename)
		
		if(to_plot_target_phase):
			plot_Q_target_phase_data(target_states0_all, target_phase_id0, phi_0, d_phi_0, '0', cost_mode)
			plot_Q_target_phase_data(target_states1_all, target_phase_id1, phi_1, d_phi_1, '1', cost_mode)
			#plt.show()
	
	elif(mode == 'BF_AB'):
		proc_T(MC_move_mode, Ls[0], e, mu[0, :], Nt, interface_mode, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_timeevol=to_plot_timeevol, to_plot_F=True, to_plot_ETS=to_plot_ETS, \
				to_plot_correlations=True and to_plot_timeevol, to_get_timeevol=to_get_timeevol, \
				OP_min=OP_min_BF, OP_max=OP_max[0], to_estimate_k=False, \
				timeevol_stride=timeevol_stride, N_saved_states_max=N_saved_states_max, \
				stab_step=stab_step, init_composition=init_composition, \
				R_clust_init=R_clust_init, \
				to_save_npz=to_save_npz, to_recomp=to_recomp)
	
	elif(mode == 'BF_many'):
		run_many(MC_move_mode, Ls[0], e, mu[0, :], N_runs, interface_mode, Nt_per_BF_run=Nt, \
				stab_step=stab_step, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, \
				mode='BF', N_saved_states_max=N_saved_states_max,
				init_composition=init_composition, \
				R_clust_init=R_clust_init, \
				to_save_npz=to_save_npz, to_recomp=to_recomp, \
				seeds=my_seeds)
	
	elif(mode == 'BF_AB_many'):
		run_many(MC_move_mode, Ls[0], e, mu[0, :], N_runs, interface_mode,
				Nt_per_BF_run=Nt, stab_step=stab_step, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, \
				mode='BF_AB', N_saved_states_max=N_saved_states_max, \
				init_composition=init_composition, \
				R_clust_init=R_clust_init, \
				to_save_npz=to_save_npz, to_recomp=to_recomp, \
				seeds=my_seeds)
	
	elif(mode == 'FFS_AB'):
		proc_FFS_AB(MC_move_mode, Ls[0], e, mu[0, :], N_init_states_AB, \
					OP_interfaces_AB[0], interface_mode, \
					stab_step=stab_step, \
					OP_sample_BF_to=OP_sample_BF_A_to[0], \
					OP_match_BF_to=OP_match_BF_A_to[0], \
					timeevol_stride=timeevol_stride, \
					to_get_timeevol=to_get_timeevol, \
					init_gen_mode=init_gen_mode, \
					init_composition=init_composition, \
					Dtop_Nruns=Dtop_Nruns, \
					to_recomp=to_recomp, \
					to_save_npz=to_save_npz, \
					to_plot_time_evol=to_plot_timeevol, \
					to_plot_hists=True, \
					to_plot=True, \
					N_fourier=N_fourier, \
					to_post_proc=to_post_proc)
	
	elif(mode == 'FFS_AB_many'):
		F_FFS, d_F_FFS, M_hist_centers_FFS, OP_hist_lens_FFS, ln_k_AB_FFS, d_ln_k_AB_FFS, \
			flux0_AB_FFS, d_flux0_AB_FFS, prob_AB_FFS, d_prob_AB_FFS, PB_AB_FFS, d_PB_AB_FFS, \
			OP0_AB, d_OP0_AB, ZeldovichG_AB, d_ZeldovichG_AB, chi2_rho = \
				run_many(MC_move_mode, Ls[0], e, mu[0, :], N_runs, interface_mode, \
					stab_step=stab_step, \
					N_init_states_AB=N_init_states_AB, \
					OP_interfaces_AB=OP_interfaces_AB[0], \
					to_plot_k_distr=False, N_k_bins=N_k_bins, \
					mode='FFS_AB', init_gen_mode=init_gen_mode, \
					OP_sample_BF_A_to=OP_sample_BF_A_to[0], \
					OP_match_BF_A_to=OP_match_BF_A_to[0], \
					timeevol_stride=timeevol_stride, \
					to_get_timeevol=to_get_timeevol, \
					init_composition=init_composition, \
					to_save_npz=to_save_npz, to_recomp=to_recomp, \
					N_fourier=N_fourier, seeds=my_seeds)
		
		get_log_errors(np.log(flux0_AB_FFS), d_flux0_AB_FFS / flux0_AB_FFS, lbl='flux0_AB', print_scale=print_scale_flux)
		k_AB_FFS_mean, k_AB_FFS_min, k_AB_FFS_max, d_k_AB_FFS = get_log_errors(ln_k_AB_FFS, d_ln_k_AB_FFS, lbl='k_AB_FFS', print_scale=print_scale_k)
		print('N* =', my.errorbar_str(OP0_AB, d_OP0_AB))
		print('ZeldovichG =', my.errorbar_str(ZeldovichG_AB, d_ZeldovichG_AB))
	
	elif('mu' in mode):
		get_mu_dependence(MC_move_mode, mu, Ls[0], e, N_runs, interface_mode, \
						N_init_states_AB, OP_interfaces_AB, init_gen_mode=init_gen_mode, \
						stab_step=stab_step, to_recomp=to_recomp, \
						to_plot=True, to_save_npy=True, seeds=my_seeds)
	
	elif('Tphi1' in mode):
		get_Tphi1_dependence(Temp_s, phi1_s, phi2, \
							MC_move_mode_name, Ls[0], e, interface_mode, \
							OP_interfaces=OP_interfaces_AB[0], \
							init_gen_mode=init_gen_mode, \
							OP_sample_BF_to=OP_sample_BF_A_to[0], \
							OP_match_BF_to=OP_match_BF_A_to[0], \
							timeevol_stride=timeevol_stride if(to_get_timeevol) else None, \
							to_get_timeevol=to_get_timeevol, \
							to_save_npz=to_save_npz, \
							to_recomp=to_recomp, \
							N_fourier=N_fourier, \
							OP0_constraint_s=OP0_constr_s, \
							N_ID_groups=N_ID_groups, \
							to_plot=True, \
							seeds_range=np.arange(1000, 1200))
					
	
	elif('L' in mode):
		ln_k_AB = np.empty(N_L)
		d_ln_k_AB = np.empty(N_L)
		k_AB_mean = np.empty(N_L)
		k_AB_min = np.empty(N_L)
		k_AB_max = np.empty(N_L)
		d_k_AB = np.empty(N_L)
		k_AB_BFcount_N_mean = np.empty(N_L)
		k_AB_BFcount_N_min = np.empty(N_L)
		k_AB_BFcount_N_max = np.empty(N_L)
		d_k_AB_BFcount_N = np.empty(N_L)
		F_FFS = []
		d_F_FFS = []
		OP_hist_centers_FFS = []
		OP_hist_lens_FFS = []
		if('BF' in mode):
			k_AB_BFcount_N = np.empty(N_L)
			d_k_AB_BFcount_N = np.empty(N_L)
		if('FFS' in mode):
			flux0_AB_FFS = np.empty(N_L)
			d_flux0_AB_FFS = np.empty(N_L)
			prob_AB_FFS = np.empty((N_OP_interfaces - 1, N_L))
			d_prob_AB_FFS = np.empty((N_OP_interfaces - 1, N_L))
			PB_AB_FFS = np.empty((N_OP_interfaces, N_L))
			d_PB_AB_FFS = np.empty((N_OP_interfaces, N_L))
			OP0_AB = np.empty(N_L)
			d_OP0_AB = np.empty(N_L)
			
		for i_l in range(N_L):
			set_OP_defaults(Ls[i_l]**2)
			
			if('BF' in mode):
				F_FFS_new, d_F_FFS_new, OP_hist_centers_new, OP_hist_lens_new, \
					ln_k_AB[i_l], d_ln_k_AB[i_l], k_AB_BFcount_N[i_l], d_k_AB_BFcount_N[i_l] = \
						run_many(MC_move_mode, Ls[i_l], e, mu[0, :], \
								N_runs, interface_mode, Nt_per_BF_run=Nt, \
								stab_step=stab_step, \
								OP_A=OP_0[i_l], OP_B=OP_max[i_l], \
								N_spins_up_init=N_spins_up_init, \
								to_get_timeevol=to_get_timeevol, mode='BF_AB', \
								N_saved_states_max=N_saved_states_max, \
								init_composition=init_composition, \
								R_clust_init=R_clust_init, \
								to_save_npz=to_save_npz, to_recomp=to_recomp, \
								seeds=my_seeds)
			elif('FFS' in mode):
				F_FFS_new, d_F_FFS_new, OP_hist_centers_FFS_new, OP_hist_lens_FFS_new, \
					ln_k_AB[i_l], d_ln_k_AB[i_l], \
					flux0_AB_FFS[i_l], d_flux0_AB_FFS[i_l], \
					prob_AB_FFS[:, i_l], d_prob_AB_FFS[:, i_l], \
					PB_AB_FFS[:, i_l], d_PB_AB_FFS[:, i_l], \
					OP0_AB[i_l], d_OP0_AB[i_l], _, _, _ = \
						run_many(MC_move_mode, Ls[i_l], e, mu[0, :], N_runs, interface_mode, \
							stab_step=stab_step, \
							N_init_states_AB=N_init_states_AB, \
							OP_interfaces_AB=OP_interfaces_AB[i_l], \
							mode='FFS_AB', init_gen_mode=init_gen_mode, \
							OP_sample_BF_A_to=OP_sample_BF_A_to[i_l], \
							OP_match_BF_A_to=OP_match_BF_A_to[i_l], \
							init_composition=init_composition, \
							to_get_timeevol=to_get_timeevol, \
							to_save_npz=to_save_npz, to_recomp=to_recomp, \
							to_plot_committer=False, seeds=my_seeds, \
							N_fourier=N_fourier)
				# ZeldG, d_ZeldG, chi2_rho are ignored
				
				F_FFS.append(F_FFS_new)
				d_F_FFS.append(d_F_FFS_new)
				OP_hist_centers_FFS.append(OP_hist_centers_FFS_new)
				OP_hist_lens_FFS.append(OP_hist_lens_FFS_new)
				print(my.errorbar_str(OP0_AB[i_l], d_OP0_AB[i_l]))
			
			k_AB_mean[i_l], k_AB_min[i_l], k_AB_max[i_l], d_k_AB[i_l] = \
				get_log_errors(ln_k_AB[i_l], d_ln_k_AB[i_l], lbl='k_AB_L' + str(Ls[i_l]), print_scale=print_scale_k)
			
			#k_AB_BFcount_N_mean[i_l], k_AB_BFcount_N_min[i_l], k_AB_BFcount_N_max[i_l], d_k_AB_BFcount_N[i_l] = \
			#	get_log_errors(np.log(k_AB_BFcount_N[i_l]), d_k_AB_BFcount_N[i_l] / k_AB_BFcount_N[i_l], lbl='N_AB_L' + str(Ls[i_l]))
		
		if('FFS' in mode):
			P_AB = k_AB_mean / flux0_AB_FFS
			d_P_AB = P_AB * np.sqrt(d_ln_k_AB**2 + (d_flux0_AB_FFS / flux0_AB_FFS)**2)
			PB_lin = -np.log(1/PB_AB_FFS[:-1, :] - 1)   # remove P_B == 1 at [-1]
			d_PB_lin = d_PB_AB_FFS[:-1, :] / (PB_AB_FFS[:-1, :] * (1 - PB_AB_FFS[:-1, :]))
			
			if(N_L > 1):
				i_L_max = np.argmax(Ls)
				Dlt_PB_lin = np.empty((PB_lin.shape[0], PB_lin.shape[1]))
				d_Dlt_PB_lin = np.empty((PB_lin.shape[0], PB_lin.shape[1]))
				for i_l in range(N_L):
					Dlt_PB_lin[:, i_l] = PB_lin[:, i_l] - PB_lin[:, i_L_max]
					d_Dlt_PB_lin[:, i_l] = np.sqrt(d_PB_lin[:, i_l]**2 + d_PB_lin[:, i_L_max]**2)
		
		Th_lbl = '$\epsilon_{11, 22}/T = ' + my.f2s(e[1,1]) + ',' + my.f2s(e[2,2]) + '$; $\mu_{1,2}/T = ' + my.f2s(mu[1]) + ',' + my.f2s(mu[2]) + '$'
		
		fig_kAB_L, ax_kAB_L, _ = my.get_fig('L', r'$k_{AB} / L^2$ [trans / (sweep $\cdot l^2$)]', title=r'$k_{AB}(L)$; ' + Th_lbl, xscl='log')
		ax_kAB_L.errorbar(Ls, k_AB_mean, yerr=d_k_AB, fmt='.')
		
		if('BF' in mode):
			fig_kABcount_L, ax_kABcount_L, _ = my.get_fig('L', r'$N_{AB}$ [event]', title=r'$N_{AB}(L)$; ' + Th_lbl, xscl='log')
			ax_kABcount_L.errorbar(Ls, k_AB_BFcount_N, yerr=d_k_AB_BFcount_N, fmt='.')
		
		if('FFS' in mode):
			fig_flux0_L, ax_flux0_L, _ = my.get_fig('L', r'$\Phi_{A} / L^2$ [trans / (sweep $\cdot l^2$)]', title=r'$\Phi_{A}(L)$; ' + Th_lbl, xscl='log')
			fig_P_L, ax_P_L, _ = my.get_fig('L', r'$P_{AB}$', title=r'$P_{AB}(L)$; ' + Th_lbl, xscl='log')
			
			ax_flux0_L.errorbar(Ls, flux0_AB_FFS, yerr=d_flux0_AB_FFS, fmt='.')
			ax_P_L.errorbar(Ls, P_AB, yerr=d_P_AB, fmt='.')
			
			if(N_L > 1):
				fig_PB_L, ax_PB_L, _ = my.get_fig('$CS$', r'$-\ln(1/P_{B} - 1)$', title=r'$P_{B}(L, CS)$; ' + Th_lbl)
				fig_dPB_L, ax_dPB_L, _ = my.get_fig('$CS$', r'$\Delta \ln(1/P_{B} - 1)$', title=r'$P_{B}(L, CS)$; ' + Th_lbl)
				
				for i_l in range(N_L):
					ax_PB_L.errorbar(OP_interfaces_AB[i_l][:-1], PB_lin[:, i_l], yerr=d_PB_lin[:, i_l], \
									label='L = ' + str(Ls[i_l]))
					if(i_l == i_L_max):
						ax_dPB_L.plot([min(OP_interfaces_AB[0][:-1]), max(OP_interfaces_AB[0][:-1])], [0] * 2, '--', label=r'$L = ' + str(Ls[i_L_max]) + '$')
					else:
						ax_dPB_L.errorbar(OP_interfaces_AB[i_l][:-1], Dlt_PB_lin[:, i_l], yerr=d_Dlt_PB_lin[:, i_l], \
										label='L = ' + str(Ls[i_l]))
				ax_PB_L.plot([min(OP_interfaces_AB[0][:-1]), max(OP_interfaces_AB[0][:-1])], [0] * 2, '--', label=r'$P_B = 1/2$')
				
				my.add_legend(fig_PB_L, ax_PB_L)
				my.add_legend(fig_dPB_L, ax_dPB_L)
			
			if(interface_mode == 'CS'):
				fig_Nc_L, ax_Nc_L, _ = my.get_fig('L', '$N_c$', title='$N_c(L)$; ' + Th_lbl, xscl='log')
				ax_Nc_L.errorbar(Ls, OP0_AB, yerr=d_OP0_AB, fmt='.')
			elif(interface_mode == 'M'):
				fig_Nc_L, ax_Nc_L, _ = my.get_fig('L', '$m_c + 1$', title='$m_c(L)$; ' + Th_lbl, xscl='log', yscl='log')
				ax_Nc_L.errorbar(Ls, OP0_AB + 1, yerr=d_OP0_AB, fmt='.')
	
	elif('compare' in mode):
		F_FFS, d_F_FFS, OP_hist_centers_FFS, OP_hist_lens_FFS, \
			ln_k_AB_FFS, d_ln_k_AB_FFS, \
			flux0_AB_FFS, d_flux0_AB_FFS, \
			prob_AB_FFS, d_prob_AB_FFS, \
			PB_AB_FFS, d_PB_AB_FFS, \
			OP0_AB, d_OP0_AB, _, _, _ = \
				run_many(MC_move_mode, Ls[0], e, mu[0, :], N_runs, interface_mode, \
					stab_step=stab_step, \
					N_init_states_AB=N_init_states_AB, \
					OP_interfaces_AB=OP_interfaces_AB[0], \
					to_plot_k_distr=False, N_k_bins=N_k_bins, \
					mode='FFS_AB', init_gen_mode=init_gen_mode, \
					OP_sample_BF_A_to=OP_sample_BF_A_to[0], \
					OP_match_BF_A_to=OP_match_BF_A_to[0], \
					init_composition=init_composition, \
					to_get_timeevol=to_get_timeevol, \
					to_save_npz=to_save_npz, to_recomp=to_recomp, \
					N_fourier=N_fourier, seeds=my_seeds)
		# ZeldG, d_ZeldG, chi2_rho are ignored
		
		F_BF, d_F_BF, M_hist_centers_BF, M_hist_lens_BF, ln_k_AB_BF, d_ln_k_AB_BF, ln_k_BA_BF, d_ln_k_BA_BF, \
			ln_k_bc_AB_BF, d_ln_k_bc_AB_BF, ln_k_bc_BA_BF, d_ln_k_bc_BA_BF = \
				run_many(MC_move_mode, Ls[0], e, mu[0, :], N_runs, interface_mode, \
						stab_step=stab_step, \
						OP_A=OP_0[0], OP_B=OP_max[0], \
						Nt_per_BF_run=Nt, mode='BF', \
						to_plot_k_distr=False, N_k_bins=N_k_bins, \
						N_saved_states_max=N_saved_states_max, \
						init_composition=init_composition, \
						R_clust_init=R_clust_init, \
						to_save_npz=to_save_npz, to_recomp=to_recomp, \
						seeds=my_seeds)
		#F_BF = F_BF - min(F_BF)
		F_BF = F_BF - F_BF[0]
		
		#assert(np.all(abs(M_hist_centers_BF - OP_hist_centers_FFS) < (M_hist_centers_BF[0] - M_hist_centers_BF[1]) * 1e-3)), 'ERROR: returned M_centers do not match'
		#assert(np.all(abs(M_hist_lens_BF - OP_hist_lens_FFS) < M_hist_lens_BF[0] * 1e-3)), 'ERROR: returned M_centers do not match'
		
		k_bc_AB_BF_mean, k_bc_AB_BF_min, k_bc_AB_BF_max, d_k_bc_AB_BF = get_log_errors(ln_k_bc_AB_BF, d_ln_k_bc_AB_BF, lbl='k_bc_AB_BF', print_scale=print_scale_k)
		k_AB_BF_mean, k_AB_BF_min, k_AB_BF_max, d_k_AB_BF = get_log_errors(ln_k_AB_BF, d_ln_k_AB_BF, lbl='k_AB_BF', print_scale=print_scale_k)
		get_log_errors(np.log(flux0_AB_FFS), d_flux0_AB_FFS / flux0_AB_FFS, lbl='flux0_AB', print_scale=print_scale_flux)
		k_AB_FFS_mean, k_AB_FFS_min, k_AB_FFS_max, d_k_AB_FFS = get_log_errors(ln_k_AB_FFS, d_ln_k_AB_FFS, lbl='k_AB_FFS', print_scale=print_scale_k)
		
		k_bc_BA_BF_mean, k_bc_BA_BF_min, k_bc_BA_BF_max, d_k_bc_BA_BF = get_log_errors(ln_k_bc_BA_BF, d_ln_k_bc_BA_BF, lbl='k_bc_BA_BF', print_scale=print_scale_k)
		k_BA_BF_mean, k_BA_BF_min, k_BA_BF_max, d_k_BA_BF = get_log_errors(ln_k_BA_BF, d_ln_k_BA_BF, lbl='k_BA_BF', print_scale=print_scale_k)
		#if('AB' not in mode):
		#	get_log_errors(np.log(flux0_BA_FFS), d_flux0_BA_FFS / flux0_BA_FFS, lbl='flux0_BA', print_scale=print_scale_flux)
		#	k_BA_FFS_mean, k_BA_FFS_min, k_BA_FFS_max, d_k_BA_FFS = get_log_errors(ln_k_BA_FFS, d_ln_k_BA_FFS, lbl='k_BA_FFS', print_scale=print_scale_k)
		
		ThL_lbl = get_ThL_lbl(e, mu, init_composition, L, swap_type_move)
		fig_F, ax_F, _ = my.get_fig(title[interface_mode], r'$F / J$', title=r'$F(' + feature_label[interface_mode] + ')$; ' + ThL_lbl)
		ax_F.errorbar(M_hist_centers_BF, F_BF, yerr=d_F_BF, label='BF')
		if(to_get_timeevol):
			ax_F.errorbar(OP_hist_centers_FFS, F_FFS, yerr=d_F_FFS, label='FFS')
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
		my.add_legend(fig_F, ax_F)
	
	else:
		print('ERROR: mode=%s is not supported' % (mode))
	
	if(to_show_on_screen):
		plt.show()

if(__name__ == "__main__"):
	main()

'''
	elif(mode == 'BF_mu_grid'):
		mu1_vec = [4, 5, 6]
		mu2_vec = [5, 7.5, 10]
		
		mu1_vec = [4, 5]
		mu2_vec = [5, 10]
		
		#mu1_vec = [5,6]
		#mu2_vec = [7.5, 10]
		
		mu1_vec = [4, 4.5, 4.95, 5.0, 5.03, 5.1, 5.12, 6, 7, 10]
		mu2_vec = [10, 11, 15, 20, 50, 100]
		
		mu1_vec = np.linspace(4.8, 5.2, 41)
		mu2_vec = np.linspace(4,10,61)
		
		#mu1_vec = np.linspace(4.8, 5.2, 41)
		#mu2_vec = np.linspace(0,4,41)
		
		#mu1_vec = np.linspace(5.4, 5.8, 41)
		#mu2_vec = np.linspace(0,10,101)
		
		mu1_vec = np.linspace(5.8, 10, 43)
		mu2_vec = np.linspace(2, 4, 21)

		mu1_vec = np.linspace(5.8, 10, 43)
		mu2_vec = np.linspace(3, 4, 101)

		mu1_vec = np.linspace(2, 8, 301)
		mu2_vec = np.linspace(2, 5, 151)

		mu1_vec = np.linspace(2, 8, 601)
		mu2_vec = np.linspace(4, 5, 100)

		mu1_vec = np.linspace(2, 3, 21)   # t=1.03
		mu2_vec = np.linspace(4, 6, 41)


'''
