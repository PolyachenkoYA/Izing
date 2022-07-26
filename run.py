import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import sys
import scipy.interpolate

import mylib as my
import table_data

# ========================== general params ==================
mu_shift = np.log(2)
mu_shift = 0
dE_avg = 6   # J
N_species = 3
print_scale_k = 1e5
print_scale_flux = 1e2
OP_min_default = {}
OP_max_default = {}
OP_scale = {}
dOP_step = {}
OP_std_overestimate = {}
OP_hist_edges_default = {}
OP_possible_jumps = {}

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

# ========================== recompile ==================
if(__name__ == "__main__"):
	to_recompile = True
	if(to_recompile):
		filebase = 'lattice_gas'
		path_to_so = os.path.join(filebase, 'cmake-build-release')
		N_dirs_down = path_to_so.count('/') + 1
		path_back = '/'.join(['..'] * N_dirs_down)

		os.chdir(path_to_so)
		my.run_it('cmake --build . --target %s.so -j 9' % (filebase))
		os.chdir(path_back)
		my.run_it('cp %s/%s.so.cpython-38-x86_64-linux-gnu.so ./%s.so' % (path_to_so, filebase, filebase))
		print('recompiled %s' % (filebase))

	import lattice_gas
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
	fig, ax, _ = my.get_fig(r'$%s$%s' % (lbl, units_inv_str), r'$\rho$%s' % (units_str), r'$\rho(%s)$' % (lbl))
	
	ax.bar(k_centers, k_hist / k_lens, yerr=d_k_hist / k_lens, width=k_lens, align='center')
	
def flip_OP(OP, L2, interface_mode):
	if(interface_mode == 'M'):
		return -np.flip(OP)
	elif(interface_mode == 'CS'):
		return L2 - np.flip(OP)

def get_sigmoid_fit(PB, d_PB, OP, fit_w=None):
	OP0_guess =  OP[np.argmin(abs(np.log(PB) - np.log(1/2)))]
	
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

def plot_PB_AB(ax, ax_log, ax_sgm, title, x_lbl, OP, PB, d_PB=None, PB_sgm=None, d_PB_sgm=None, linfit=None, linfit_inds=None, OP0=None, d_OP0=None, clr=None, N_fine_points=-10):
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

def center_crd(x, L):
	r2 = np.empty(L)
	for i in range(L):
		r2[i] = np.std(np.mod(x + i, L))
	
	return np.argmin(r2), r2

def draw_state(state_crds, to_show=False, ax=None):
	if(ax is None):
		fig, ax, _ = my.get_fig('x', 'y')
	
	ax.scatter(state_crds[:, 0], state_crds[:, 1])
	
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
	
	x_shift, _ = center_crd(x_crds, L)
	y_shift, _ = center_crd(y_crds, L)
	
	crds = np.empty((N, 2))

	crds[:, 0] = np.mod(inds % L + x_shift, L)
	crds[:, 1] = np.mod(inds // L + y_shift, L)
	
	crds[:, 0] = crds[:, 0] - np.mean(crds[:, 0])
	crds[:, 1] = crds[:, 1] - np.mean(crds[:, 1])
	
	return crds, x_shift, y_shift

def proc_order_parameter_FFS(L, e, mu, flux0, d_flux0, probs, \
							d_probs, OP_interfaces, N_init_states, \
							interface_mode, \
							OP_data=None, time_data=None, Nt=None, OP_hist_edges=None, memory_time=1, \
							to_plot_time_evol=False, to_plot_hists=False, stride=1, \
							x_lbl='', y_lbl='', Ms_alpha=0.5, OP_match_BF_to=None, \
							states=None, OP_optim_minstep=None, cluster_map_dx=1.41, cluster_map_dr=0.5):
	L2 = L**2
	ln_k_AB = np.log(flux0 * 1) + np.sum(np.log(probs))   # [flux0 * 1] = 1, because [flux] = 1/time = 1/step
	d_ln_k_AB = np.sqrt((d_flux0 / flux0)**2 + np.sum((d_probs / probs)**2))
	Temp = 4 / e[1, 1]
	
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
	Nc_0 = table_data.Nc_reference_data[str(Temp)] if(str(Temp) in table_data.Nc_reference_data) else None
	P_B_sigmoid, d_P_B_sigmoid, linfit, linfit_inds, OP0 = \
		get_sigmoid_fit(P_B[:-1], d_P_B[:-1], OP_interfaces_scaled[:-1])
	
	OP_optimize_f_interp_vals = 1 - np.log(P_B) / np.log(P_B[0])
	d_OP_optimize_f_interp_vals = d_P_B / P_B / np.log(P_B[0])
	#M_optimize_f_interp_fit = lambda OP: (1 - np.log(P_B_opt_fnc(OP)) / np.log(P_B[0]))
	OP_optimize_f_interp_interp1d = scipy.interpolate.interp1d(OP_interfaces_scaled, OP_optimize_f_interp_vals, fill_value='extrapolate')
	#OP_optimize_f_desired = np.linspace(0, 1, N_OP_interfaces)
	OP_optimize_f_desired = np.concatenate((np.linspace(0, OP_optimize_f_interp_interp1d(OP0 * 1.05), N_OP_interfaces - 1), np.array([1])))
	
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
	if(np.any(OP_optimize_new_OP_interfaces[1:] - OP_optimize_new_OP_interfaces[:-1] < OP_optim_minstep)):
		print('WARNING: too close interfaces, removing duplicates')
		for i in range(N_OP_interfaces - 1):
			if(OP_optimize_new_OP_interfaces[i+1] - OP_optimize_new_OP_interfaces[i] < OP_optim_minstep):
				if(i == 0):
					OP_optimize_new_OP_interfaces[1] = OP_optimize_new_OP_interfaces[0] + OP_optim_minstep
				else:
					if(OP_optimize_new_OP_interfaces[i-1] + OP_optim_minstep < OP_optimize_new_OP_interfaces[i] * (1 - 1e-5 * np.sign(OP_optimize_new_OP_interfaces[i]))):
						OP_optimize_new_OP_interfaces[i] = OP_optimize_new_OP_interfaces[i-1] + OP_optim_minstep
					else:
						OP_optimize_new_OP_interfaces[i+1] = OP_optimize_new_OP_interfaces[i] + OP_optim_minstep
		#OP_optimize_new_OP_interfaces = np.unique(OP_optimize_new_OP_interfaces)
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
	
	if(to_plot_hists):
		if(states is not None):
			# ============ process ===========
			OP_init_states = []
			OP_init_hist = np.empty((N_OP_hist, N_OP_interfaces))
			d_OP_init_hist = np.empty((N_OP_hist, N_OP_interfaces))
			cluster_centered_crds = []
			cluster_centered_crds_per_interface = [[]] * N_OP_interfaces
			for i in range(N_OP_interfaces):
				OP_init_states.append(np.empty(N_init_states[i], dtype=int))
				cluster_centered_crds.append([[]] * N_init_states[i])
				for j in range(N_init_states[i]):
					if(interface_mode == 'M'):
						OP_init_states[i][j] = np.sum(states[i][j, :, :])
					elif(interface_mode == 'CS'):
						#(cluster_element_inds, cluster_sizes, cluster_types) = lattice_gas.cluster_state(states[i][j, :, :].reshape((L2)))
						(cluster_element_inds, cluster_sizes, cluster_types) = lattice_gas.cluster_state(states[i][j, :, :].flatten())
						#OP_init_states[i][j] = max(cluster_sizes)
						
						max_cluster_id = np.argmax(cluster_sizes)
						OP_init_states[i][j] = cluster_sizes[max_cluster_id]
						max_cluster_1st_ind_id = np.sum(cluster_sizes[:max_cluster_id])
						cluster_centered_crds[i][j], _, _ = \
							center_cluster(cluster_element_inds[max_cluster_1st_ind_id : max_cluster_1st_ind_id + cluster_sizes[max_cluster_id]], L)
				
				cluster_centered_crds_per_interface[i] = np.concatenate(tuple(cluster_centered_crds[i]), axis=0)
				
				# plt.hist2d(cluster_centered_crds_joined[:, 0], cluster_centered_crds_joined[:, 1], \
										# bins=[int(max(cluster_centered_crds_joined[:, 0]) - min(cluster_centered_crds_joined[:, 0] + 0.5)),\
											  # int(max(cluster_centered_crds_joined[:, 1]) - min(cluster_centered_crds_joined[:, 1] + 0.5))])
				# plt.show()
				
				assert(OP_hist_edges is not None), 'ERROR: no "OP_hist_edges" provided but "OP_init_states" is asked to be analyzed (_states != None)'
				OP_init_hist[:, i], _ = np.histogram(OP_init_states[i], bins=OP_hist_edges * OP_scale[interface_mode])
				OP_init_hist[:, i] = OP_init_hist[:, i] / N_init_states[i]
				d_OP_init_hist[:, i] = np.sqrt(OP_init_hist[:, i] * (1 - OP_init_hist[:, i]) / N_init_states[i])
			
			cluster_centered_crds_all = np.concatenate(tuple(cluster_centered_crds_per_interface), axis=0)
			def stepped_linspace(x, dx, margin=1e-3):
				mn = min(x) - dx * margin
				mx = max(x) + dx * margin
				return np.linspace(mn, mx, int((mx - mn) / dx + 0.5))
			cluster_map_edges = [stepped_linspace(cluster_centered_crds_all[:, 0], cluster_map_dx), \
								 stepped_linspace(cluster_centered_crds_all[:, 1], cluster_map_dx)]
			cluster_Rdens_edges = stepped_linspace(np.linalg.norm(cluster_centered_crds_all, axis=1), cluster_map_dr)
			cluster_map_centers = [(cluster_map_edges[0][1:] + cluster_map_edges[0][:-1]) / 2, \
									(cluster_map_edges[1][1:] + cluster_map_edges[1][:-1]) / 2]
			cluster_Rdens_centers = (cluster_Rdens_edges[1:] + cluster_Rdens_edges[:-1]) / 2
			cluster_centered_map = [[]] * N_OP_interfaces
			cluster_centered_Rdens = [[]] * N_OP_interfaces
			for i in range(N_OP_interfaces):
				cluster_centered_map[i], _, _ = np.histogram2d(cluster_centered_crds_per_interface[i][:, 0], \
														 cluster_centered_crds_per_interface[i][:, 1], \
														 bins=cluster_map_edges)
				cluster_centered_Rdens[i], _ = np.histogram(np.linalg.norm(cluster_centered_crds_per_interface[i], axis=1), \
														 bins=cluster_Rdens_edges)
				cluster_centered_map[i] = cluster_centered_map[i] / N_init_states[i] / cluster_map_dx**2
				cluster_centered_Rdens[i] = cluster_centered_Rdens[i] / N_init_states[i] / (np.pi * (cluster_Rdens_edges[1:]**2 - cluster_Rdens_edges[:-1]**2))
			
			# ============ plot ===========
			ThL_lbl = '$\epsilon_{11, 22}/T = ' + my.f2s(e[1,1]) + ',' + my.f2s(e[2,2]) + '$; $\mu_{1,2}/T = ' + my.f2s(mu[1]) + ',' + my.f2s(mu[2]) + '$; L = ' + str(L)
			
			fig_OPinit_hists, ax_OPinit_hists, _ = my.get_fig(y_lbl, r'$p_i(' + x_lbl + ')$', title=r'$p(' + x_lbl + ')$; ' + ThL_lbl, yscl='log')
			fig_cluster_Rdens, ax_cluster_Rdens, _ = my.get_fig('r', r'$\rho$', title=r'$\rho(r)$; ' + ThL_lbl)
			fig_cluster_Rdens_log, ax_cluster_Rdens_log, _ = my.get_fig('r', r'$\rho$', title=r'$\rho(r)$; ' + ThL_lbl, yscl='log')
			fig_cluster_map = [[]] * N_OP_interfaces
			ax_cluster_map = [[]] * N_OP_interfaces
			for i in range(N_OP_interfaces):
				ax_OPinit_hists.bar(OP_hist_centers, OP_init_hist[:, i], yerr=d_OP_init_hist[:, i], width=OP_hist_lens, align='center', label='OP = ' + str(OP_interfaces[i]), alpha=Ms_alpha)
				
				cluster_lbl = r'$OP[%d] = %d$; ' % (i, OP_interfaces[i])
				ax_cluster_Rdens.plot(cluster_Rdens_centers, cluster_centered_Rdens[i], label=cluster_lbl)
				ax_cluster_Rdens_log.plot(cluster_Rdens_centers, cluster_centered_Rdens[i], label=cluster_lbl)
				
				fig_cluster_map[i], ax_cluster_map[i], fig_id = my.get_fig('x', 'y', title=cluster_lbl + ThL_lbl)
				im = ax_cluster_map[i].imshow(cluster_centered_map[i], \
										extent = [min(cluster_map_centers[1]), max(cluster_map_centers[1]), \
												  min(cluster_map_centers[0]), max(cluster_map_centers[0])], \
										interpolation ='bilinear', origin ='lower', aspect='auto')
				# im = ax_cluster_map[i].imshow(cluster_centered_map[i], \
										# extent = [min(cluster_map_centers[1]), max(cluster_map_centers[1]), \
												  # min(cluster_map_centers[0]), max(cluster_map_centers[0])], \
										# interpolation ='bilinear', origin ='lower', aspect='auto')
				plt.figure(fig_id)
				cbar = plt.colorbar(im)
			
			ax_OPinit_hists.legend()
			ax_cluster_Rdens.legend()
			ax_cluster_Rdens_log.legend()
		
	rho = None
	d_rho = None
	
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
		
		ThL_lbl = '$\epsilon_{11, 22}/T = ' + my.f2s(e[1,1]) + ',' + my.f2s(e[2,2]) + '$; $\mu_{1,2}/T = ' + my.f2s(mu[1]) + ',' + my.f2s(mu[2]) + '$; L = ' + str(L)
		if(to_plot_time_evol):
			fig_OP, ax_OP, _ = my.get_fig('step', y_lbl, title=x_lbl + '(step); ' + ThL_lbl)
			
			for i in range(N_OP_interfaces):
				ax_OP.plot(timesteps[i][::stride], OP[i][::stride], label='data, i=%d' % (i))
				
			ax_OP.legend()
		
		if(to_plot_hists):
			fig_OPhists, ax_OPhists, _ = my.get_fig(y_lbl, r'$\rho_i(' + x_lbl + ') \cdot P(i | A)$', title=r'$\rho(' + x_lbl + ')$; ' + ThL_lbl, yscl='log')
			for i in range(N_OP_interfaces - 1):
				ax_OPhists.bar(OP_hist_centers, rho_for_sum[:, i], yerr=d_rho_for_sum[:, i], width=OP_hist_lens, align='center', label=str(i), alpha=Ms_alpha)
			
			for i in range(N_OP_interfaces):
				ax_OPhists.plot([OP_interfaces_scaled[i]] * 2, [min(rho), max(rho)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_OPhists.legend()
			
			fig_OPjumps_hists, ax_OPjumps_hists, _ = my.get_fig('d' + x_lbl, 'probability', title=x_lbl + ' jumps distribution; ' + ThL_lbl, yscl='log')
			for i in range(N_OP_interfaces):
				ax_OPjumps_hists.plot(OP_possible_jumps[interface_mode], OP_jumps_hists[:, i], label='$' + x_lbl + ' \in (' + str((OP_min_default[interface_mode]) if(i == 0) else OP_interfaces_scaled[i-1]) + ';' + str(OP_interfaces_scaled[i]) + ']$')
			ax_OPjumps_hists.legend()
			
			fig_F, ax_F, _ = my.get_fig(y_lbl, r'$F(' + x_lbl + r') = -T \ln(\rho(%s) \cdot d%s(%s))$' % (x_lbl, x_lbl, x_lbl), title=r'$F(' + x_lbl + ')$; ' + ThL_lbl)
			ax_F.errorbar(OP_hist_centers[BF_inds], F_BF, yerr=d_F_BF, fmt='.', label='$F_{BF}$')
			ax_F.errorbar(OP_hist_centers, F, yerr=d_F, fmt='.', label='$F_{total}$')
			for i in range(N_OP_interfaces):
				ax_F.plot([OP_interfaces_scaled[i]] * 2, [min(F), max(F)], '--', label=('interfaces' if(i == 0) else None), color=my.get_my_color(0))
			ax_F.legend()
		
	if(to_plot_hists):
		fig_PB_log, ax_PB_log, _ = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; ' + ThL_lbl, yscl='log')
		fig_PB, ax_PB, _ = my.get_fig(y_lbl, r'$P_B(' + x_lbl + ') = P(i|0)$', title=r'$P_B(' + x_lbl + ')$; ' + ThL_lbl)
		fig_PB_sgm, ax_PB_sgm, _ = my.get_fig(y_lbl, r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; ' + ThL_lbl)
		
		mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sgm, OP_interfaces_scaled, P_B[:-1], P_B_sigmoid, interface_mode)
		plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sgm, 'P_B', x_lbl, OP_interfaces_scaled[:-1], P_B[:-1], d_PB=d_P_B[:-1], \
					PB_sgm=P_B_sigmoid, d_PB_sgm=d_P_B_sigmoid, linfit=linfit, linfit_inds=linfit_inds, OP0=OP0, \
					clr=my.get_my_color(1))
		ax_PB_log.legend()
		ax_PB.legend()
		ax_PB_sgm.legend()
		
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
		
		ax_fl.legend()
		ax_fl_i.legend()

	return ln_k_AB, d_ln_k_AB, P_B, d_P_B, P_B_sigmoid, d_P_B_sigmoid, linfit, linfit_inds, OP0, \
			rho, d_rho, OP_hist_centers, OP_hist_lens, \
			ax_OP, ax_OPhists, ax_F, ax_PB_log, ax_PB, ax_PB_sgm, ax_fl, ax_fl_i

def proc_FFS_AB(L, e, mu, N_init_states, OP_interfaces, interface_mode, \
				verbose=None, to_get_timeevol=False, \
				OP_sample_BF_to=None, OP_match_BF_to=None, 
				to_plot_time_evol=False, to_plot_hists=False, timeevol_stride=-3000, \
				init_gen_mode=-2, Ms_alpha=0.5, OP_optim_minstep=8):
	print('=============== running mu = %s ==================' % (str(mu)))
	
	# py::tuple run_FFS(int L, py::array_t<double> e, py::array_t<double> mu, pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces,
	#			  int to_remember_timeevol, int init_gen_mode, int interface_mode,
	#			  std::optional<int> _verbose)
	# return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M, biggest_cluster_sizes, time);
	(_states, probs, d_probs, Nt, flux0, d_flux0, _E, _M, _CS, times) = \
		lattice_gas.run_FFS(L, e.flatten(), mu, N_init_states, OP_interfaces, verbose=verbose, \
					to_remember_timeevol=to_get_timeevol, init_gen_mode=init_gen_mode, \
					interface_mode=OP_C_id[interface_mode])
	
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
			proc_order_parameter_FFS(L, e, mu, flux0, d_flux0, probs, d_probs, OP_interfaces, N_init_states, \
						interface_mode, OP_match_BF_to=OP_match_BF_to, \
						OP_data=(data[interface_mode] / OP_scale[interface_mode] if(to_get_timeevol) else None), \
						time_data = times, \
						Nt=Nt, OP_hist_edges=get_OP_hist_edges(interface_mode, OP_max=OP_interfaces[-1]), \
						memory_time=max(1, OP_std_overestimate[interface_mode] / OP_step[interface_mode]), 
						to_plot_time_evol=to_plot_time_evol, to_plot_hists=to_plot_hists, stride=timeevol_stride, \
						x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], Ms_alpha=Ms_alpha, \
						states=states, OP_optim_minstep=OP_optim_minstep)
	
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

def proc_order_parameter_BF(L, e, mu, m, E, stab_step, dOP_step, OP_hist_edges, OP_peak_guess, OP_OP_fit2_width, \
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
		
		OP_fit2_OPmin_ind = np.argmax(OP_hist_centers > OP_peak_guess - OP_OP_fit2_width)
		OP_fit2_OPmax_ind = np.argmax(OP_hist_centers > OP_peak_guess + OP_OP_fit2_width)
		assert(OP_fit2_OPmin_ind > 0), 'issues finding analytical border'
		assert(OP_fit2_OPmax_ind > 0), 'issues finding analytical border'
		OP_fit2_inds = (OP_fit2_OPmin_ind <= np.arange(len(OP_hist_centers))) & (OP_fit2_OPmax_ind >= np.arange(len(OP_hist_centers)))
		OP_fit2 = np.polyfit(OP_hist_centers[OP_fit2_inds], F[OP_fit2_inds], 2, w = 1/d_F[OP_fit2_inds])
		OP_peak = -OP_fit2[1] / (2 * OP_fit2[0])
		m_peak_ind = np.argmin(abs(OP_hist_centers - OP_peak))
		F_max = np.polyval(OP_fit2, OP_peak)
		#m_peak_ind = np.argmin(OP_hist_centers - OP_peak)
		bc_Z_AB = exp_integrate(OP_hist_centers[ : OP_fit2_OPmin_ind + 1], -F[ : OP_fit2_OPmin_ind + 1]) + exp2_integrate(- OP_fit2, OP_hist_centers[OP_fit2_OPmin_ind])
		bc_Z_BA = exp_integrate(OP_hist_centers[OP_fit2_OPmax_ind : ], -F[OP_fit2_OPmax_ind : ]) + abs(exp2_integrate(- OP_fit2, OP_hist_centers[OP_fit2_OPmax_ind]))
		k_bc_AB = (np.mean(abs(OP_hist_centers[m_peak_ind + 1] - OP_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max) / bc_Z_AB)
		k_bc_BA = (np.mean(abs(OP_hist_centers[m_peak_ind + 1] - OP_hist_centers[m_peak_ind - 1]))/2 / 2) * (np.exp(-F_max) / bc_Z_BA)
		
		get_log_errors(np.log(k_AB), d_k_AB / k_AB, lbl='k_AB_BF', print_scale=print_scale_k)
		get_log_errors(np.log(k_BA), d_k_BA / k_BA, lbl='k_BA_BF', print_scale=print_scale_k)
	else:
		k_AB = None
		d_k_AB = None
		k_BA = None
		d_k_BA = None
		k_bc_AB = None
		k_bc_BA = None
	
	ThL_lbl = '$\epsilon_{11, 22}/T = ' + my.f2s(e[1,1]) + ',' + my.f2s(e[2,2]) + '$; $\mu_{1,2}/T = ' + my.f2s(mu[1]) + ',' + my.f2s(mu[2]) + '$; L = ' + str(L)
	if(to_plot_time_evol):
		fig_OP, ax_OP, _ = my.get_fig('step', y_lbl, title='$' + x_lbl + '$(step); ' + ThL_lbl)
		ax_OP.plot(steps[::stride], m[::stride], label='data')
		ax_OP.plot([stab_step] * 2, [min(m), max(m)], '--', label='equilibr')
		ax_OP.plot([stab_step, Nt], [OP_mean] * 2, '--', label=('$<' + x_lbl + '> = ' + my.errorbar_str(OP_mean, d_OP_mean) + '$'))
		if(to_estimate_k):
			ax_OP.plot([0, Nt], [OP_A] * 2, '--', label='state A, B', color=my.get_my_color(1))
			ax_OP.plot([0, Nt], [OP_B] * 2, '--', label=None, color=my.get_my_color(1))
		ax_OP.legend()
		
		fig_hA, ax_hA, _ = my.get_fig('step', '$h_A$', title='$h_{A, ' + x_lbl + '}$(step); ' + ThL_lbl)
		ax_hA.plot(steps[::stride], hA[::stride], label='$h_A$')
		ax_hA.plot(steps[jump_inds+1], hA_jump[jump_inds], '.', label='$h_A$ jumps')
		ax_hA.legend()

		#print('k_' + x_lbl + '_AB    =    (' + my.errorbar_str(k_AB, d_k_AB) + ')    (1/step);    dk/k =', my.f2s(d_k_AB / k_AB))
		#print('k_' + x_lbl + '_BA    =    (' + my.errorbar_str(k_BA, d_k_BA) + ')    (1/step);    dk/k =', my.f2s(d_k_BA / k_BA))
	else:
		ax_OP = None
	
	if(to_plot_F):
		fig_OP_hist, ax_OP_hist, _ = my.get_fig(y_lbl, r'$\rho(' + x_lbl + ')$', title=r'$\rho(' + x_lbl + ')$; ' + ThL_lbl)
		ax_OP_hist.bar(OP_hist_centers, rho, yerr=d_rho, width=OP_hist_lens, label=r'$\rho(' + x_lbl + ')$', align='center')
		if(to_estimate_k):
			ax_OP_hist.plot([OP_A] * 2, [0, max(rho)], '--', label='state A,B', color=my.get_my_color(1))
			ax_OP_hist.plot([OP_B] * 2, [0, max(rho)], '--', label=None, color=my.get_my_color(1))
		ax_OP_hist.legend()
		
		fig_OP_jumps_hist, ax_OP_jumps_hist, _ = my.get_fig('d' + x_lbl, 'probability', title=x_lbl + ' jumps distribution; ' + ThL_lbl, yscl='log')
		ax_OP_jumps_hist.plot(possible_jumps, OP_jumps_hist, '.')
		
		fig_F, ax_F, _ = my.get_fig(y_lbl, r'$F/J$', title=r'$F(' + x_lbl + ')/T$; ' + ThL_lbl)
		ax_F.errorbar(OP_hist_centers, F, yerr=d_F, fmt='.', label='$F(' + x_lbl + ')/T$')
		if(to_estimate_k):
			ax_F.plot([OP_A] * 2, [min(F), max(F)], '--', label='state A, B', color=my.get_my_color(1))
			ax_F.plot([OP_B] * 2, [min(F), max(F)], '--', label=None, color=my.get_my_color(1))
			ax_F.plot(OP_hist_centers[OP_fit2_inds], np.polyval(OP_fit2, OP_hist_centers[OP_fit2_inds]), label='fit2; $' + x_lbl + '_0 = ' + my.f2s(OP_peak) + '$')
		
		if(to_plot_ETS):
			ax_F.plot(OP_hist_centers, E_avg, '.', label='<E>(' + x_lbl + ')/T')
			ax_F.plot(OP_hist_centers, S, '.', label='$T \cdot S(' + x_lbl + ')$')
		ax_F.legend()
		
		if(to_estimate_k):
			print('k_' + x_lbl + '_bc_AB   =   ' + my.f2s(k_bc_AB * print_scale_k) + '   (%e/step);' % (1/print_scale_k))
			print('k_' + x_lbl + '_bc_BA   =   ' + my.f2s(k_bc_BA * print_scale_k) + '   (%e/step);' % (1/print_scale_k))
	else:
		ax_OP_hist = None
		ax_F = None
	
	return F, d_F, OP_hist_centers, OP_hist_lens, rho_interp1d, d_rho_interp1d, k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_mean, OP_std, d_OP_mean, E_avg, S, ax_OP, ax_OP_hist, ax_F

def plot_correlation(x, y, x_lbl, y_lbl):
	fig, ax, _ = my.get_fig(x_lbl, y_lbl)
	R = scipy.stats.pearsonr(x, y)[0]
	ax.plot(x, y, '.', label='R = ' + my.f2s(R))
	ax.legend()
	
	return ax, R

def proc_T(L, e, mu, Nt, interface_mode, verbose=None, N_spins_up_init=None, to_get_timeevol=True, \
			to_plot_time_evol=False, to_plot_F=False, to_plot_ETS=False, to_plot_correlations=False, N_saved_states_max=-1, \
			to_estimate_k=False, timeevol_stride=-3000, OP_min=None, OP_max=None, OP_A=None, OP_B=None):
	L2 = L**2
	
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
	
	(E, M, CS, hA, times, k_AB_launches, time_total) = lattice_gas.run_bruteforce(L, e.flatten(), mu, Nt, N_spins_up_init=N_spins_up_init, OP_A=OP_A, OP_B=OP_B, \
										OP_min=OP_min, OP_max=OP_max, to_remember_timeevol=to_get_timeevol, N_saved_states_max=N_saved_states_max, \
										interface_mode=OP_C_id[interface_mode], \
										verbose=verbose)
	
	k_AB_BFcount = k_AB_launches / time_total
	if(k_AB_BFcount > 0):
		d_k_AB_BFcount = k_AB_BFcount / np.sqrt(k_AB_launches)
		print('k_AB_BFcount = (' + my.errorbar_str(k_AB_BFcount, d_k_AB_BFcount) + ') 1/step')
	else:
		d_k_AB_BFcount = 1
		print('k_AB_BFcount = 0')

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
					proc_order_parameter_BF(L, e, mu, data[interface_mode] / OP_scale[interface_mode], E / L2, stab_step, \
									dOP_step[interface_mode], hist_edges[interface_mode], OP_peak_guess[interface_mode], OP_fit2_width[interface_mode], \
									x_lbl=feature_label[interface_mode], y_lbl=title[interface_mode], verbose=verbose, to_estimate_k=to_estimate_k, hA=hA, \
									to_plot_time_evol=to_plot_time_evol, to_plot_F=to_plot_F, to_plot_ETS=to_plot_ETS, stride=timeevol_stride, \
									OP_A=OP_A / OP_scale[interface_mode], OP_B=OP_B / OP_scale[interface_mode], OP_jumps_hist_edges=OP_jumps_hist_edges, \
									possible_jumps=OP_possible_jumps[interface_mode])

		BC_cluster_size = L2 / np.pi
		#C = ((OP_std['E'] * L2)**2) / L2   # heat capacity per spin

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
	else:
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
		
	return F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_avg, d_OP_avg, E_avg, k_AB_BFcount, d_k_AB_BFcount, k_AB_launches#, C
	
	
def run_many(L, e, mu, N_runs, interface_mode, \
			OP_A=None, OP_B=None, N_spins_up_init=None, N_saved_states_max=-1, \
			OP_interfaces_AB=None, OP_interfaces_BA=None, \
			OP_sample_BF_A_to=None, OP_sample_BF_B_to=None, \
			OP_match_BF_A_to=None, OP_match_BF_B_to=None, \
			Nt_per_BF_run=None, verbose=None, to_plot_time_evol=False, \
			timeevol_stride=-3000, to_plot_k_distr=False, N_k_bins=10, \
			mode='BF', N_init_states_AB=None, N_init_states_BA=None, \
			init_gen_mode=-2, to_plot_committer=None, to_get_timeevol=None):
	L2 = L**2
	old_seed = lattice_gas.get_seed()
	if(to_get_timeevol is None):
		to_get_timeevol = ('BF' == mode)
	
	if('BF' in mode):
		assert(Nt_per_BF_run is not None), 'ERROR: BF mode but no Nt_per_run provided'
		if(OP_A is None):
			assert(len(OP_interfaces_AB) > 0), 'ERROR: nor OP_A neither OP_interfaces_AB were provided for run_many(BF)'
			OP_A = OP_interfaces_AB[0]
		
	elif('FFS' in mode):
		assert(N_init_states_AB is not None), 'ERROR: FFS_AB mode but no "N_init_states_AB" provided'
		assert(OP_interfaces_AB is not None), 'ERROR: FFS_AB mode but no "OP_interfaces_AB" provided'
		
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
			
	for i in range(N_runs):
		lattice_gas.init_rand(i + old_seed)
		if(mode == 'BF'):
			F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, \
				rho_fncs[i], d_rho_fncs[i], k_bc_AB_BF, k_bc_BA_BF, k_AB_BF, _, k_BA_BF, _, _, _, _, k_AB_BFcount, _, _ = \
					proc_T(L, e, mu, Nt_per_BF_run, interface_mode, \
							OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
							verbose=verbose, to_plot_time_evol=to_plot_time_evol, \
							to_plot_F=False, to_plot_correlations=False, to_plot_ETS=False, \
							timeevol_stride=timeevol_stride, to_estimate_k=True, 
							to_get_timeevol=True, N_saved_states_max=N_saved_states_max)
			
			ln_k_bc_AB_data[i] = np.log(k_bc_AB_BF * 1)   # proc_T returns 'k', not 'log(k)'; *1 for units [1/step]
			ln_k_bc_BA_data[i] = np.log(k_bc_BA_BF * 1)
			ln_k_AB_data[i] = np.log(k_AB_BF * 1)
			ln_k_BA_data[i] = np.log(k_BA_BF * 1)
		elif(mode  == 'BF_AB'):
			F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, \
				rho_fncs[i], d_rho_fncs[i], _, _, _, _, _, _, _, _, _, k_AB_BFcount, _, k_AB_BFcount_N_data[i] = \
					proc_T(L, e, mu, Nt_per_BF_run, interface_mode, \
							OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
							verbose=verbose, to_plot_time_evol=to_plot_time_evol, \
							timeevol_stride=timeevol_stride, to_estimate_k=False,
							to_get_timeevol=to_get_timeevol, OP_max=OP_B, \
							N_saved_states_max=N_saved_states_max)
			
			ln_k_AB_data[i] = np.log(k_AB_BFcount * 1)
		elif(mode == 'FFS_AB'):
			probs_AB_data[:, i], _, ln_k_AB_data[i], _, flux0_AB_data[i], _, rho_new, d_rho_new, OP_hist_centers_new, OP_hist_lens_new, \
			PB_AB_data[:, i], d_PB_AB_data[:, i], PB_sigmoid_data[:, i], d_PB_sigmoid_data[:, i], PB_linfit_data[:, i], PB_linfit_inds_data[:, i], OP0_AB_data[i] = \
				proc_FFS_AB(L, e, mu, N_init_states_AB, OP_interfaces_AB, interface_mode, \
							OP_sample_BF_to=OP_sample_BF_A_to, OP_match_BF_to=OP_match_BF_A_to, init_gen_mode=init_gen_mode, to_get_timeevol=to_get_timeevol)
			if(to_get_timeevol):
				rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, rho_new, fill_value='extrapolate')
				d_rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, d_rho_new, fill_value='extrapolate')
				rho_AB = rho_fncs[i](OP_hist_centers_new)
				d_rho_AB = d_rho_fncs[i](OP_hist_centers_new)
				F_new = -np.log(rho_AB * OP_hist_lens_new)
				d_F_new = d_rho_AB / rho_AB
				F_new = F_new - F_new[0]
			
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
	
		
	if(mode == 'BF_AB'):
		k_AB_BFcount_N, d_k_AB_BFcount_N = get_average(k_AB_BFcount_N_data)

	if(mode == 'BF'):
		ln_k_bc_AB, d_ln_k_bc_AB = get_average(ln_k_bc_AB_data)
		ln_k_bc_BA, d_ln_k_bc_BA = get_average(ln_k_bc_BA_data)
		
	elif('FFS' in mode):
		flux0_AB, d_flux0_AB = get_average(flux0_AB_data, mode='log')
		OP0_AB, d_OP0_AB = get_average(OP0_AB_data)
		probs_AB, d_probs_AB = get_average(probs_AB_data, mode='log', axis=1)
		PB_AB, d_PB_AB = get_average(PB_AB_data, mode='log', axis=1)

		#Temp = 4 / e[1,1]
		#Nc_0 = table_data.Nc_reference_data[str(Temp)] if(str(Temp) in table_data.Nc_reference_data) else None
		PB_sigmoid, d_PB_sigmoid, linfit_AB, linfit_AB_inds, m0_AB_fit = \
			get_sigmoid_fit(PB_AB[:-1], d_PB_AB[:-1], OP_AB[:-1])

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
		ThL_lbl = '$\epsilon_{11, 22}/T = ' + my.f2s(e[1,1]) + ',' + my.f2s(e[2,2]) + '$; $\mu_{1,2}/T = ' + my.f2s(mu[1]) + ',' + my.f2s(mu[2]) + '$; L = ' + str(L)
		fig_PB_log, ax_PB_log, _ = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; ' + ThL_lbl, yscl='log')
		fig_PB, ax_PB, _ = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; ' + ThL_lbl)
		fig_PB_sigmoid, ax_PB_sigmoid, _ = my.get_fig(title[interface_mode], r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; ' + ThL_lbl)
		
		mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sigmoid, OP, PB_AB, PB_sigmoid, interface_mode)
		plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, 'P_B', feature_label[interface_mode], \
					OP_AB[:-1], PB_AB[:-1], d_PB=d_PB_AB[:-1], \
					PB_sgm=PB_sigmoid, d_PB_sgm=d_PB_sigmoid, \
					linfit=linfit_AB, linfit_inds=linfit_AB_inds, \
					OP0=OP0_AB, d_OP0=d_OP0_AB, \
					clr=my.get_my_color(1))
					
		ax_PB_log.legend()
		ax_PB.legend()
		ax_PB_sigmoid.legend()
	
	lattice_gas.init_rand(old_seed)
	
	if(mode == 'FFS'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, flux0_AB, d_flux0_AB, flux0_BA, d_flux0_BA, probs_AB, d_probs_AB, probs_BA, d_probs_BA, PB_AB, d_PB_AB, PA_BA, d_PA_BA, OP0_AB, d_OP0_AB, OP0_BA, d_OP0_BA
	elif(mode == 'FFS_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, probs_AB, d_probs_AB, PB_AB, d_PB_AB, OP0_AB, d_OP0_AB
	elif(mode == 'BF'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, ln_k_bc_AB, d_ln_k_bc_AB, ln_k_bc_BA, d_ln_k_bc_BA
	elif(mode == 'BF_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, k_AB_BFcount_N, d_k_AB_BFcount_N

def get_mu_dependence(mu_arr, L, e, N_runs, interface_mode, N_init_states, OP_interfaces, init_gen_mode=-3, to_plot=False, to_save_npy=True, to_recomp=False):
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
		npy_data = np.load(npy_filename)
		
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
				OP0_AB[i_mu], d_OP0_AB[i_mu] = \
					run_many(L, e, mu_arr[i_mu, :], N_runs, interface_mode, \
						N_init_states_AB=N_init_states, \
						OP_interfaces_AB=OP_interfaces[i_mu], \
						mode='FFS_AB', init_gen_mode=init_gen_mode, \
						to_get_timeevol=False, to_plot_committer=False)
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
			ax_k_err.legend()
			
			ax_k_ratio.errorbar(mu_arr[:, 1], k_AB_mean / k_AB_ref[mu_kAB_ref_inds], yerr=d_ln_k_AB * abs(k_AB_mean / k_AB_ref[mu_kAB_ref_inds]), fmt='.', label='data')
			ax_k_ratio.legend()

			ax_Nc_err.errorbar(mu_arr[:, 1], Nc_err, yerr=d_Nc_err, fmt='.', label='data')
			ax_Nc_err.legend()
		ax_k.legend()
		ax_Nc.legend()
		


def main():
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
	# TODO: remove outdated inputs
	[L, potential_filenames, mode, Nt, N_states_FFS, N_init_states_FFS, to_recomp, to_get_timeevol, verbose, my_seed, N_OP_interfaces, N_runs, init_gen_mode, OP_0, OP_max, interface_mode, OP_min_BF, OP_max_BF, Nt_sample_A, Nt_sample_B, N_spins_up_init, to_plot_ETS, interface_set_mode, timeevol_stride, to_plot_timeevol, N_saved_states_max, J, h, OP_interfaces_set_IDs], _ = \
		my.parse_args(sys.argv,            [ '-L', '-potential_filenames',  '-mode',        '-Nt', '-N_states_FFS', '-N_init_states_FFS',     '-to_recomp', '-to_get_timeevol', '-verbose', '-my_seed', '-N_OP_interfaces', '-N_runs', '-init_gen_mode', '-OP_0', '-OP_max', '-interface_mode', '-OP_min_BF', '-OP_max_BF', '-Nt_sample_A', '-Nt_sample_B',  '-N_spins_up_init',   '-to_plot_ETS', '-interface_set_mode', '-timeevol_stride', '-to_plot_timeevol', '-N_saved_states_max',   '-J',   '-h', '-OP_interfaces_set_IDs'], \
					  possible_arg_numbers=[['+'],                   None,      [1],       [0, 1],          [0, 1],               [0, 1],           [0, 1],             [0, 1],     [0, 1],     [0, 1],                [1],    [0, 1],           [0, 1],   ['+'],     ['+'],               [1],       [0, 1],       [0, 1],         [0, 1],         [0, 1],              [0, 1],           [0, 1],                [0, 1],             [0, 1],              [0, 1],                [0, 1], [0, 1],   None,                    None], \
					  default_values=      [ None,                 [None],     None, ['-1000000'],       ['-5000'],             ['5000'], [my.no_flags[0]],              ['1'],      ['1'],     ['23'],               None,     ['3'],           ['-3'],    None,      None,              None,       [None],       [None],   ['-1000000'],   ['-1000000'],              [None], [my.no_flags[0]],           ['optimal'],          ['-3000'],    [my.no_flags[0]],              ['1000'], [None], [None],                  [None]])
	
	Ls = np.array([int(l) for l in L], dtype=int)
	N_L = len(Ls)
	L = Ls[0]
	L2 = L**2
	
	if(J[0] is None):
		assert(potential_filenames[0] is not None), 'Nor J neither potential_filenames are provided, avorting'
		# load potentials
	else:
		assert(potential_filenames[0] is None), 'Both J and potential_filenames are provided, avorting'
		e = np.zeros((N_species, N_species))
		N_param_points = len(h)
		mu = np.zeros((N_param_points, N_species))
		
		e[1, 1] = 4 * float(J[0])
		mu[:, 1] = (2 * np.array([float(hh) for hh in h]) * float(J[0]) - 2 * e[1, 1]) + mu_shift
		mu[:, 2] = -1e10
		mu[:, 2] = 0

	#Temp = 1.0
	
	N_mu = mu.shape[0]
	N_param_points = N_L if('L' in mode) else (N_mu if('mu' in mode) else 1)
	
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
	OP_0 = np.array(([int(OP_0[0])] * N_param_points) if(len(OP_0) == 1) else [int(x) for x in OP_0], dtype=int)
	OP_max = np.array(([int(OP_max[0])] * N_param_points) if(len(OP_max) == 1) else [int(x) for x in OP_max], dtype=int)
	interface_mode = interface_mode
	N_spins_up_init = (None if(N_spins_up_init[0] is None) else int(N_spins_up_init[0]))
	to_plot_ETS = (to_plot_ETS[0] in my.yes_flags)
	interface_set_mode = interface_set_mode[0]
	timeevol_stride = int(timeevol_stride[0])
	to_plot_timeevol = (to_plot_timeevol[0] in my.yes_flags)
	N_saved_states_max = int(N_saved_states_max[0])
	
	OP_0_handy = np.copy(OP_0)
	OP_max_handy = np.copy(OP_max)
	if(interface_mode == 'M'):
		OP_0 = OP_0 - L2
		OP_max = OP_max - L2
	
	OP_min_BF = OP_min_default[interface_mode] if(OP_min_BF[0] is None) else int(OP_min_BF[0])
	OP_max_BF = OP_max_default[interface_mode] if(OP_max_BF[0] is None) else int(OP_max_BF[0])
	
	#assert(interface_mode == 'CS'), 'ERROR: interface_mode==M is temporary not supported'
	
	print('Ls=%s, mode=%s, Nt=%d, N_states_FFS=%d, N_OP_interfaces=%d, interface_mode=%s, N_runs=%d, init_gen_mode=%d, OP_0=%s, OP_max=%s, N_spins_up_init=%s, OP_min_BF=%d, OP_max_BF=%d, to_get_timeevol=%r, verbose=%d, my_seed=%d, to_recomp=%r, N_param_points=%d\ne: %s\nmu: %s\n' % \
		(str(Ls), mode, Nt, N_states_FFS, N_OP_interfaces, interface_mode, N_runs, init_gen_mode, str(OP_0), str(OP_max), str(N_spins_up_init), OP_min_BF, OP_max_BF, to_get_timeevol, verbose, my_seed, to_recomp, N_param_points, str(e), str(mu)))
	
	lattice_gas.init_rand(my_seed)
	lattice_gas.set_verbose(verbose)

	dE_avg = ((np.sqrt(e[1,1]) + np.sqrt(e[2,2])) / 2)**2
	Nt, N_states_FFS = \
		tuple([(int(-n * np.exp(3 - dE_avg) + 1) if(n < 0) else n) for n in [Nt, N_states_FFS]])

	BC_cluster_size = L2 / np.pi

	if(('compare' in mode) or ('many' in mode)):
		N_k_bins=int(np.round(np.sqrt(N_runs) / 2) + 1)
	
	if('FFS' in mode):
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
		
		print('OP_AB:', OP_interfaces_AB)
		
	if(mode == 'BF_1'):
		proc_T(Ls[0], e, mu[0, :], Nt, interface_mode, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_time_evol=to_plot_timeevol, to_plot_F=True, to_plot_ETS=to_plot_ETS, \
				to_plot_correlations=True and to_plot_timeevol, to_get_timeevol=True, \
				OP_min=OP_min_BF, OP_max=OP_max_BF, to_estimate_k=True, \
				timeevol_stride=timeevol_stride, N_saved_states_max=N_saved_states_max)
		
	if(mode == 'BF_AB'):
		proc_T(Ls[0], e, mu[0, :], Nt, interface_mode, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_time_evol=to_plot_timeevol, to_plot_F=True, to_plot_ETS=to_plot_ETS, \
				to_plot_correlations=True and to_plot_timeevol, to_get_timeevol=to_get_timeevol, \
				OP_min=OP_min_BF, OP_max=OP_max[0], to_estimate_k=False, \
				timeevol_stride=timeevol_stride, N_saved_states_max=N_saved_states_max)
	
	elif(mode == 'BF_many'):
		run_many(Ls[0], e, mu[0, :], N_runs, interface_mode, Nt_per_BF_run=Nt, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, \
				mode='BF', N_saved_states_max=N_saved_states_max)
	
	elif(mode == 'BF_AB_many'):
		run_many(Ls[0], e, mu[0, :], N_runs, interface_mode, Nt_per_BF_run=Nt, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, \
				mode='BF_AB', N_saved_states_max=N_saved_states_max)
	
	elif(mode == 'FFS_AB'):
		proc_FFS_AB(Ls[0], e, mu[0, :], N_init_states_AB, OP_interfaces_AB[0], interface_mode, \
					OP_sample_BF_to=OP_sample_BF_A_to[0], OP_match_BF_to=OP_match_BF_A_to[0], timeevol_stride=timeevol_stride, \
					to_get_timeevol=to_get_timeevol, init_gen_mode=init_gen_mode, to_plot_time_evol=to_plot_timeevol, to_plot_hists=True)
		
	elif(mode == 'FFS_AB_many'):
		F_FFS, d_F_FFS, M_hist_centers_FFS, OP_hist_lens_FFS, ln_k_AB_FFS, d_ln_k_AB_FFS, \
			flux0_AB_FFS, d_flux0_AB_FFS, prob_AB_FFS, d_prob_AB_FFS, PB_AB_FFS, d_PB_AB_FFS, \
			OP0_AB, d_OP0_AB = \
				run_many(Ls[0], e, mu[0, :], N_runs, interface_mode, \
					N_init_states_AB=N_init_states_AB, \
					OP_interfaces_AB=OP_interfaces_AB[0], \
					to_plot_k_distr=False, N_k_bins=N_k_bins, \
					mode='FFS_AB', init_gen_mode=init_gen_mode, \
					OP_sample_BF_A_to=OP_sample_BF_A_to[0], \
					OP_match_BF_A_to=OP_match_BF_A_to[0], \
					to_get_timeevol=to_get_timeevol)
		
		get_log_errors(np.log(flux0_AB_FFS), d_flux0_AB_FFS / flux0_AB_FFS, lbl='flux0_AB', print_scale=print_scale_flux)
		k_AB_FFS_mean, k_AB_FFS_min, k_AB_FFS_max, d_k_AB_FFS = get_log_errors(ln_k_AB_FFS, d_ln_k_AB_FFS, lbl='k_AB_FFS', print_scale=print_scale_k)
		print(my.errorbar_str(OP0_AB, d_OP0_AB))
		
	elif('mu' in mode):
		get_mu_dependence(mu, Ls[0], e, N_runs, interface_mode, \
						N_init_states_AB, OP_interfaces_AB, init_gen_mode=init_gen_mode, \
						to_plot=True, to_save_npy=True, to_recomp=to_recomp)
	
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
						run_many(Ls[i_l], e, mu[0, :], N_runs, interface_mode, Nt_per_BF_run=Nt, \
								OP_A=OP_0[i_l], OP_B=OP_max[i_l], N_spins_up_init=N_spins_up_init, \
								to_get_timeevol=to_get_timeevol, mode='BF_AB', N_saved_states_max=N_saved_states_max)
			elif('FFS' in mode):
				F_FFS_new, d_F_FFS_new, OP_hist_centers_FFS_new, OP_hist_lens_FFS_new, \
					ln_k_AB[i_l], d_ln_k_AB[i_l], \
					flux0_AB_FFS[i_l], d_flux0_AB_FFS[i_l], \
					prob_AB_FFS[:, i_l], d_prob_AB_FFS[:, i_l], \
					PB_AB_FFS[:, i_l], d_PB_AB_FFS[:, i_l], \
					OP0_AB[i_l], d_OP0_AB[i_l] = \
						run_many(Ls[i_l], e, mu[0, :], N_runs, interface_mode, \
							N_init_states_AB=N_init_states_AB, \
							OP_interfaces_AB=OP_interfaces_AB[i_l], \
							mode='FFS_AB', init_gen_mode=init_gen_mode, \
							OP_sample_BF_A_to=OP_sample_BF_A_to[i_l], \
							OP_match_BF_A_to=OP_match_BF_A_to[i_l], \
							to_get_timeevol=to_get_timeevol, to_plot_committer=False)
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
				
				ax_PB_L.legend()
				ax_dPB_L.legend()

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
			OP0_AB, d_OP0_AB = \
				run_many(Ls[0], e, mu[0, :], N_runs, interface_mode, \
					N_init_states_AB=N_init_states_AB, \
					OP_interfaces_AB=OP_interfaces_AB[0], \
					to_plot_k_distr=False, N_k_bins=N_k_bins, \
					mode='FFS_AB', init_gen_mode=init_gen_mode, \
					OP_sample_BF_A_to=OP_sample_BF_A_to[0], \
					OP_match_BF_A_to=OP_match_BF_A_to[0], \
					to_get_timeevol=to_get_timeevol)

		F_BF, d_F_BF, M_hist_centers_BF, M_hist_lens_BF, ln_k_AB_BF, d_ln_k_AB_BF, ln_k_BA_BF, d_ln_k_BA_BF, \
			ln_k_bc_AB_BF, d_ln_k_bc_AB_BF, ln_k_bc_BA_BF, d_ln_k_bc_BA_BF = \
				run_many(Ls[0], e, mu[0, :], N_runs, interface_mode, \
						OP_A=OP_0[0], OP_B=OP_max[0], \
						Nt_per_BF_run=Nt, mode='BF', \
						to_plot_k_distr=False, N_k_bins=N_k_bins, 
						N_saved_states_max=N_saved_states_max)
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
		if('AB' not in mode):
			get_log_errors(np.log(flux0_BA_FFS), d_flux0_BA_FFS / flux0_BA_FFS, lbl='flux0_BA', print_scale=print_scale_flux)
			k_BA_FFS_mean, k_BA_FFS_min, k_BA_FFS_max, d_k_BA_FFS = get_log_errors(ln_k_BA_FFS, d_ln_k_BA_FFS, lbl='k_BA_FFS', print_scale=print_scale_k)
		
		ThL_lbl = '$\epsilon_{11, 22}/T = ' + my.f2s(e[1,1]) + ',' + my.f2s(e[2,2]) + '$; $\mu_{1,2}/T = ' + my.f2s(mu[1]) + ',' + my.f2s(mu[2]) + '$; L = ' + str(L)
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
		ax_F.legend()
		
	else:
		print('ERROR: mode=%s is not supported' % (mode))

	plt.show()

if(__name__ == "__main__"):
	main()

