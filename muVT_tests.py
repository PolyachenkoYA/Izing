import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import shutil

import mylib as my
import table_data
#import lattice_gas
import izing

def proc_fit(x, y, d_y, clr, ylbl=None, d_x=None, to_plot=True, xlbl=r'$\mu_1$', n_draw=100, fig=None, ax=None, lbl='data', fit_lbl=lambda fit2: 'quad_fit', order=2):
	fit2 = np.polyfit(x, y, order, w=(1/d_y) if(d_y is not None) else np.ones(y.shape))
	
	if(to_plot):
		if(ax is None):
			fig, ax, _ = my.get_fig(xlbl, ylbl)
		x_draw = np.linspace(min(x), max(x), n_draw)
		
		errorbar_kwargs = {'fmt' : '.', 'label' : lbl, 'color' : clr}
		if(d_y is not None):
			errorbar_kwargs['yerr'] = d_y
		if(d_x is not None):
			errorbar_kwargs['xerr'] = d_x
		#ax.errorbar(x, y, yerr=d_y, fmt='.', label=lbl, color=clr)
		ax.errorbar(x, y, **errorbar_kwargs)
		ax.plot(x_draw, np.polyval(fit2 , x_draw), label=fit_lbl(fit2), color=clr)
		
		my.add_legend(fig, ax)
	
	return fit2, fig, ax

def get_mu0(x, y, y_ref, fit2):
	mu0_opt = scipy.optimize.root_scalar(lambda xx, fit2=fit2: (np.polyval(fit2, xx) - y_ref), \
										x0=x[np.argmin(np.abs(y - y_ref))], \
										fprime=lambda xx, fit2_prime=np.polyder(fit2): np.polyval(fit2_prime, xx))
	return mu0_opt.root

def fit_Ns(phi1_s, Ns, d_Ns=None, d_phi1_s=None, L=320, \
			bounds_mult=1.5, mode='dy', \
			lnNs_fitFnc = lambda x, prm: prm[1] - 2 * np.log(x - prm[0]), \
			lnPhi1_fitFnc = lambda y, prm: np.exp((prm[1] - y) / 2) + prm[0]):
	x = np.log(phi1_s)
	y = np.log(Ns)
	d_y = (1 / np.sqrt(Ns)) if(d_Ns is None) else (d_Ns / Ns)
	d_x = (1 / np.sqrt(phi1_s * L**2)) if(d_phi1_s is None) else (d_phi1_s / phi1_s)
	lnNs_fit_prm0 = np.empty(2)
	lnNs_fit_prm0[0] = (x[0] * np.exp((y[0] - y[-1]) / 2) - x[-1]) / (np.exp((y[0] - y[-1]) / 2) - 1)
	lnNs_fit_prm0[1] = y[0] + 2 * np.log(x[0] - lnNs_fit_prm0[0])
	lnNs_opt = scipy.optimize.minimize(lambda prm: np.sum(((lnNs_fitFnc(x, prm) - y) / d_y)**2) * ('dy' in mode) + np.sum(((lnPhi1_fitFnc(y, prm) - x) / d_x)**2) * ('dx' in mode), \
									lnNs_fit_prm0, method='SLSQP', \
									bounds=((lnNs_fit_prm0[0] - np.log(bounds_mult), lnNs_fit_prm0[0] + np.log(bounds_mult)), \
										   (lnNs_fit_prm0[1] - np.log(bounds_mult**2), lnNs_fit_prm0[1] + np.log(bounds_mult**2))))
	
	rho_c = np.exp(lnNs_opt.x[0])
	sgmT = np.sqrt(np.exp(lnNs_opt.x[1]) / np.pi)
	
	return rho_c, sgmT, lnNs_opt, lnNs_fitFnc

def mu1_phi1_interp():
	# ====== e11 = -2.68 =====
	# mu2 = np.array([4.92238326, 1e10])
	# mu1 = [np.array([5.1, 5.125, 5.15, 5.175, 5.2]), \
			# np.array([5.025, 5.05, 5.075, 5.1, 5.125, 5.15, 5.175, 5.2])]
	
	# phi1 = [np.array([1.39e-2, 1.2e-2, 1.1e-2, 1.02e-2, 0.97e-2]), \
			# np.array([0.01565, 0.01405, 0.01158, 0.0101, 0.009637, 0.009379, 0.008799, 0.008304])]
	# d_phi1 = [np.array([0.05e-2, 0.05e-2, 0.03e-2, 0.03e-2, 0.03e-2]), \
			# np.array([0.0003, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0001, 0.00014])]
	# dF = [np.array([1.41, 2.03, 3.15, 4.326, 6.344]), \
		# np.array([-0.0690407, 0.79863, 1.14676, 2.04316, 3.365314, 4.19758, 5.868074, 8.200319])]
	# d_dF = [np.array([0.09, 0.1, 0.1, 0.1, 0.08]), \
		# np.array([0.04048, 0.08102, 0.1194, 0.11894, 0.09664, 0.14713, 0.06713, 0.1743])]
	# Ns = [np.array([38.73, 43.56, 60.85705, 66.5, 84.8]), \
		# np.array([29.52248, 32.3859, 37.46289, 42.45807, 51.97981, 57.9825, 73.98546, 100.1522])]
	# d_Ns = [np.array([0.4, 0.2, 0.3, 0.5, 0.7]), \
		# np.array([0.2545, 1.0597, 0.8923, 0.2842, 0.4219, 1.791, 1.401, 1.5134])]
	
	'''
	'''
	
	# ====== e11 = -2.6666666666667 =====
	mu2 = np.array([1e10])
	mu1 = [np.array([5.025, 5.05, 5.075, 5.1, 5.125, 5.15, 5.175, 5.2])]
	
	phi1 = [np.array([0.01127993, 0.01092002, 0.01055463, 0.01005816, 0.009681567, 0.009336748, 0.008989878, 0.007863043])]
	d_phi1 = [np.array([0.00012997, 0.0001145, 0.0001611, 0.0001682, 0.00013127, 8.864e-05, 0.00015656, 0.00012324])]
	dF = [np.array([8.903408, 9.693966, 10.1888, 11.5485, 12.8307, 14.34696, 16.03613, 19.17258])]
	d_dF = [np.array([0.1592, 0.17655, 0.3256, 0.371, 0.4929, 0.09819, 0.2823, 0.11304])]
	Ns = [np.array([33.73477, 34.91058, 43.0346, 49.28451, 58.21434, 72.21557, 97.17265, 124.6213])]
	d_Ns = [np.array([0.9715, 0.7803, 1.967, 1.2, 0.8955, 1.6744, 1.827, 0.9167])]
	
	# ======================= processing ==========================
	N_mu2 = len(mu2)
	
	rho_c = np.empty(N_mu2)
	sgmT = np.empty(N_mu2)
	lnNs_opts = []
	lnNs_fitFnc = []
	for i2 in range(N_mu2):
		rho_c[i2], sgmT[i2], lnNs_opt_new, lnNs_fitFnc_new = \
			fit_Ns(phi1[i2], Ns[i2], d_Ns[i2], bounds_mult=2, mode='dx')
		lnNs_opts.append(lnNs_opt_new)
		lnNs_fitFnc.append(lnNs_fitFnc_new)
	
	fig_phi1, ax_phi1, _ = my.get_fig(r'$\mu_1$', r'$\phi_{1,bulk}$')
	fig_dF, ax_dF, _ = my.get_fig(r'$\mu_1$', r'$\Delta F/T$')
	fig_Ns, ax_Ns, _ = my.get_fig(r'$\mu_1$', r'$N^*$')
	fig_S, ax_S, _ = my.get_fig(r'$-e_{11} z/2 + \mu_1$', r'$\ln(\rho_1)$')
	fig_phiN, ax_phiN, _ = my.get_fig(r'$\phi_1$', r'$N^*$')#, xscl='log', yscl='log'
	
	e11 = -2.68010292
	for i2 in range(N_mu2):
		h = -e11 - mu1[i2]/2
		Ns_th = np.pi * (izing.sgm_th_izing(-e11) / (2 * h))**2
		
		clr_id = 3 * i2
		lbl = r'$\mu_2 = %s$' % my.f2s(mu2[i2])
		
		phi1_fit2, _, _  = \
			proc_fit(mu1[i2], phi1[i2], d_phi1[i2], fig=fig_phi1, ax=ax_phi1, \
					lbl=lbl, clr=my.get_my_color(1 + clr_id), to_plot=True)
		dF_fit2, _, _  = \
			proc_fit(mu1[i2], dF[i2], d_dF[i2], fig=fig_dF, ax=ax_dF, \
					lbl=lbl, clr=my.get_my_color(1 + clr_id), to_plot=True)
		Ns_fit2, _, _ = \
			proc_fit(mu1[i2], Ns[i2], d_Ns[i2], fig=fig_Ns, ax=ax_Ns, \
					lbl=lbl, clr=my.get_my_color(1 + clr_id), to_plot=True)
		S_fit, _, _ = \
					proc_fit(2 * h, np.log(phi1[i2]), d_phi1[i2]/phi1[i2], order=1, \
					fig=fig_S, ax=ax_S, lbl=lbl, \
					fit_lbl=lambda fit: (r'$\rho_c = %s$, $k = %s$' % (my.f2s(np.exp(fit[1])), my.f2s(fit[0]))), \
					clr=my.get_my_color(1 + clr_id), to_plot=True)
		rho_c_linfit = np.exp(S_fit[1])
		
		ax_Ns.plot(mu1[i2], Ns_th, label='CNT', color=my.get_my_color(0))
		
		phi1_draw = np.linspace(min(phi1[i2]), max(phi1[i2]), 100)
		ax_phiN.errorbar(phi1[i2], Ns[i2], yerr=d_Ns[i2], xerr=d_phi1[i2], fmt='.', label=lbl, color=my.get_my_color(1 + clr_id))
		ax_phiN.plot(phi1_draw, np.exp(lnNs_fitFnc[i2](np.log(phi1_draw), lnNs_opts[i2].x)), label=r'$\phi_c = %s$' % (my.f2s(rho_c[i2])), color=my.get_my_color(1 + clr_id))
		
		dF_ref = 12    # dF/T
		mu0 = get_mu0(mu1[i2], dF[i2], dF_ref, dF_fit2)
		#mu0 = get_mu0(mu, Ns, 35, Ns_fit2)
		
		Ns_0 = np.polyval(Ns_fit2, mu0)
		phi1_0 = np.polyval(phi1_fit2, mu0)
		dF0 = np.polyval(dF_fit2, mu0)
		
		ax_dF.plot([min(mu1[i2]), mu0], [dF0] * 2, '--', label=r'$\Delta F_0 / T = %s$' % my.f2s(dF0), color=my.get_my_color(1 + clr_id))
		ax_dF.plot([mu0] * 2, [min(dF[i2]), dF0], '--', label=r'$\mu_{1-0} = %s$' % my.f2s(mu0, n=4), color=my.get_my_color(1 + clr_id))
		ax_Ns.plot([min(mu1[i2]), mu0], [Ns_0] * 2, '--', label=r'$N^*_0 = %s$' % my.f2s(Ns_0), color=my.get_my_color(1 + clr_id))
		ax_Ns.plot([mu0] * 2, [min(Ns[i2]), Ns_0], '--', label=r'$\mu_{1-0} = %s$' % my.f2s(mu0, n=4), color=my.get_my_color(1 + clr_id))
		ax_phi1.plot([min(mu1[i2]), mu0], [phi1_0] * 2, '--', label=r'$\phi_{1,0} = %s$' % my.f2s(phi1_0, n=5), color=my.get_my_color(1 + clr_id))
		ax_phi1.plot([mu0] * 2, [min(phi1[i2]), phi1_0], '--', label=r'$\mu_{1-0} = %s$' % my.f2s(mu0, n=4), color=my.get_my_color(1 + clr_id))
	
	my.add_legend(fig_dF, ax_dF)
	my.add_legend(fig_phi1, ax_phi1)
	my.add_legend(fig_Ns, ax_Ns)
	my.add_legend(fig_phiN, ax_phiN)
	
	
	plt.show()

def supersat(verbose=1, to_plot=True):
	phi1_s = np.array([0.011, 0.01105, 0.0111, 0.01115, 0.0112, 0.01125, 0.0113, 0.01135, 0.0114, 0.01145, 0.0115, 0.01155, 0.0116, 0.01165, 0.0117, 0.01175, 0.0118, 0.01185, 0.0119, 0.01195, 0.012, 0.01205, 0.0121, 0.01215, 0.0122, 0.01225])
	ID_s = [np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int)]
	
	'''
	phi1_s = np.array([0.0121, 0.0119, 0.0117, 0.01145, 0.0112, 0.0111])
	ID_s = [np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int), \
			np.array([1000, 1001, 1002, 1003], dtype=int)]
	'''
	
	N_models = len(phi1_s)
	assert(len(ID_s) == N_models)
	
	OPset_name = 'nvt21'
	phi2 = 0.01
	nFFS_init = 60
	nFFS = 30
	
	OP_interfaces = table_data.OP_interfaces_table[OPset_name]
	
	def no_empty_split_cmd(cmd, spr=' '):
		return list(filter(None, my.run_it(cmd, check=True, verbose=False).split(spr)))
		#list(filter(None, s.split(spr)))
	def log_interval_parse(res):
		if(len(res[4].split(')')) > 1):
			y_scale = float(res[4].split(')')[1])
			y = float(res[2][1:]) * y_scale
			d_y = float(res[4].split(')')[0]) * y_scale
			return np.log(y), np.log((y + d_y) / (y - d_y)) / 2
		else:
			y_scale = float(res[3].split(']')[1])
			y_min = float(res[2][1:-1]) * y_scale
			y_max = float(res[3].split(']')[0]) * y_scale
			return np.log(y_min * y_max) / 2, np.log(y_max / y_min) / 2
	
	ln_flux0 = np.empty((2, N_models))
	ln_kAB = np.empty((2, N_models))
	Ns = np.empty((2, N_models))
	ZeldovichG = np.empty((2, N_models))
	Dtop = np.empty((2, N_models))
	dF = np.empty((2, N_models))
	dip = np.empty((2, N_models))
	Dtop_OP0_match = np.empty((2, N_models))
	for i in range(N_models):
		res = my.run_it('python run.py -mode FFS_AB_many -L 320 -OP_interfaces_set_IDs %s -to_get_timeevol 0 -N_states_FFS %d -N_init_states_FFS %d -init_composition %s %s %s -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -to_recomp 0 -Dtop_Nruns 300 -my_seeds %s -to_show_on_screen 0 -to_plot 0' % \
					(OPset_name, nFFS, nFFS_init, my.f2s(1 - phi2 - phi1_s[i], n=5), my.f2s(phi1_s[i], n=5), my.f2s(phi2, n=5), ' '.join([str(j) for j in ID_s[i]])), \
					check=True)
		if(verbose > 0):
			print(res)
		
		res_flux0 = no_empty_split_cmd(r'echo "%s" | grep flux0_AB' % res)
		if(verbose > 1):
			print(res_flux0)
		ln_flux0[0, i], ln_flux0[1, i] = log_interval_parse(res_flux0)
		if(verbose > 0):
			print('ln_flux0 =', ln_flux0[0, i], '+-', ln_flux0[1, i])
		
		res_kAB = no_empty_split_cmd(r'echo "%s" | grep k_AB_FFS' % res)
		if(verbose > 1):
			print(res_kAB)
		ln_kAB[0, i], ln_kAB[1, i] = log_interval_parse(res_kAB)
		if(verbose > 0):
			print('ln_kAB =', ln_kAB[0, i], '+-', ln_kAB[1, i])
		
		res_Ns = no_empty_split_cmd(r'echo "%s" | grep "N\* ="' % res)
		if(verbose > 1):
			print(res_Ns)
		Ns[0, i] = float(res_Ns[2])
		Ns[1, i] = float(res_Ns[4])
		if(verbose > 0):
			print('Ns =', Ns[0, i], '+-', Ns[1, i])
		
		res_ZeldovichG = no_empty_split_cmd(r'echo "%s" | grep "ZeldovichG ="' % res)
		if(verbose > 1):
			print(res_ZeldovichG)
		ZeldovichG[0, i] = float(res_ZeldovichG[2])
		ZeldovichG[1, i] = float(res_ZeldovichG[4])
		if(verbose > 0):
			print('ZeldovichG =', ZeldovichG[0, i], '+-', ZeldovichG[1, i])
		
		res_Dtop = no_empty_split_cmd(r'echo "%s" | grep "Dtop ="' % res)
		if(verbose > 1):
			print(res_Dtop)
		Dtop[0, i] = float(res_Dtop[2])
		Dtop[1, i] = float(res_Dtop[4])
		if(verbose > 0):
			print('Dtop =', Dtop[0, i], '+-', Dtop[1, i])
		
		res_dF = no_empty_split_cmd(r'echo "%s" | grep "dF/T ="' % res)
		if(verbose > 1):
			print(res_dF)
		dF[0, i] = float(res_dF[2])
		dF[1, i] = float(res_dF[4])
		if(verbose > 0):
			print('dF =', dF[0, i], '+-', dF[1, i])
		
		res_dip = no_empty_split_cmd(r'echo "%s" | grep "rho_dip ="' % res)
		if(verbose > 1):
			print(res_dip)
		dip[0, i] = float(res_dip[2])
		dip[1, i] = float(res_dip[4])
		if(verbose > 0):
			print('dip =', dip[0, i], '+-', dip[1, i])
		
		#input('ok')
		
		Dtop_OP0_match[0, i] = OP_interfaces[np.argmin(np.abs(OP_interfaces - Ns[0, i]))] - Ns[0, i]
		Dtop_OP0_match[1, i] = Ns[1, i]
	
	closeOP0_inds = np.abs(Dtop_OP0_match[0, :]) < 100.0
	
	rho_c, sgmT, lnNs_opt, lnNs_fitFnc = \
		fit_Ns(phi1_s[closeOP0_inds], Ns[0, closeOP0_inds], Ns[1, closeOP0_inds], bounds_mult=1.5, mode='dy')
	
	slp_phi1s = (np.log(phi1_s[closeOP0_inds][:-1]) + np.log(phi1_s[closeOP0_inds][1:])) / 2
	slp = np.diff(np.log(Ns[0, closeOP0_inds])) / np.diff(np.log(phi1_s[closeOP0_inds]))
	d_slp = np.sqrt((Ns[1, closeOP0_inds][:-1] / Ns[0, closeOP0_inds][:-1])**2 + (Ns[1, closeOP0_inds][1:] / Ns[0, closeOP0_inds][1:])**2) / np.diff(np.log(phi1_s[closeOP0_inds]))
	
	if(to_plot):
		fig_extrapol, ax_extrapol, _ = my.get_fig(r'$\phi_{bulk}$', r'$N^*$')
		fig_dF, ax_dF, _ = my.get_fig(r'$\phi_{bulk}$', r'$\Delta F/T$')
		fig_OP0match, ax_OP0match, _ = my.get_fig(r'$\phi_{bulk}$', r'$N_{cl,closest} - N^*$')
		#fig_NsSlp, ax_NsSlp, _ = my.get_fig(r'$\ln(\phi_{bulk})$', r'$\partial \ln(\phi_{bulk}) / \partial \ln(N^*)$')
		
		ax_OP0match.errorbar(phi1_s, Dtop_OP0_match[0, :], yerr=Dtop_OP0_match[1, :], fmt='.', label='data')
		ax_OP0match.plot([min(phi1_s), max(phi1_s)], [0]*2, label='perfect OP0 match')
		
		ax_dF.errorbar(phi1_s[closeOP0_inds], dF[0, closeOP0_inds], yerr=dF[1, closeOP0_inds])
		
		x_draw = np.linspace(min(phi1_s[closeOP0_inds]), max(phi1_s[closeOP0_inds]), 100)
		ax_extrapol.errorbar(phi1_s[closeOP0_inds], Ns[0, closeOP0_inds], yerr=Ns[1, closeOP0_inds], fmt='.', label='data')
		ax_extrapol.plot(x_draw, np.exp(lnNs_fitFnc(np.log(x_draw), lnNs_opt.x)), label=r'$\phi_c = %s$, $\sigma/T = %s$' % (my.f2s(rho_c), my.f2s(sgmT)), color=my.get_my_color(2))
		ax_extrapol.plot(x_draw, np.exp(lnNs_fitFnc(np.log(x_draw), lnNs_fit_prm0)), '--', label=r'init: $\phi_c = %s$, $\sigma/T = %s$' % (my.f2s(np.exp(lnNs_fit_prm0[0])), my.f2s(np.sqrt(np.exp(lnNs_fit_prm0[1]) / np.pi))), color=my.get_my_color(3))
		print(phi1_s[closeOP0_inds])
		print(Ns[0, closeOP0_inds])
		print(Ns[1, closeOP0_inds])
		
		#ax_NsSlp.errorbar(slp_phi1s, 1/slp, yerr=d_slp/slp**2)
		
		my.add_legend(fig_extrapol, ax_extrapol)
		
		plt.show()
	
	
def main():
	mu1_phi1_interp()
	#supersat(verbose=0)

if(__name__ == "__main__"):
	main()


