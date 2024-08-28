import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import shutil

import mylib as my

def get_P_matr(n, Ns, tau, dNcl, f_fnc, fp_fnc):
	P = np.zeros((n, n))
	P[0, 0] = 1
	P[n - 1, n - 1] = 1
	a_j = fp_fnc(dNcl)
	dNcl_low = np.arange(-Ns, dNcl[0] + 1)
	fp_dNcl_low = fp_fnc(dNcl_low)
	dNcl_max = dNcl[-1] + tau * max(a_j) + 2*np.sqrt(tau) * 10 + 2
	dNcl_high = np.arange(dNcl[-1], int(dNcl_max + 0.5))
	fp_dNcl_high = fp_fnc(dNcl_high)
	for j in range(1, n - 1):
		#P[j-1 : j+2, j] = np.exp(-f_fnc(dNcl[j-1 : j+2]))

		#P[1:-1, j] = np.exp(-(dNcl[1:-1] - dNcl[j] + a_j[1:-1] * tau)**2 / (4 * tau))# / np.sqrt(4 * np.pi * tau)
		#P[0, j] = np.sum(np.exp(-(dNcl_low - dNcl[j] + fp_dNcl_low * tau)**2 / (4 * tau)))# / np.sqrt(4 * np.pi * tau)
		#P[-1, j] = np.sum(np.exp(-(dNcl_high - dNcl[j] + fp_dNcl_high * tau)**2 / (4 * tau)))# / np.sqrt(4 * np.pi * tau)

		P[1:-1, j] = (scipy.special.erf(((dNcl[1:-1] - dNcl[j] + a_j[1:-1] * tau) + 1/2) / np.sqrt(4 * tau)) - \
					 scipy.special.erf(((dNcl[1:-1] - dNcl[j] + a_j[1:-1] * tau) - 1/2) / np.sqrt(4 * tau)))
		P[0, j] = (scipy.special.erf((1/2 + (dNcl[0] - dNcl[j] + a_j[0] * tau)) / np.sqrt(4 * tau)) - \
					 scipy.special.erf((-Ns) / np.sqrt(4 * tau)))
		P[-1, j] = (scipy.special.erfc(((dNcl[-1] - dNcl[j] + a_j[-1] * tau) - 1/2) / np.sqrt(4 * tau)))

		P[:, j] /= np.sum(P[:, j])

	return P

def plot_matr(P, tit):
	fig_P, ax_P, figID_P = my.get_fig('j', 'i', title=tit)
	im_P = ax_P.imshow(np.flip(P, axis=0), interpolation ='none', origin ='lower')
	plt.figure(figID_P)
	plt.colorbar(im_P)


def test_P(n_1side, tau, Ns, f_fnc, fp_fnc):
	n = 2 * n_1side + 1
	dNcl = np.arange(n) - n_1side

	P1 = get_P_matr(n, Ns, tau, dNcl, f_fnc, fp_fnc)
	P2 = get_P_matr(n, Ns, 2 * tau, dNcl, f_fnc, fp_fnc)
	dP = P2 - P1**2

	plot_matr(P1, '$P_1$')
	plot_matr(P2, '$P_2$')
	plot_matr(np.maximum(np.log(np.abs(dP)), -7), '$\ln|P_2 - P_1^2|$')

	plt.show()

def master_eq(n_1side, tau, Ns, Nt=None, f_fnc=None, fp_fnc=None, G=1, to_plot=False):
	n = 2 * n_1side + 1
	#tau = D * dt

	if(G < 0):
		G *= -1 / np.sqrt(np.pi) / n_1side
	if(f_fnc is None):
		f_fnc = lambda dNcl, g=G: (-np.pi * g**2) * dNcl**2
		fp_fnc = lambda dNcl, g=G: (-2 * np.pi * g**2) * dNcl

	dNcl = np.arange(n) - n_1side

	P = get_P_matr(n, Ns, tau, dNcl, f_fnc, fp_fnc)

	#print('P:', P)

	Nt_relax = int(n_1side / (2 * tau) + 0.5)
	if(Nt is None):
		Nt = -2
	if(Nt < 0):
		Nt *= -Nt_relax
	t_relax = Nt_relax * tau

	s = np.empty((Nt + 1, n))
	s[0, :] = np.zeros(n)
	s[0, n_1side] = 1
	#print('s0:', s[0, :])

	t_arr = (np.arange(Nt) + 1) * tau
	dNcl_mean = np.empty(Nt)
	dNcl_msd2 = np.empty(Nt)
	for it in range(1, Nt + 1):
		s[it, :] = np.dot(P, s[it-1, :])
		#assert(np.all(s[it, :] >= 0))

		dNcl_mean[it - 1] = np.average(dNcl[1:-1], weights=s[it, 1:-1])
		dNcl_msd2[it - 1] = np.average((dNcl[1:-1] - dNcl_mean[it - 1])**2, weights=s[it, 1:-1])

		print('done %s %%                 \r' % (my.f2s(it / Nt * 100)), end='')
	print('done                    ')
	#print(s)

	if(to_plot):
		fig_f, ax_f, _ = my.get_fig(r'$\Delta N_{cl} = N_{cl} - N^*$', 'F/T', title='$(F/T)(\Delta N_{cl})$')
		ax_f.plot(dNcl, f_fnc(dNcl), '--.')

		fig_sEvolLn, ax_sEvolLn, figID_sEvolLn = my.get_fig('Dt', r'$N_{cl}$', title=r'$\ln[\rho(t, N_{cl})]$')
		fig_sEvol, ax_sEvol, figID_sEvol = my.get_fig('Dt', r'$N_{cl}$', title=r'$\rho(t, N_{cl})$')
		fig_mean, ax_mean, _ = my.get_fig('Dt', r'$\langle N_{cl} \rangle$')
		fig_msd2, ax_msd2, _ = my.get_fig('Dt', r'$\langle (\Delta N_{cl})^2 \rangle$')

		plot_matr(P, r'$P(i, D \delta t = \tau | j, 0)$; $\tau = %s$' % my.f2s(tau))

		im_sEvolLn = ax_sEvolLn.imshow(np.log(s).T, extent = [0, max(t_arr), min(dNcl), max(dNcl)], \
				interpolation ='bilinear', origin ='lower', aspect='auto')
		plt.figure(figID_sEvolLn)
		plt.colorbar(im_sEvolLn)
		im_sEvol = ax_sEvol.imshow(s.T, extent = [0, max(t_arr), min(dNcl), max(dNcl)], \
				interpolation ='bilinear', origin ='lower', aspect='auto')
		plt.figure(figID_sEvol)
		plt.colorbar(im_sEvol)

		ax_mean.plot(t_arr, dNcl_mean, '--.', label='data')
		ax_mean.plot([t_relax] * 2, [min(dNcl_mean), max(dNcl_mean)], '--', label=r'$\tau_{relax}$')
		
		ax_msd2.plot(t_arr, dNcl_msd2, '--.', label='data')
		ax_msd2.plot([t_relax] * 2, [min(dNcl_msd2), max(dNcl_msd2)], '--', label=r'$\tau_{relax}$')
		
		#my.add_legend(fig_sEvol, ax_sEvol)
		my.add_legend(fig_msd2, ax_msd2)
		
		plt.show()
	
	return s, t_arr, dNcl_mean, dNcl_msd2

def n_var():
	G = 1/50
	ns = 35
	
	n_1s = np.array([3,4,5,6,7,8,9,10,12,15,20,30], dtype=int)
	
	N_ns = len(n_1s)
	D_s = np.empty(N_ns)
	print(n_1s)
	for i_n, n_1side in enumerate(n_1s):
		_, t_arr, _, msd2 = master_eq(n_1side, 1, ns, G=G, to_plot=True)#
		D_s[i_n] = msd2[0] / t_arr[0] / 2
	
	fig, ax, _ = my.get_fig('$|\Delta N_{cl}|_{max}$', '$D$', xscl='log')
	
	ax.plot(n_1s, D_s)
	
	plt.show()

def D_var():
	tau_s = np.array([4, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02])
	N_taus = len(tau_s)
	D_s = np.empty(N_taus)
	for it, tau in enumerate(tau_s):
		_, t_arr, _, msd2 = master_eq(6, tau, 35, Nt = -2, G=1/50)

		#slp[it] = (msd2[1] - msd2[0]) / tau
		#msd2_linfit = np.polyfit(t_arr[:4], msd2[:4], 1)
		#slp[it] = np.mean(msd2[:4]) / np.mean(t_arr[:4])
		D_s[it] = msd2[0] / t_arr[0]
		#print(slp[it])


	fig_slp, ax_slp, _ = my.get_fig(r'$\tau = D_{th} \delta t$', r'$D / D_{th}$', xscl='log', yscl='log')
	ax_slp.plot(tau_s, D_s)
	my.add_legend(fig_slp, ax_slp)

	plt.show()

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

	#avg_times_Dtop()

	f_fnc=lambda dNcl: -0.3 * dNcl**2 + 0.01 * dNcl**3
	fp_fnc=lambda dNcl: -0.3 * 2 * dNcl + 0.01 * 3 * dNcl**2

	f_fnc=None
	fp_fnc=None

	master_eq(5, 0.05, 35, G=0, to_plot=True, \
			f_fnc=f_fnc, fp_fnc=fp_fnc)

	#n_var()
	D_var()
	#test_P(5, 0.1, 35, f_fnc=f_fnc, fp_fnc=fp_fnc)


if(__name__ == "__main__"):
	main()

