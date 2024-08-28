import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import sys
import scipy.interpolate

import mylib as my
import table_data

def load_data(Temp=None, L=None, hmin=None, hmax=None, N_OP=None, N_runs=None, N_init=None, N_states=None, OP0=None, npy_filename=None):
	if(npy_filename is None):
		npy_filename = interface_mode + '_TempJ' + str(Temp) + '_L' + str(L) \
						+ '_hJ' + (str(hmin) + '_' + str(hmax)) \
						+ '_Nintrf' +  str(N_OP) \
						+ '_Nruns' + str(N_runs) + '_Ninit' + str(N_init_states[0]) \
						+ '_Nstates' + str(N_states) \
						+ '_OP0' + (str(OP0)) + '.npz'
	
	print('loading from "' + npy_filename + '"')
	
	npy_data = np.load(npy_filename)
	
	h_arr = npy_data['h_arr']
	k_AB_mean = npy_data['k_AB']
	k_AB_errbars = npy_data['d_k_AB']
	OP0_AB = npy_data['P_B']
	d_OP0_AB = npy_data['d_P_B']
	
	return h_arr, k_AB_mean, k_AB_errbars, OP0_AB, d_OP0_AB

def chi_fnc(K):
	return (1 - 1/np.sinh(2*K)**4 )**(1/8)

def sgm(T, J):
	K = J/T
	chi = chi_fnc(K)
	sgm_par = (2 * K + np.log(np.tanh(K))) * T
	sgm_diag = (np.sqrt(2) * np.log(np.sinh(2 * K))) * T
	return (sgm_par + sgm_diag) / (2 * np.sqrt(chi))

def sgm_exact_one(T, J, Kc=1/2.27, tol=1e-4):
	def q_fnc(K):
		return (3 - np.exp(-4*K)) / np.sinh(4*K)
	def m_fnc(K):
		return 8*(np.cosh(4*K) - 1) / (np.cosh(4*K) + 1)**2
	
	Tc = J/Kc
	K = J/T
	
	intgr = lambda a: ((1 - q_fnc(a)) * scipy.special.ellipk(m_fnc(a)))
	sgm2 = scipy.integrate.quad(intgr, Kc, K)
	sgm2_err = sgm2[1] * (16 / np.pi / chi_fnc(K))
	sgm2 = sgm2[0] * (16 / np.pi / chi_fnc(K))
	
	if(sgm2_err / sgm2 > tol):
		print('WARNING: T/T_c = ', T/Tc, '; d_int / int ~ ', sgm2_err / sgm2)
	
	#print(T, J, K - Kc, sgm2)
	
	if(sgm2 < 0):
		fig, ax = my.get_fig('K', 'f', title='K=J/T = ' + my.f2s(K))
		K_draw = np.linspace(Kc, K, 1000)
		ax.plot(K_draw, intgr(K_draw), label='int')
		ax.plot(K_draw, scipy.special.ellipk(m_fnc(K_draw)), label='ell')
		ax.plot(K_draw, 1 - q_fnc(K_draw), label='1-q')
		ax.plot(K_draw, [0] * len(K_draw), label='y=0')
		ax.plot([Kc] * 2, [-1, 10], label='Kc')
		ax.legend()
	
	return np.sqrt(sgm2) * T

sgm_paper = scipy.interpolate.interp1d(table_data.sgm_analyt[:, 0], table_data.sgm_analyt[:, 1])

sgm_exact = np.vectorize(sgm_exact_one)
sgm_exact = lambda T, J: sgm_paper(T/J) * J


def add_data(data, h_min, h_max, T, lbl, i, ax_Nc=None, ax_k=None, to_norm_sgm=0, T_units=False):
	h = data[0, :]
	draw_inds = (h_min <= h) & (h <= h_max)
	h = h[draw_inds]
	Nc = data[1, draw_inds]
	Nc_th = np.pi / 4 / h**2
	if(to_norm_sgm == 0):
		Nc_th *= sgm(T, 1)**2
		if(T_units):
			Nc *= sgm(T, 1)**2;
	else:
		if(not T_units):
			if(to_norm_sgm == 1):
				Nc /= sgm(T, 1)**2
			elif(to_norm_sgm == 2):
				Nc /= sgm_exact(T, 1)**2
	
	#dy = data[2, :] - data[1, :]
	#ax.errorbar(x, y, yerr=dy, label=lbl, fmt='.--')
	if(ax_Nc is not None):
		T_mult = (T if(T_units) else 1)
		ax_Nc.plot(h, Nc, '+-' if(T_units) else '.-', label=lbl, color=my.get_my_color(i))
		ax_Nc.plot(h, Nc_th, '--', label=None, color=my.get_my_color(i))
	

#Temp = 1.5
Tc = 2.27
Jc = 1 / Tc

[qiwei_h, qiwei_A] = np.load("string_2d_Ising_V3_2_critical_nucleus_size_thresholding.npy", allow_pickle=True)
# "string_2d_Ising_V3_2_critical_nucleus_size_material.npy"
table_data.qiwei_data[0, :] = qiwei_h
table_data.qiwei_data[1, :] = qiwei_A

#######################################################################################################################
data41 = np.load(
    "string_2d_Ising_V4_1_critical_nucleus_size_normalized_gamma.npy", allow_pickle=True
)
data42 = np.load(
    "string_2d_Ising_V4_2_critical_nucleus_size_normalized_gamma.npy", allow_pickle=True
)

#Jc = 1 / 4
#J41 = Jc * 2.27 / 1.5
#J42 = Jc * 2.27 / 1.9
#######################################################################################################################

h_min = min(table_data.my_data[0, :]) * 0.9
h_max = max(table_data.my_data[0, :]) * 1.1
h_min = 0
h_max = 100

h_2p0, kAB_2p0, _, Nc_2p0, _ = load_data(npy_filename='CS_TempJ2.0_L32_hJ0.04_0.09_Nintrf9_Nruns20_Ninit800_Nstates400_OP024.npz')
Nc_data_2p0 = np.concatenate((h_2p0[np.newaxis, :], Nc_2p0[np.newaxis, :]))

#data = [table_data.qiwei_data, Nc_data_2p0, table_data.Nc_reference_data['1.9'], table_data.Nc_reference_data['1.5'], table_data.Nc_reference_data['1.0']]
data = [np.concatenate((data41[0][np.newaxis, :], data41[1][np.newaxis, :])), \
		np.concatenate((data42[0][np.newaxis, :], data42[1][np.newaxis, :])), \
		Nc_data_2p0, table_data.Nc_reference_data['1.9'], table_data.Nc_reference_data['1.5'], table_data.Nc_reference_data['1.0']]
N = len(data)

print()

#Temps = np.array([1.5, 2.0, 1.9, 1.5, 1.0])
Temps = np.array([1.5, 1.9, 2.0, 1.9, 1.5, 1.0])
Q_ind = 2
lbls = [('Qiwei, J/T = ' + my.f2s(1.0/Temps[i])) for i in range(Q_ind)] + [('Yury, J/T = ' + my.f2s(1.0/Temps[i])) for i in range(Q_ind, N)]

assert(N == len(lbls))


#fig_k, ax_k = my.get_fig('h/J', r'$k_{AB}$ [trans / flips]', title=r'$k_{AB}(h)$; T/J = ' + str(Temp), xscl='log', yscl='log')
fig_Nc, ax_Nc = my.get_fig('$h/J$', r'$N_c$', title=r'$N_c(h)$, $(J / T)_c = ' + my.f2s(Jc) + '$', xscl='log', yscl='log')
fig_NcNorm, ax_NcNorm = my.get_fig('$h/J$', r'$N_c/\sigma_{T \sim T_c}^2$', title=r'$N_c(h)$, $(J / T)_c = ' + my.f2s(Jc) + '$', xscl='log', yscl='log')
#fig_NcNormEx, ax_NcNormEx = my.get_fig('$h/J$', r'$N_c/\sigma^2$', title=r'$N_c(h)$, $J_c / T = ' + my.f2s(Jc) + '$', xscl='log', yscl='log')
for i in range(N):
	add_data(data[i], h_min, h_max, Temps[i], lbls[i], i, ax_Nc = ax_Nc, T_units = (i < Q_ind), to_norm_sgm=0)
	add_data(data[i], h_min, h_max, Temps[i], lbls[i], i, ax_Nc = ax_NcNorm, T_units = (i < Q_ind), to_norm_sgm=1)
	#add_data(data[i], h_min, h_max, Temps[i], lbls[i], i, ax_Nc = ax_NcNormEx, T_units = (i < Q_ind), to_norm_sgm=2)

#ax_k.legend()
ax_Nc.legend()
ax_NcNorm.legend()
#ax_NcNormEx.legend()

fig_sgm, ax_sgm = my.get_fig('$K=J/T$', r'$\sigma/T$', title=r'$\sigma(T)$, $J_c / T = ' + my.f2s(Jc) + '$')
fig_sgm_T, ax_sgm_T = my.get_fig('$T/J$', r'$\sigma/J$', title=r'$\sigma(T)$, $(J / T)_c = ' + my.f2s(Jc) + '$')

x_draw = np.linspace(min(Temps), Tc * 0.99, 10)
sgm_draw = sgm(x_draw, 1.0)
#sgm_draw = sgm_paper(x_draw) * 1.0

ax_sgm.plot(1 / x_draw, sgm_draw / x_draw, label='$T > T_c/4$')
ax_sgm.plot(1 / x_draw, sgm_exact(x_draw, 1.0) / x_draw, label='$exact$')
ax_sgm.plot([1 / Tc] * 2, [min(sgm_draw / x_draw), max(sgm_draw / x_draw)], label='$K_c$')

ax_sgm_T.plot(x_draw, sgm_draw, '-x', label='$T > T_c/4$')
ax_sgm_T.plot(x_draw, sgm_exact(x_draw, 1.0), label='$exact$')
ax_sgm_T.plot([Tc] * 2, [min(sgm_draw), max(sgm_draw)], label='$K_c$')
#ax_sgm_T.plot(table_data.sgm_analyt[:, 0], table_data.sgm_analyt[:, 1], label='$paper$')

# fig_sgm_red, ax_sgm_red = my.get_fig('$T$', r'$\sigma/J \chi(T)$', title=r'$\sigma(T)$, $(J / T)_c = ' + my.f2s(Jc) + '$')
# J0 = 3/4
# x_draw = np.linspace(min(Temps) * 1e-3, Tc * 0.99, 10) * J0
# x_draw = np.linspace(min(Temps) * 0.9, Tc * 0.99, 10) * J0
# sgm_draw = sgm(x_draw, J0) / chi_fnc(J0 / x_draw) / J0
# ax_sgm_red.plot(x_draw, sgm_draw, label='$T > T_c/4$')
# ax_sgm_red.plot(x_draw, sgm_exact(x_draw, J0) / chi_fnc(J0 / x_draw) / J0, label='$exact$')
# ax_sgm_red.plot([Tc * J0] * 2, [min(sgm_draw), max(sgm_draw)], label='$K_c$')
# ax_sgm_red.legend()

ax_sgm.legend()
ax_sgm_T.legend()

plt.show()
