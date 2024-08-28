import argparse

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import scipy
from numba import jit

import mylib as my

rand_time = 0
N_gauss_rands = 1000000
D0_for_data = 0.53
N0_for_data = 50

def get_F_fnc(mode, F_scale=1, Ncl_scale=1):
	Ncl = np.array([1, 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 117, 125, 133, 141, 149, 157, 165, 173, 181, 189, 197, 205, 213, 221, 229, 237, 245, 253, 261, 269, 277, 285, 293, 301, 309, 317, 325, 333, 341, 349, 357, 365, 373, 381, 389, 397], dtype=int) * Ncl_scale
	F_data = np.array([20, 0.00000000e+00, -2.67073772e+00, -2.10564012e+00, -1.25198896e+00, -6.79585889e-01, -2.71254170e-01, 8.75065242e-02, 7.15283205e-03, -1.00340632e-01, -2.97704696e-01, -4.30221936e-01, -5.99914870e-01, -8.57073279e-01, -1.07214088e+00, -1.29701102e+00, -1.54623396e+00, -1.74680853e+00, -1.94689559e+00, -2.20434883e+00, -2.38196263e+00, -2.51789735e+00, -2.69507291e+00, -2.83593969e+00, -2.92908237e+00, -2.95898174e+00, -2.99012348e+00, -3.00142606e+00, -3.00303263e+00, -2.96182397e+00, -2.83723393e+00, -2.68325558e+00, -2.49843071e+00, -2.23309525e+00, -1.91149376e+00, -1.51708915e+00, -1.08459350e+00, -6.36255620e-01, -2.46464207e-01, 2.94872607e-01, 9.95470058e-01, 1.51240623e+00, 2.38369195e+00, 4.27749039e+00, 7.04946359e+00, 1.36161360e+01, 1.36161360e+01, 1.36161360e+01, 1.36161360e+01, 1.36161360e+01, 1.36161360e+01]) * F_scale
	d_F_data = np.array([0.1, 1.51801446e-02, 3.90773313e-03, 5.23658553e-03, 8.08525698e-03, 1.07904271e-02, 1.32483080e-02, 1.58611639e-02, 1.52347034e-02, 1.44349392e-02, 1.30735240e-02, 1.22316800e-02, 1.12316276e-02, 9.86812130e-03, 8.85409429e-03, 7.90326001e-03, 6.96582079e-03, 6.29064084e-03, 5.68017634e-03, 4.97755781e-03, 4.54158086e-03, 4.23226494e-03, 3.85821105e-03, 3.58251746e-03, 3.41004109e-03, 3.35625513e-03, 3.30102828e-03, 3.28118204e-03, 3.27836954e-03, 3.35118132e-03, 3.58006868e-03, 3.88218768e-03, 4.27533028e-03, 4.90442800e-03, 5.78387907e-03, 7.06959223e-03, 8.79862637e-03, 1.10282124e-02, 1.34142221e-02, 1.75987334e-02, 2.49960044e-02, 3.23760406e-02, 5.00620378e-02, 1.29061019e-01, 5.16095768e-01, 1.37614973e+01, 1.37614973e+01, 1.37614973e+01, 1.37614973e+01, 1.37614973e+01, 1.37614973e+01]) * F_scale
	
	ok_inds = d_F_data < F_scale
	
	Ncl = Ncl[ok_inds]
	F_data = F_data[ok_inds]
	d_F_data = d_F_data[ok_inds]
	
	if(mode == 'lin'):
		F_fnc = scipy.interpolate.interp1d(Ncl, F_data, bounds_error=False, fill_value='extrapolate')
		F_grad_fnc = lambda n, dn=0.01, f=F_fnc: (f(n + dn) - f(n - dn)) / (2 * dn)
	elif(mode == 'cube'):
		F_fnc = scipy.interpolate.CubicSpline(Ncl, F_data)
		F_grad_fnc = lambda n, f=F_fnc: f(n, 1)
	
	return F_fnc, F_grad_fnc, Ncl, F_data, d_F_data
	
def get_D_fnc(D0, N0=1, mode='const'):
	if(mode == 'const'):
		D_fnc = lambda n, d0=D0: (n if(isinstance(n, float)) else np.ones(n.shape)) * d0 * (n >= 0)
		D_grad_fnc = lambda n : 0 if(isinstance(n, float)) else np.zeros(n.shape)
	elif(mode == 'surf'):
		D_fnc = lambda n, n0=N0, d0=D0: d0 * np.sqrt(n / n0) * (n >= 0)
		D_grad_fnc = lambda n, n0=N0, d0=D0: d0 / (2 * np.sqrt(n * n0)) * (n >= 0)
	
	return D_fnc, D_grad_fnc

def get_dt_thr(Ncl, alp, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc):
	D = D_fnc(Ncl)
	dD = D_grad_fnc(Ncl)
	F = F_fnc(Ncl)
	dF = F_grad_fnc(Ncl)
	d2FD = dF**2 + (dD / D)**2
	if(isinstance(Ncl, np.ndarray)):
		dt_thr_F_arr = np.where(abs(dD)/D < 1e-14, np.inf, (np.sqrt(1 + ((alp / dF)**2) * d2FD) - 1) / (D * d2FD)) 
		dt_thr_D_arr = np.where(abs(dF) < 1e-14, np.inf, (np.sqrt(1 + ((alp * D/dD)**2) * d2FD) - 1) / (D * d2FD))
		dt_thr_arr = np.where(abs(dF) < 1e-14, np.inf, (np.sqrt(1 + d2FD) - 1) / (D * d2FD))
		def get_min_and_argmin(x, y, fnc=lambda arr: np.argmin(arr)):
			i = fnc(y)
			return x[i], y[i]
		
		dt_F_thr_Ncl, dt_F_thr = get_min_and_argmin(Ncl, dt_thr_F_arr)
		dt_D_thr_Ncl, dt_D_thr = get_min_and_argmin(Ncl, dt_thr_D_arr)
		dt_thr_Ncl, dt_thr = get_min_and_argmin(Ncl, dt_thr_arr)
	else:
		dt_thr_F_arr = (np.sqrt(1 + ((alp / dF)**2) * d2FD) - 1) / (D * d2FD) if(abs(dF) > 1e-14) else np.inf
		dt_thr_D_arr = (np.sqrt(1 + ((alp * D/dD)**2) * d2FD) - 1) / (D * d2FD) if(abs(dD)/D > 1e-14) else np.inf
		dt_thr_arr = min((np.sqrt(1 + d2FD) - 1) / (D * d2FD), 1/(2*D))
		dt_F_thr_Ncl, dt_F_thr = Ncl, dt_thr_F_arr
		dt_D_thr_Ncl, dt_D_thr = Ncl, dt_thr_D_arr
		dt_thr_Ncl, dt_thr = Ncl, dt_thr_arr
	
	return min([dt_F_thr, dt_D_thr, dt_thr]), dt_F_thr_Ncl, dt_D_thr_Ncl, dt_thr_Ncl

def get_dNcl(Ncl, dt, F_fnc, dF_fnc, D_fnc, dD_fnc):
	D = D_fnc(Ncl)
	
	global gauss_rands
	global rand_time
	if(rand_time == 0):
		gauss_rands = np.random.normal(size=N_gauss_rands)
	dNcl = (dD_fnc(Ncl) - D * dF_fnc(Ncl)) * dt + np.sqrt(2 * D * dt) * gauss_rands[rand_time]
	rand_time = (rand_time + 1) % N_gauss_rands
	
	return dNcl

# def get_dNcl_jit(Ncl, dt, F_fnc, dF_fnc, D_fnc, dD_fnc):
	# D = D_fnc(Ncl)
	# dNcl = (dD_fnc(Ncl) - D * dF_fnc(Ncl)) * dt + np.sqrt(2 * D * dt) * np.random.normal()
	
	# return dNcl

def run_dynamics(Ncl_init, dt, Nt, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc, \
				OP_A, OP_B, D_mode='const', \
				verbose=0, verbose_dt=1000):
	Ncl_evol = np.empty(Nt)
	times = np.empty(Nt)
	hA = np.empty(Nt)
	Ncl = Ncl_init
	time_passed = 0
	#hA[0] = 0 if(Ncl - OP_A > OP_B - Ncl) else 1
	for it in range(Nt):
		Ncl_evol[it] = Ncl
		
		if(dt < 0):
			dt_use, _, _, _ = get_dt_thr(Ncl, -dt, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc)
		else:
			dt_use = dt
		
		times[it] = dt_use
		dNcl = get_dNcl(Ncl, dt_use, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc)
		#dNcl = get_dNcl_jit(Ncl, dt, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc)
		
		#Ncl_new = Ncl + dNcl
		hA[it] = (0 if(Ncl - OP_A > OP_B - Ncl) else 1) if (it == 0) \
				else ((1 if(Ncl < OP_B) else 0) if(hA[it - 1] == 1) else (0 if(Ncl >= OP_A) else 1))
		#Ncl = Ncl_new
		Ncl += dNcl
		time_passed += dt_use
		
		if(verbose > 0):
			if(it % verbose_dt == 0):
				print('timeevol : %s %%            \r' % (my.f2s((it + 1) / Nt * 100)), end='')
	
	if(verbose > 0):
		print('timeevol DONE                   ')
	
	return Ncl_evol, times, hA

# def run_dynamics_jit(Ncl_init, dt, Nt, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc):
	# Ncl_evol = np.empty(Nt)
	# time_evol = np.empty(Nt)
	# Ncl = Ncl_init
	# time = 0
	# for it in range(Nt):
		# Ncl_evol[it] = Ncl
		
		# dNcl = get_dNcl_jit(Ncl, dt, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc)
		
		# Ncl += dNcl
		# time += dt
		# time_evol[it] = time
		
	# return Ncl_evol, time_evol

def main():
	
	# python browian_1D.py --dt 1e-2 --mode dynamics
	# python browian_1D.py --dt 1e-2 --D_mode surf --mode dynamics
	# python browian_1D.py --dt -0.05 --D_mode surf --Nt 1000000 --mode dynamics
	# python browian_1D.py --dt 1.0 --D_mode surf --Nt 5000000 --mode dynamics
	
	# python browian_1D.py --D_mode const --mode FPE_rate_est
	# python browian_1D.py --D_mode const --mode FPE_rate_est --F_scale 1.333
	
	# =============== parse =================
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--mode', \
						choices = ['FPE_rate_est', 'FPE_rate_est_exact', 'dynamics'], \
						help='what to do')
	
	parser.add_argument('--F_interp_mode', default='lin', \
						choices = ['lin', 'cube'], \
						help='how to interpilate F-profile')
	parser.add_argument('--F_scale', type=float, default=1, \
						help='scale F data')
	parser.add_argument('--Ncl_scale', type=float, default=1, \
						help='scale Ncl data')
	parser.add_argument('--D_mode', default='const', \
						choices = ['const', 'surf'], \
						help='how to scale D(Ncl)')
	parser.add_argument('--dt', type=float, default=-1e-1, \
						help='Brownian timestep')
	parser.add_argument('--Ncl_init', type=int, default=-2, \
						help='initial clusersize')
	parser.add_argument('--Ncl_min', type=int, default=1, \
						help='Wha to integrate from for FPE')
	parser.add_argument('--Ncl_B', type=int, default=150, \
						help='Wha to integrate to for FPE')
	parser.add_argument('--time_total', type=float, default=1e3, \
						help='traj time')
	parser.add_argument('--Nt', type=int, default=0, \
						help='number of timesteps to simulate')
	parser.add_argument('--dt_alpha', type=float, default=1e-1, \
						help='max dt estimation accuracy')
	parser.add_argument('--D0', type=float, default=D0_for_data, \
						help='the reference D in D(N0)')
	parser.add_argument('--Ncl0', type=float, default=N0_for_data, \
						help='N0 in the reference D(N0)')
	parser.add_argument('--my_seed', type=int, default=0, \
						help='init random seed')
						
	parser.add_argument('--font_mode', default='work', \
						help='Font sizes mode')
	parser.add_argument('--to_plot', type=int, default=1, \
						help='Wheather to plot on screen')
	parser.add_argument('--to_show_legend', type=int, default=1, \
						help='Wheather to put legend onaggregate plots')
	
	clargs = parser.parse_args()
	
	my_seed = clargs.my_seed
	mode = clargs.mode
	F_interp_mode = clargs.F_interp_mode
	D_mode = clargs.D_mode
	dt = clargs.dt
	F_scale = clargs.F_scale
	Ncl_scale = clargs.Ncl_scale
	Ncl_min = clargs.Ncl_min
	Ncl_B = clargs.Ncl_B
	#time_total = clargs.time_total
	dt_alpha = clargs.dt_alpha
	to_plot = clargs.to_plot
	
	my.font_mode = clargs.font_mode
	np.random.seed(my_seed)
	
	F_fnc, F_grad_fnc, Ncl_interp, F_data_interp, d_F_data_interp = \
		get_F_fnc(F_interp_mode, F_scale=F_scale, Ncl_scale=Ncl_scale)
	N_range = np.array([min(Ncl_interp), max(Ncl_interp)])
	#F_fnc, F_grad_fnc, _, _, _ = get_F_fnc(F_interp_mode)
	D_fnc, D_grad_fnc = get_D_fnc(clargs.D0, N0=clargs.Ncl0, mode=clargs.D_mode)
	
	Ncl_min_ind = np.argmin(np.abs(Ncl_min - Ncl_interp))
	Ncl_min = Ncl_interp[Ncl_min_ind]
	Ncl_B_ind = np.argmin(np.abs(Ncl_B - Ncl_interp))
	Ncl_B = Ncl_interp[Ncl_B_ind]
	
	if(clargs.Ncl_init < 0):
		Ncl_init = Ncl_interp[-clargs.Ncl_init]
	else:
		Ncl_init = clargs.Ncl_init
	Ncl_init_ind = np.argmin(np.abs(Ncl_init - Ncl_interp))
	
	if(mode in ['FPE_rate_est', 'FPE_rate_est_exact']):
		Ncl_TP_inds = (Ncl_interp >= Ncl_init) & (Ncl_interp < Ncl_B)
		Ncl_barrier_inds = (Ncl_interp >= 13) & (Ncl_interp < Ncl_B)
		
		#F_top_ind = np.argmax(F_data_interp[Ncl_TP_inds])
		F_top_ind = np.argmax(F_data_interp[Ncl_barrier_inds])
		Ncl_top = Ncl_interp[Ncl_barrier_inds][F_top_ind]
		Ncl_top_ind = np.argmax(Ncl_interp == Ncl_top)
		
		F_interp_a = (F_data_interp[1:] - F_data_interp[:-1]) / (Ncl_interp[1:] - Ncl_interp[:-1])
		F_interp_b = (Ncl_interp[1:] * F_data_interp[:-1] - Ncl_interp[:-1] * F_data_interp[1:]) / (Ncl_interp[1:] - Ncl_interp[:-1])
		
		Z2 = scipy.integrate.quad(lambda z, U_fnc=F_fnc: np.exp(-U_fnc(z)), min(Ncl_interp), max(Ncl_interp))[0]
		Ncl_est = np.linspace(min(Ncl_interp), max(Ncl_interp), 1000)
		Z1 = np.trapz(np.exp(-F_fnc(Ncl_est)), x=Ncl_est)
		Z = np.sum(np.exp(-F_interp_b) * (np.exp(-F_interp_a * Ncl_interp[1:]) - np.exp(-F_interp_a * Ncl_interp[:-1])) / (-F_interp_a))
		print('Z:', Z2, Z1/Z2, Z/Z2)
		
		Ncl_est = np.linspace(Ncl_min, Ncl_top, 1000)
		A_integr1 = np.trapz(np.exp(-F_fnc(Ncl_est))/D_fnc(Ncl_est), x=Ncl_est) / Z
		if(D_mode == 'const'):
			A_integr = np.sum(np.exp(-F_interp_b[Ncl_min_ind : Ncl_top_ind]) * (np.exp(-F_interp_a[Ncl_min_ind : Ncl_top_ind] * Ncl_interp[Ncl_min_ind+1 : Ncl_top_ind+1]) - np.exp(-F_interp_a[Ncl_min_ind : Ncl_top_ind] * Ncl_interp[Ncl_min_ind : Ncl_top_ind])) / (-F_interp_a[Ncl_min_ind : Ncl_top_ind])) / Z / D_fnc(1.0)
		elif(D_mode == 'surf'):
			A_integr_fnc = np.vectorize(lambda x, a: scipy.special.erf(x) if(a < 0) else scipy.special.erfi(x))
			A_integr = np.sum(np.exp(-F_interp_b[Ncl_min_ind : Ncl_top_ind]) / np.sqrt(np.abs(F_interp_a[Ncl_min_ind : Ncl_top_ind])) * \
							(A_integr_fnc(np.sqrt(np.abs(F_interp_a[Ncl_min_ind : Ncl_top_ind]) * Ncl_interp[Ncl_min_ind+1 : Ncl_top_ind+1]), -F_interp_a[Ncl_min_ind : Ncl_top_ind]) - \
							A_integr_fnc(np.sqrt(np.abs(F_interp_a[Ncl_min_ind : Ncl_top_ind]) * Ncl_interp[Ncl_min_ind : Ncl_top_ind]), -F_interp_a[Ncl_min_ind : Ncl_top_ind]))) \
						* (np.sqrt(clargs.Ncl0 * np.pi) / clargs.D0 / Z)
		A_integr2 = scipy.integrate.quad(lambda z, U_fnc=F_fnc, D_fnc=D_fnc: (np.exp(-U_fnc(z))/D_fnc(z)), Ncl_min, Ncl_top)[0] / Z
		
		print('A:', A_integr, A_integr1/A_integr, A_integr2/A_integr)
		
		TP_integr2 = scipy.integrate.quad(lambda y, U_fnc=F_fnc: np.exp(U_fnc(y)), Ncl_init, Ncl_B)[0] * Z
		Ncl_est = np.linspace(Ncl_init, Ncl_B, 1000)
		TP_integr1 = np.trapz(np.exp(F_fnc(Ncl_est)), x=Ncl_est) * Z
		TP_integr = np.sum(np.exp(F_interp_b[Ncl_init_ind : Ncl_B_ind]) * (np.exp(F_interp_a[Ncl_init_ind : Ncl_B_ind] * Ncl_interp[Ncl_init_ind+1 : Ncl_B_ind+1]) - np.exp(F_interp_a[Ncl_init_ind : Ncl_B_ind] * Ncl_interp[Ncl_init_ind : Ncl_B_ind])) / (F_interp_a[Ncl_init_ind : Ncl_B_ind])) * Z
		print('TP:', TP_integr, TP_integr1/TP_integr, TP_integr2/TP_integr)
		
		t_1stpass_approx = TP_integr * A_integr
		
		print('Ncl_min =', Ncl_min, '; Ncl_init =', Ncl_init, '; Ncl_top =', Ncl_top, '; Ncl_B =', Ncl_B)
		print('t_1stP_approx =', t_1stpass_approx)
		
		if(mode == 'FPE_rate_est_exact'):
			if(D_mode == 'const'):
				t_1stpass = scipy.integrate.dblquad(\
					lambda z, y, U_fnc=F_fnc: np.exp(U_fnc(y) - U_fnc(z)) /  D_fnc(1.0), \
					Ncl_init, Ncl_B, Ncl_min, lambda y: y)
			elif(D_mode == 'surf'):
				t_1stpass = scipy.integrate.dblquad(\
					lambda z, y, U_fnc=F_fnc, D_fnc=D_fnc: np.exp(U_fnc(y) - U_fnc(z)) /  D_fnc(z), \
					Ncl_init, Ncl_B, Ncl_min, lambda y: y)
			
			print('t_1stP =', t_1stpass)
			
			# D=const: <t> = 30631 +- 4 (approx 26821), k~3.26e-5
			# k_AB_1D = (12.8 +- 0.3) e-5 (dt=0.2); (2.88 +- 0.7)e-5 (dt=1.0)
			
			# D~surf: <t> = 46949 +- 7 (approx 44309), k~2.13e-5
			# k_AB_1D = (0.96 +- 0.2) e-5 (dt=0.2); (2.82 +- 0.7)e-5 (dt=1.0)
			# k_AB_AB_latt = (4.2 +- 1.0) e-5
		
		if(to_plot):
			fig_ps, ax_ps, _ = my.get_fig('$N_{cl}$', '$p_s(N_{cl}) / D(N_{cl})$')
			fig_psinv, ax_psinv, _ = my.get_fig('$N_{cl}$', '$1/p_s(N_{cl})$')
			
			Ncl_draw = np.linspace(Ncl_min, Ncl_top, 1000)
			ax_ps.plot(Ncl_draw, np.exp(-F_fnc(Ncl_draw))/D_fnc(Ncl_draw)/Z, label='inerp')
			
			Ncl_draw = np.linspace(Ncl_init, Ncl_B, 1000)
			ax_psinv.plot(Ncl_draw, np.exp(F_fnc(Ncl_draw))*Z, label='inerp')
		
	elif(mode == 'dynamics'):
		#Nt = int(time_total / dt + 0.5) if(clargs.Nt == 0) else clargs.Nt
		#time_total = dt * Nt
		Nt = clargs.Nt
		
		# ============ 
		
		if(dt > 0):
			if(D_mode == 'const'):
				F_grad_data_interp = F_grad_fnc(Ncl_interp[1:])
				dt_F_thr = (np.sqrt(1 + dt_alpha**2) - 1) / (clargs.D0 * max(F_grad_data_interp)**2)
				if(dt > dt_F_thr):
					print('WARNING: dt = %s = %s * dt_max (at alpha = %s), F_thr_ind = %d' % (my.f2s(dt), my.f2s(dt / dt_F_thr), dt_alpha, np.argmax(F_grad_data_interp) + 1))
			else:
				dt_thr, dt_F_thr_Ncl, dt_D_thr_Ncl = get_dt_thr(Ncl_interp[1:], dt_alpha, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc)
				if(dt > dt_thr):
					print('WARNING: dt = %s = %s * dt_max (at alpha = %s); F_thr_Ncl = %d, D_thr_Ncl = %d' % (my.f2s(dt), my.f2s(dt / dt_thr), dt_alpha, dt_F_thr_Ncl, dt_D_thr_Ncl))
		
		Ncl_evol, times = run_dynamics(Ncl_init, dt, Nt, F_fnc, F_grad_fnc, D_fnc, D_grad_fnc, verbose=1)
		time_evol = np.cumsum(times)
		
		# ===========
		
		if(to_plot):
			fig_Ncl_timeevol, ax_Ncl_timeevol, _ = my.get_fig('$t$ [sweep]', r'$N_{cl}$')
			
			ax_Ncl_timeevol.plot(time_evol, Ncl_evol, label='data')
			
			if(clargs.to_show_legend):
				my.add_legend(fig_Ncl_timeevol, ax_Ncl_timeevol)
			else:
				fig_Ncl_timeevol.tight_layout()
			
		
	if(to_plot):
		fig_F, ax_F, _ = my.get_fig(r'$N_{cl}$', '$F/T$')
		fig_D, ax_D, _ = my.get_fig(r'$N_{cl}$', '$D$ [1/sweep]')
		
		ax_F.errorbar(Ncl_interp, F_data_interp, yerr=d_F_data_interp, label='data')
		Ncl_draw = np.linspace(N_range[0], N_range[1] * 1.1, 1000)
		
		F_draw = F_fnc(Ncl_draw)
		ax_F.plot(Ncl_draw, F_draw, label='interp')
		minmax_F = [min(F_draw), max(F_draw)]
		ax_F.plot([Ncl_min] * 2, minmax_F, label=r'$N_{min} = %s$' % (my.f2s(Ncl_min)))
		ax_F.plot([Ncl_top] * 2, minmax_F, label=r'$N_{top} = %s$' % (my.f2s(Ncl_top)))
		ax_F.plot([Ncl_init] * 2, minmax_F, label=r'$N_{init} = %s$' % (my.f2s(Ncl_init)))
		ax_F.plot([Ncl_B] * 2, minmax_F, label=r'$N_{B} = %s$' % (my.f2s(Ncl_B)))
		
		D_draw = D_fnc(Ncl_draw)
		ax_D.plot(Ncl_draw, D_draw, label='interp')
		minmax_D = [min(D_draw), max(D_draw)]
		ax_D.plot([Ncl_min] * 2, minmax_D, label=r'$N_{min} = %s$' % (my.f2s(Ncl_min)))
		ax_D.plot([Ncl_top] * 2, minmax_D, label=r'$N_{min} = %s$' % (my.f2s(Ncl_top)))
		ax_D.plot([Ncl_init] * 2, minmax_D, label=r'$N_{min} = %s$' % (my.f2s(Ncl_init)))
		ax_D.plot([Ncl_B] * 2, minmax_D, label=r'$N_{min} = %s$' % (my.f2s(Ncl_B)))
		
		if(clargs.to_show_legend):
			my.add_legend(fig_F, ax_F)
			my.add_legend(fig_D, ax_D)
		else:
			fig_F.tight_layout()
			fig_D.tight_layout()
		
		plt.show()

if(__name__ == "__main__"):
	main()

