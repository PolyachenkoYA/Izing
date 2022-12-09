//
// Created by ypolyach on 10/27/21.
//

#include <gsl/gsl_rng.h>
#include <cmath>
#include <optional>

#include <pybind11/pytypes.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <python3.8/Python.h>

#include <vector>

namespace py = pybind11;

#include "lattice_gas.h"

void test_my(int k, py::array_t<long> *_Nt, py::array_t<double> *_probs, py::array_t<double> *_d_probs, int l)
// A function for DEBUG purposes
{
	printf("checking step %d\n", k);
	py::buffer_info Nt_info = (*_Nt).request();
	py::buffer_info probs_info = (*_probs).request();
	py::buffer_info d_probs_info = (*_d_probs).request();
	printf("n%d: %ld\n", k, Nt_info.shape[0]);
	printf("p%d: %ld\n", k, probs_info.shape[0]);
	printf("d%d: %ld\n", k, d_probs_info.shape[0]);
}

py::int_ init_rand(int my_seed)
{
	lattice_gas::init_rand_C(my_seed);
	return 0;
}

py::int_ get_seed()
{
	return lattice_gas::seed;
}

py::int_ set_verbose(int new_verbose)
{
	lattice_gas::verbose_dafault = new_verbose;
	return 0;
}

int compute_hA(py::array_t<int> *h_A, int *OP, long Nt, int OP_A, int OP_B)
{
	py::buffer_info h_A_info = (*h_A).request();
	int *h_A_ptr = static_cast<int *>(h_A_info.ptr);
	h_A_ptr[0] = OP[0] - OP_A <= OP_B - OP[0] ? 1 : 0;   // if closer to A at t=0, then 1, else 0
	int i;
	for(i = 1; i < Nt; ++i){
		h_A_ptr[i] = h_A_ptr[i-1] == 1 ? (OP[i] <= OP_B ? 1 : 0) : (OP[i] > OP_A ? 0 : 1);
	}

	return 0;
}

void print_state(py::array_t<int> state)
{
	py::buffer_info state_info = state.request();
	int *state_ptr = static_cast<int *>(state_info.ptr);
	assert(state_info.ndim == 1);

	int L2 = state_info.shape[0];
	int L = lround(sqrt(L2));
	assert(L * L == L2);   // check for the full square

	lattice_gas::print_S(state_ptr, L, 0);
}

py::tuple cluster_state(py::array_t<int> state, int default_state, std::optional<int> _verbose)
{
	py::buffer_info state_info = state.request();
	int *state_ptr = static_cast<int *>(state_info.ptr);
	assert(state_info.ndim == 1);

	int L2 = state_info.shape[0];
	int L = lround(sqrt(L2));
	assert(L * L == L2);   // check for the full square

	int verbose = (_verbose.has_value() ? _verbose.value() : lattice_gas::verbose_dafault);
	if(verbose > 5){
		lattice_gas::print_S(state_ptr, L, 'i');
	}

	py::array_t<int> cluster_element_inds = py::array_t<int>(L2);
	py::buffer_info cluster_element_inds_info = cluster_element_inds.request();
	int *cluster_element_inds_ptr = static_cast<int *>(cluster_element_inds_info.ptr);

	py::array_t<int> cluster_sizes = py::array_t<int>(L2);
	py::buffer_info cluster_sizes_info = cluster_sizes.request();
	int *cluster_sizes_ptr = static_cast<int *>(cluster_sizes_info.ptr);

	py::array_t<int> is_checked = py::array_t<int>(L2);
	py::buffer_info is_checked_info = is_checked.request();
	int *is_checked_ptr = static_cast<int *>(is_checked_info.ptr);

	int N_clusters = L2;

	lattice_gas::clear_clusters(cluster_element_inds_ptr, cluster_sizes_ptr, &N_clusters);
	lattice_gas::uncheck_state(is_checked_ptr, L2);
	lattice_gas::cluster_state_C(state_ptr, L, cluster_element_inds_ptr, cluster_sizes_ptr, &N_clusters, is_checked_ptr, default_state);

	return py::make_tuple(cluster_element_inds, cluster_sizes);
}

py::tuple run_bruteforce(int L, double **e, double *mu, long Nt_max, long N_saved_states_max,
						 std::optional<int> _N_spins_up_init, std::optional<int> _to_remember_timeevol,
						 std::optional<int> _OP_A, std::optional<int> _OP_B,
						 std::optional<int> _OP_min, std::optional<int> _OP_max,
						 std::optional<int> _interface_mode, std::optional<int> _default_spin_state,
						 std::optional<int> _verbose)
/**
 *
 * @param L - the side-size of the lattice
 * @param e - matrix NxN, particles interaction energies
 * @param mu - vector Nx1, particles chemical potentials
 * 		H = - \sum_{<ij>} s_i s_j e[s_i][s_j] - \sum_i s_i mu[s_i]
 * 		mu_i = -inf   ==>   n_i = 0
 * 		T = 1
 * @param h - magnetic-field-induced multiplier; unit=J, so it's h/J
 * @param Nt_max - for many succesful MC steps I want
 * @param N_spins_up_init - the number of up-spins in the initial frame. I always used 0 for my simulations
 * @param to_remember_timeevol - whether to record the time dependence of OPs
 * @param OP_A, OP_B - the boundaries of A and B to compute h_A
 * @param OP_min, OP_max - the boundaries at which the simulation is restarted. Here I always use [most_left_possible - 1; most_gight + 1] so the system never restarts.
 * @param interface_mode - Magnetization or CS
 * @param default_spin_state - what spin state if consudered for clustering. I always use -1
 * @param _verbose - int number >= 0 or py::none(), shows how load is the process; If it's None (py::none()), then the default state 'verbose' is used
 * @return :
 * 	(E, M, CS, h_A, time, k_AB)
 * 	E, M, CS, h_A, time - arrays [Nt]
 * 	k_AB = N_restarts / Nt_max
 */
{
	int i, j;
	int L2 = L*L;

	lattice_gas::set_OP_default(L2);

// -------------- check input ----------------
	assert(L > 0);
	int verbose = (_verbose.has_value() ? _verbose.value() : lattice_gas::verbose_dafault);
	int to_remember_timeevol = (_to_remember_timeevol.has_value() ? _to_remember_timeevol.value() : 1);
	int interface_mode = (_interface_mode.has_value() ? _interface_mode.value() : 1);   // 'M' mode
	assert((interface_mode >= 0) && (interface_mode < N_interface_modes));
	int default_spin_state = (_default_spin_state.has_value() ? _default_spin_state.value() : lattice_gas::verbose_dafault);
	int OP_min = (_OP_min.has_value() ? _OP_min.value() : lattice_gas::OP_min_default[interface_mode]);
	int OP_max = (_OP_max.has_value() ? _OP_max.value() : lattice_gas::OP_max_default[interface_mode]);
	assert(OP_max > OP_min);
	int OP_A = (_OP_A.has_value() ? _OP_A.value() : OP_min);
	int OP_B = (_OP_B.has_value() ? _OP_B.value() : OP_max);
	int N_spins_up_init = (_N_spins_up_init.has_value() ? _N_spins_up_init.value() : -1);

// ----------------- create return objects --------------
	long Nt = 0;
	int N_states_saved;
	int N_launches;
	long OP_arr_len = 128;   // the initial value that will be doubling when necessary
//	py::array_t<int> state = py::array_t<int>(L2);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets
//	py::buffer_info state_info = state.request();
//	int *state_ptr = static_cast<int *>(state_info.ptr);
	int *states = (int*) malloc(sizeof (int) * L2 * std::max((long)1, N_saved_states_max));

	double *_E;
	int *_M;
	int *_biggest_cluster_sizes;
	int *_h_A;
	int *_time;
	long time_total;
	if(to_remember_timeevol){
		_E = (double*) malloc(sizeof(double) * OP_arr_len);
		_M = (int*) malloc(sizeof(int) * OP_arr_len);
		_biggest_cluster_sizes = (int*) malloc(sizeof(int) * OP_arr_len);
		_h_A = (int*) malloc(sizeof(int) * OP_arr_len);
		_time = (int*) malloc(sizeof(int) * OP_arr_len);
	}

	lattice_gas::get_equilibrated_state(L, e, mu, states, interface_mode, default_spin_state, OP_A, OP_B, verbose);
	N_states_saved = 1;

	lattice_gas::run_bruteforce_C(L, e, mu, &time_total, INT_MAX, states,
							to_remember_timeevol ? &OP_arr_len : nullptr,
								  &Nt, &_E, &_M, &_biggest_cluster_sizes, &_h_A, &_time,
								  interface_mode, default_spin_state, OP_A, OP_B,
								  OP_min, OP_max, &N_states_saved,
								  OP_min, OP_A, save_state_mode_Inside,
								  N_spins_up_init, verbose, Nt_max, &N_launches, 0,
								  0, N_saved_states_max);

	int N_last_elements_to_print = std::min(Nt, (long)10);

	py::array_t<double> E;
	py::array_t<int> M;
	py::array_t<int> biggest_cluster_sizes;
	py::array_t<int> h_A;
	py::array_t<int> time;
	if(to_remember_timeevol){
		if(verbose >= 2){
			printf("Brute-force core done, Nt = %ld\n", Nt);
			lattice_gas::print_E(&(_E[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
			lattice_gas::print_M(&(_M[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
//		lattice_gas::print_biggest_cluster_sizes(&(_M[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
		}

		E = py::array_t<double>(Nt);
		py::buffer_info E_info = E.request();
		double *E_ptr = static_cast<double *>(E_info.ptr);
		memcpy(E_ptr, _E, sizeof(double) * Nt);
		free(_E);

		M = py::array_t<int>(Nt);
		py::buffer_info M_info = M.request();
		int *M_ptr = static_cast<int *>(M_info.ptr);
		memcpy(M_ptr, _M, sizeof(int) * Nt);
		free(_M);

		biggest_cluster_sizes = py::array_t<int>(Nt);
		py::buffer_info biggest_cluster_sizes_info = biggest_cluster_sizes.request();
		int *biggest_cluster_sizes_ptr = static_cast<int *>(biggest_cluster_sizes_info.ptr);
		memcpy(biggest_cluster_sizes_ptr, _biggest_cluster_sizes, sizeof(int) * Nt);
		free(_biggest_cluster_sizes);

		h_A = py::array_t<int>(Nt);
		py::buffer_info h_A_info = h_A.request();
		int *h_A_ptr = static_cast<int *>(h_A_info.ptr);
		memcpy(h_A_ptr, _h_A, sizeof(int) * Nt);
		free(_h_A);

		time = py::array_t<int>(Nt);
		py::buffer_info time_info = time.request();
		int *time_ptr = static_cast<int *>(time_info.ptr);
		memcpy(time_ptr, _time, sizeof(int) * Nt);
		free(_time);

		if(verbose >= 2){
			printf("internal memory for EM freed\n");
			lattice_gas::print_E(&(E_ptr[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'P');
			lattice_gas::print_M(&(M_ptr[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'P');
			printf("exiting py::run_bruteforce\n");
		}

	}

	free(states);

	return py::make_tuple(E, M, biggest_cluster_sizes, h_A, time, N_launches, time_total);
}

// int run_FFS_C(double *flux0, double *d_flux0, int L, double Temp, double h, int *states, int *N_init_states, long *Nt, int *OP_arr_len, int *OP_interfaces, int N_OP_interfaces, double *probs, double *d_probs, double **E, int **M, int to_remember_timeevol, int verbose)
py::tuple run_FFS(int L, py::array_t<double> e, py::array_t<double> mu, pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces,
				  int to_remember_timeevol, int init_gen_mode, int interface_mode, int default_spin_state,
				  std::optional<int> _verbose)
/**
 *
 * @param L - the side-size of the lattice
 * @param e - matrix NxN, particles interaction energies
 * @param mu - vector Nx1, particles chemical potentials
 * 		H = - \sum_{<ij>} s_i s_j e[s_i][s_j] - \sum_i s_i mu[s_i]
 * 		mu_i = -inf   ==>   n_i = 0
 * 		T = 1
 * @param N_init_states - array of ints [N_OP_interfaces+2], how many states do I want on each interface
 * @param OP_interfaces - array of ints [N_OP_interfaces+2], contains values of M for interfaces. The map is [-L2; M_0](; M_1](...](; M_n-1](; L2]
 * @param to_remember_timeevol - T/F, determines whether the E and M evolution if stored
 * @param init_gen_mode - int, the way how the states in A are generated to be then simulated towards the M_0 interface
 * @param interface_mode - which order parameter is used for interfaces (i.e. causes exits and other influences); 0 - E, 1 - M, 2 - MaxClusterSize
 * @param _verbose - int number >= 0 or py::none(), shows how load is the process; If it's None (py::none()), then the default state 'verbose' is used
 * @return :
 * 	flux0, d_flux0 - the flux from A to M_0
 * 	Nt - array in ints [N_OP_interfaces + 1], the number of MC-steps between each pair of interfaces
 * 		Nt[interface_ID]
 * 	OP_arr_len - memory allocated for storing E and M data
 * 	probs, d_probs - array of ints [N_OP_interfaces - 1], probs[i] - probability to go from i to i+1
 * 		probs[interface_ID]
 * 	states - all the states in a 1D array. Mapping is the following:
 * 		states[0               ][0...L2-1] U states[1               ][0...L2-1] U ... U states[N_init_states[0]                 -1][0...L2-1]   U  ...\
 * 		states[N_init_states[0]][0...L2-1] U states[N_init_states[0]][0...L2-1] U ... U states[N_init_states[1]+N_init_states[0]-1][0...L2-1]   U  ...\
 * 		states[N_init_states[1]+N_init_states[0]][0...L2-1] U   ...\
 * 		... \
 * 		... states[sum_{k=0..N_OP_interfaces}(N_init_states[k])][0...L2-1]
 * 	E, M - Energy and Magnetization data from all the simulations. Mapping - indices changes from last to first:
 * 		M[interface_ID][state_ID][time_ID]
 * 		Nt[interface_ID][state_ID] is used here
 * 		M[0][0][0...Nt[0][0]] U M[0][1][0...Nt[0][1]] U ...
 */
{
	int i, j;
	int L2 = L*L;
	int state_size_in_bytes = L2 * sizeof(int);
	lattice_gas::set_OP_default(L2);

	py::buffer_info e_info = e.request();
	py::buffer_info mu_info = mu.request();
	double *e_ptr = static_cast<double *>(e_info.ptr);
	double *mu_ptr = static_cast<double *>(mu_info.ptr);
	assert(e_info.ndim == 2);
	assert(mu_info.ndim == 1);
	assert(e_info.shape[0] == N_species);
	assert(e_info.shape[1] == N_species);
	assert(mu_info.shape[0] == N_species);

// -------------- check input ----------------
	int verbose = (_verbose.has_value() ? _verbose.value() : lattice_gas::verbose_dafault);

	if(verbose) {
		printf("C module:\nL=%d, to_remember_timeevol=%d, init_gen_mode=%d, interface_mode=%d, def_spin_state=%d, verbose=%d\n",
			   L, to_remember_timeevol, init_gen_mode, interface_mode, default_spin_state, verbose);
		lattice_gas::print_e_matrix(e_ptr);
		lattice_gas::print_mu_vector(mu_ptr);
	}

	assert(L > 0);
	assert((interface_mode >= 0) && (interface_mode < N_interface_modes));
	py::buffer_info OP_interfaces_info = OP_interfaces.request();
	py::buffer_info N_init_states_info = N_init_states.request();
	int *OP_interfaces_ptr = static_cast<int *>(OP_interfaces_info.ptr);
	int *N_init_states_ptr = static_cast<int *>(N_init_states_info.ptr);
	assert(OP_interfaces_info.ndim == 1);
	assert(N_init_states_info.ndim == 1);

	int N_OP_interfaces = OP_interfaces_info.shape[0];
	assert(N_OP_interfaces == N_init_states_info.shape[0]);
	switch (interface_mode) {
		case mode_ID_M:   // M
			for(i = 1; i < N_OP_interfaces; ++i) {
				assert(OP_interfaces_ptr[i] > OP_interfaces_ptr[i-1]);
				assert((OP_interfaces_ptr[i] - OP_interfaces_ptr[1]) % lattice_gas::OP_step[interface_mode] == 0);
			}
			break;
		case mode_ID_CS:   // CS
			for(i = 1; i < N_OP_interfaces; ++i) {
				assert(OP_interfaces_ptr[i] > OP_interfaces_ptr[i-1]);
			}
			break;
	}

// ----------------- create return objects --------------
	double flux0, d_flux0;
	long OP_arr_len = 128;   // the initial value that will be doubling when necessary
// [(-L2)---M_0](---M_1](---...---M_n-2](---M_n-1](---L2]
//        A       1       2 ...n-1       n-1        B
//        0       1       2 ...n-1       n-1       n
	py::array_t<long> Nt = py::array_t<long>(N_OP_interfaces);
	py::array_t<double> probs = py::array_t<double>(N_OP_interfaces - 1);
	py::array_t<double> d_probs = py::array_t<double>(N_OP_interfaces - 1);
	py::buffer_info Nt_info = Nt.request();
	py::buffer_info probs_info = probs.request();
	py::buffer_info d_probs_info = d_probs.request();
	long *Nt_ptr = static_cast<long *>(Nt_info.ptr);
	double *probs_ptr = static_cast<double *>(probs_info.ptr);
	double *d_probs_ptr = static_cast<double *>(d_probs_info.ptr);

	int N_states_total = 0;
	for(i = 0; i < N_OP_interfaces; ++i) {
		N_states_total += N_init_states_ptr[i];
	}
	py::array_t<int> states = py::array_t<int>(N_states_total * L2);
	py::buffer_info states_info = states.request();
	int *states_ptr = static_cast<int *>(states_info.ptr);

    double *_E;
    int *_M;
	int *_biggest_cluster_sizes;
	int *_time;
    if(to_remember_timeevol){
        _E = (double*) malloc(sizeof(double) * OP_arr_len);
        _M = (int*) malloc(sizeof(int) * OP_arr_len);
		_biggest_cluster_sizes = (int*) malloc(sizeof(int) * OP_arr_len);
		_time = (int*) malloc(sizeof(int) * OP_arr_len);
    }

	if(verbose){
		printf("OP interfaces:\n");
		for(i = 0; i < N_OP_interfaces; ++i){
			printf("%d ", OP_interfaces_ptr[i]);
		}
		printf("\n");
	}

	lattice_gas::run_FFS_C(&flux0, &d_flux0, L, e_ptr, mu_ptr, states_ptr, N_init_states_ptr,
						   Nt_ptr, to_remember_timeevol ? &OP_arr_len : nullptr, OP_interfaces_ptr, N_OP_interfaces,
						   probs_ptr, d_probs_ptr,  to_remember_timeevol? &_E : nullptr,
					 to_remember_timeevol ? &_M : nullptr,
					 to_remember_timeevol ? &_biggest_cluster_sizes : nullptr,
					 to_remember_timeevol ? &_time : nullptr,
						   verbose, init_gen_mode, interface_mode,
						   default_spin_state);

	if(verbose >= 2){
		printf("FFS core done\nNt: ");
	}
	long Nt_total = 0;
	for(i = 0; i < N_OP_interfaces; ++i) {
		if(verbose >= 2){
			printf("%ld ", Nt_ptr[i]);
		}
		Nt_total += Nt_ptr[i];
	}

	py::array_t<double> E;
    py::array_t<int> M;
	py::array_t<int> biggest_cluster_sizes;
	py::array_t<int> time;
    if(to_remember_timeevol){
		int N_last_elements_to_print = 10;
		if(verbose >= 2){
			lattice_gas::print_E(&(_E[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'F');
			lattice_gas::print_M(&(_M[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'F');
			printf("copying time evolution, Nt_total = %ld\n", Nt_total);
		}

		E = py::array_t<double>(Nt_total);
		py::buffer_info E_info = E.request();
		double *E_ptr = static_cast<double *>(E_info.ptr);
        memcpy(E_ptr, _E, sizeof(double) * Nt_total);
		free(_E);

		M = py::array_t<int>(Nt_total);
		py::buffer_info M_info = M.request();
		int *M_ptr = static_cast<int *>(M_info.ptr);
        memcpy(M_ptr, _M, sizeof(int) * Nt_total);
		free(_M);

		biggest_cluster_sizes = py::array_t<int>(Nt_total);
		py::buffer_info biggest_cluster_sizes_info = biggest_cluster_sizes.request();
		int *biggest_cluster_sizes_ptr = static_cast<int *>(biggest_cluster_sizes_info.ptr);
		memcpy(biggest_cluster_sizes_ptr, _biggest_cluster_sizes, sizeof(int) * Nt_total);
		free(_biggest_cluster_sizes);

		time = py::array_t<int>(Nt_total);
		py::buffer_info time_info = time.request();
		int *time_ptr = static_cast<int *>(time_info.ptr);
		memcpy(time_ptr, _time, sizeof(int) * Nt_total);
		free(_time);

		if(verbose >= 2){
			printf("data copied\n", Nt_total);
			printf("internal memory for EM freed\n");
			lattice_gas::print_E(E_ptr, Nt_total < N_last_elements_to_print ? Nt_total : N_last_elements_to_print, 'P');
			lattice_gas::print_M(M_ptr, Nt_total < N_last_elements_to_print ? Nt_total : N_last_elements_to_print, 'P');
		}
    }

	if(verbose >= 2){
		printf("exiting py::run_FFS\n");
	}
    return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M, biggest_cluster_sizes, time);
}

namespace lattice_gas
{
    gsl_rng *rng;
    int seed;
    int verbose_dafault;
	int OP_min_default[N_interface_modes];
	int OP_max_default[N_interface_modes];
	int OP_step[N_interface_modes];

	void set_OP_default(int L2)
	{
		OP_min_default[mode_ID_M] = -1;
		OP_max_default[mode_ID_M] = L2+1;
		OP_step[mode_ID_M] = 1;

		OP_min_default[mode_ID_CS] = -1;
		OP_max_default[mode_ID_CS] = L2+1;
		OP_step[mode_ID_CS] = 1;
	}

	int run_FFS_C(double *flux0, double *d_flux0, int L, double *e, double *mu, int *states, int *N_init_states, long *Nt,
				  long *OP_arr_len, int *OP_interfaces, int N_OP_interfaces, double *probs, double *d_probs, double **E, int **M,
				  int **biggest_cluster_sizes, int **time, int verbose, int init_gen_mode, int interface_mode, int default_spin_state)
	/**
	 *
	 * @param flux0 - the flux from l_A
	 * @param d_flux0 - estimate for the flux random error
	 * @param L - see run_FFS
	 * @param e - see run_FFS
	 * @param mu - see run_FFS
	 * @param states - all the states in a 1D array. Mapping is the following:
 * 		states[0               ][0...L2-1] U states[1               ][0...L2-1] U ... U states[N_init_states[0]                 -1][0...L2-1]   U  ...\
 * 		states[N_init_states[0]][0...L2-1] U states[N_init_states[0]][0...L2-1] U ... U states[N_init_states[1]+N_init_states[0]-1][0...L2-1]   U  ...\
 * 		states[N_init_states[1]+N_init_states[0]][0...L2-1] U   ...\
 * 		... \
 * 		... states[sum_{k=0..N_OP_interfaces}(N_init_states[k])][0...L2-1]
	 * @param N_init_states - array of ints [N_OP_interfaces], how many states do I want on each interface
	 * @param Nt - array of ints [N_OP_interfaces - 1],
	 * 	"the number of MC-steps between interface `interface_ID` and `interface_ID+1` pair of interfaces" = Nt[interface_ID]
	 * @param OP_arr_len - the length of the OP-time-evolution array
	 * @param OP_interfaces - array of ints [N_OP_interfaces], contains values of OP for interfaces. The map is [...; M_0](; M_1](...](; M_n-1]
	 * @param N_OP_interfaces - int = len(OP_interfaces)
	 * @param probs - probs[i] - probability to go from i to i+1
	 * @param d_probs - estimations of errors of `probs` assuming the binomial histogram distribution
	 * @param E, M, biggest_cluster_sizes - arrays with time-evolution data
	 * @param verbose - int >= 0, shows how load is the process
	 * @param init_gen_mode - int, the way how the states at l_A interface are generated
	 * @param interfaces_mode - which order parameter is used for interfaces (i.e. causes exits and other influences); 0 - E, 1 - M, 2 - MaxClusterSize
	 * @param default_spin_state - int (-1). Clustering searches for clusters of spins != `default_spin_state`
	 * @return - the Error code (none implemented yet)
	 *
  	 */
	{
		int i, j;
		int L2 = L*L;

// get the initial states; they should be sampled from the distribution in [-L^2; M_0), but they are all set to have M == -L^2 because then they all will fall into the local optimum and almost forget the initial state, so it's almost equivalent to sampling from the proper distribution if 'F(M_0) - F_min >~ T'
		long Nt_total = 0;
		long time_0;
		get_init_states_C(L, e, mu, &time_0, N_init_states[0], states, init_gen_mode,
						  OP_interfaces[0], interface_mode, default_spin_state,
						  OP_interfaces[0], OP_interfaces[N_OP_interfaces - 1],
						  E, M, biggest_cluster_sizes, nullptr, time, &Nt_total, OP_arr_len,
						  verbose);

		Nt[0] = Nt_total;

		*flux0 = (double)N_init_states[0] / time_0;   // [1/step]
		*d_flux0 = *flux0 / sqrt(N_init_states[0]);   // [1/step]; this works only if 'N_init_states[0] >> 1 <==>  (d_f/f << 1)'
		if(verbose){
			printf("Init states (N_states = %d, Nt = %ld) generated with mode %d\n", N_init_states[0], Nt[0], init_gen_mode);
			printf("flux0 = (%e +- %e) 1/step\n", *flux0, *d_flux0);
		}

		int N_states_analyzed = 0;
		int state_size_in_bytes = sizeof(int) * L2;
		long Nt_total_prev;
		for(i = 1; i < N_OP_interfaces; ++i){
			Nt_total_prev = Nt_total;
			// run each FFS step starting from states saved on the previous step (at &(states[L2 * N_states_analyzed]))
			// and saving steps for the next step (to &(states[L2 * (N_states_analyzed + N_init_states[i - 1])])).
			// Keep track of E, M, CS, the timestep. You have `N_init_states[i - 1]` to choose from to start a trial run
			// and you need to generate 'N_init_states[i]' at the next interface. Use OP_A = OP_interfaces[0], OP_next = OP_interfaces[i].
			probs[i - 1] = process_step(&(states[L2 * N_states_analyzed]),
									&(states[L2 * (N_states_analyzed + N_init_states[i - 1])]),
									E, M, biggest_cluster_sizes, time, &Nt_total, OP_arr_len, N_init_states[i - 1], N_init_states[i],
									L, e, mu, OP_interfaces[0], OP_interfaces[i],
									interface_mode, default_spin_state, verbose); // OP_interfaces[0] - (i == 1 ? 1 : 0)
			//d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] / probs[i]));
			d_probs[i - 1] = probs[i - 1] / sqrt(N_init_states[i - 1] * (1 - probs[i - 1]));

			N_states_analyzed += N_init_states[i - 1];

			Nt[i] = Nt_total - Nt_total_prev;
			if(verbose){
				printf("-log10(p_%d) = (%lf +- %lf)\nNt[%d] = %ld; Nt_total = %ld\n", i, -log(probs[i-1]) / log(10), d_probs[i-1] / probs[i-1] / log(10), i, Nt[i], Nt_total);
				// this assumes p<<1
				if(verbose >= 2){
					int N_last_elements_to_print = 10;
					printf("\nstate[%d] beginning: ", i-1);
					for(j = 0; j < (Nt[i] > N_last_elements_to_print ? N_last_elements_to_print : Nt[i]); ++j)  printf("%d ", states[L2 * N_states_analyzed - N_init_states[i-1] + j]);
					printf("\n");
				}
			}
		}

		double ln_k_AB = log(*flux0 * 1);   // flux has units = 1/time; Here, time is in steps, so it's not a problem. But generally speaking it's not clear what time to use here.
		double d_ln_k_AB = lattice_gas::sqr(*d_flux0 / *flux0);
		for(i = 0; i < N_OP_interfaces - 1; ++i){   // we don't need the last prob since it's a P from M=M_last to M=L2
			ln_k_AB += log(probs[i]);
			d_ln_k_AB += lattice_gas::sqr(d_probs[i] / probs[i]);   // this assumes dp/p << 1,
		}
		d_ln_k_AB = sqrt(d_ln_k_AB);

		if(verbose){
			printf("-log10(k_AB * [1 step]) = (%lf +- %lf)\n", - ln_k_AB / log(10), d_ln_k_AB / log(10));
			if(verbose >= 2){
				int N_last_elements_to_print = 10;
				printf("Nt_total = %ld\n", Nt_total);
				if(E) print_E(&((*E)[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'f');
				if(M) print_M(&((*M)[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'f');
				printf("Nt:");
				for(i = 0; i < N_OP_interfaces; ++i) printf(" %ld", Nt[i]);
				printf("\n");
			}
		}

		return 0;
	}

	double process_step(int *init_states, int *next_states, double **E, int **M, int **biggest_cluster_sizes, int **time,
						long *Nt, long *OP_arr_len, int N_init_states, int N_next_states,
						int L, double *e, double *mu, int OP_0, int OP_next,
						int interfaces_mode, int default_spin_state, int verbose)
	/**
	 *
	 * @param init_states - are assumed to contain 'N_init_states * state_size_in_bytes' ints representing states to start simulations from
	 * @param next_states - are assumed to be allocated to have 'N_init_states * state_size_in_bytes' ints
	 * @param E - see run_FFS
	 * @param M - see run_FFS
	 * @param biggest_cluster_sizes - see run_FFS
	 * @param Nt - see run_FFS
	 * @param OP_arr_len - see run_FFS
	 * @param N_init_states - the number of states with M==M_init to start the trajectories from
	 * @param N_next_states - the number of states with M==M_next to generate. The step is completed when this number of states over M_next is obtained
	 * @param L - see run_FFS
	 * @param Temp - see run_FFS
	 * @param h - see run_FFS
	 * @param OP_0 - the lower-border to stop the simulations at. If a simulation reaches this M==OP_0, it's terminated and discarded
	 * @param OP_next - the upper-border to stop the simulations at. If a simulation reaches this M==OP_next, it's stored to be a part of the `init_states set` of states for the next FFS step
	 * @param interfaces_mode - see run_FFS
	 * @param default_spin_state - see run_FFS
	 * @param verbose - see run_FFS
	 * @return - the fraction of successful runs (the runs that reached M==OP_next)
	 */
	{
		int i;
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;

		int N_succ = 0;
		int N_runs = 0;
		int *state_under_process = (int*) malloc(state_size_in_bytes);
		int *cluster_element_inds = (int*) malloc(sizeof(int) * L2);
		int *cluster_sizes = (int*) malloc(sizeof(int) * L2);
		int *is_checked = (int*) malloc(sizeof(int) * L2);
		if(verbose){
			printf("doing step:(%d; %d]\n", OP_0, OP_next);
//			if(verbose >= 2){
//				printf("press any key to continue...\n");
//				getchar();
//			}
		}
		int init_state_to_process_ID;
		int run_status;
		long OP_arr_len_old;
		long Nt_old;
		long time_of_step_total;
		while(N_succ < N_next_states){
			init_state_to_process_ID = gsl_rng_uniform_int(rng, N_init_states);
			if(verbose >= 2){
				printf("state[%d] (id in set = %d):\n", N_succ, init_state_to_process_ID);
			}
			memcpy(state_under_process, &(init_states[init_state_to_process_ID * L2]), state_size_in_bytes);   // get a copy of the chosen init state
			Nt_old = *Nt;
//			OP_arr_len_old = *OP_arr_len;

			run_status = run_state(state_under_process, L, e, mu, &time_of_step_total, OP_0, OP_next,
								   E, M, biggest_cluster_sizes, nullptr, time,
								   cluster_element_inds, cluster_sizes, is_checked,
								   Nt, OP_arr_len, interfaces_mode, default_spin_state, verbose);

			switch (run_status) {
				case 0:  // reached <=OP_A  => unsuccessful run
					++N_runs;
					break;
				case 1:  // reached ==OP_next  => successful run
					// Interpolation is needed if 's' and 'OP' are continuous

					// Nt is not reinitialized to 0 and that's correct because it shows the total number of OPs datapoints
					++N_runs;
					++N_succ;
					if(next_states) {   // save the resulting system state for the next step
						memcpy(&(next_states[(N_succ - 1) * L2]), state_under_process, state_size_in_bytes);
					}
					if(verbose) {
						double progr = (double)N_succ/N_next_states;
						if(verbose < 2){
							if(N_succ % (N_next_states / 1000 + 1) == 0){
								printf("%lf %%          \r", progr * 100);
								fflush(stdout);
							}
						} else { // verbose == 1
							printf("state %d saved for future, N_runs=%d\n", N_succ - 1, N_runs + 1);
							printf("%lf %%\n", progr * 100);
						}
					}
					break;
				case -1:  // reached >OP_next => overshoot => discard the trajectory => revert all the "pointers to the current stage" to their previous values
					*Nt = Nt_old;
//					*OP_arr_len = OP_arr_len_old;
					break;
			}
		}
		if(verbose >= 2) {
			printf("\n");
		}

		free(state_under_process);
		free(cluster_element_inds);
		free(cluster_sizes);
		free(is_checked);

		return (double)N_succ / N_runs;   // the probability P(i+1 | i) to go from i to i+1
	}

	int run_state(int *s, int L, double *e, double *mu, long *time_total,
				  int OP_0, int OP_next, double **E, int **M, int **biggest_cluster_sizes, int **h_A, int **time,
				  int *cluster_element_inds, int *cluster_sizes, int *is_checked, long *Nt, long *OP_arr_len,
				  int interfaces_mode, int default_spin_state, int verbose, long Nt_max, int* states_to_save,
				  int *N_states_saved, int N_states_to_save,  int OP_min_save_state, int OP_max_save_state,
				  int save_state_mode, int OP_A, int OP_B, long N_saved_states_max)
	/**
	 *	Run state saving states in (OP_min_save_state, OP_max_save_state) and saving OPs for the whole (OP_0; OP_next)
	 *
	 * @param s - the current state the system under simulation is in
	 * @param L - see run_FFS
	 * @param Temp - see run_FFS
	 * @param h - see run_FFS
	 * @param OP_0 - see process_step
	 * @param OP_next - see process_step
	 * @param E, M, biggest_cluster_sizes - see run_FFS
	 * @param h_A - the history-dependent function, =1 for the states that came from A, =0 otherwise (i.e. for the states that came from B)
	 * @param cluster_element_inds - array of ints, the indices of spins participating in clusters
	 * @param cluster_sizes - array of ints, sizes of clusters
	 * @param is_checked - array of ints L^2 in size. Labels of spins necessary for clustering. Says which spins are already checked by the clustering procedure during the current clustering run
	 * `cluster_element_inds` are aligned with `cluster_sizes` - there are `cluster_sizes[0]` of indices making the 1st cluster, `cluster_sizes[1]` indices of the 2nd slucter, so on.
	 * @param Nt - see run_FFS
	 * @param OP_arr_len - see process_step
	 * @param verbose - see run_FFS
	 * @param interfaces_mode - see run_FFS
	 * @param default_spin_state - see run_FFS
	 * @param Nt_max - int, default -1; It's it's > 0, then the simulation is stopped on '*Nt >= Nt_max'
	 * @param states_to_save - int*, default nullptr; The pointer to the set of states to save during the run
	 * @param N_states_saved - int*, default nullptr; The number of states already saved in 'states_to_save'
	 * @param N_states_to_save - int, default -1; If it's >0, the simulation is stopped when 'N_states_saved >= N_states_to_save'
	 * @param OP_min_save_state, OP_max_save_state - int, default 0; If(N_states_to_save > 0), states are saved when M == M_thr_save_state
	 * @param save_state_mode :
	 * 1 ("inside region") = save states that have OP in (OP_min_save_state; OP_max_save_state];
	 * 2 ("outflux") = save states that have `OP_current >= OP_min_save_state` and `OP_prev < OP_min_save_state`
	 * @param OP_A, OP_B - A dna B doundaries to compute h_A
	 * @return - the Error status (none implemented yet)
	 */
	{
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;
		int OP_current;
		int OP_prev;
		int M_current = comp_M(s, L); // remember the 1st M;
		double E_current = comp_E(s, L, e, mu); // remember the 1st energy;
		int N_clusters_current = L2;   // so that all uninitialized cluster_sizes are set to 0
		int biggest_cluster_sizes_current = 0;
		bool verbose_BF = (verbose < 0);
		if(verbose_BF) verbose = -verbose;

		if((abs(M_current) > L2) || ((L2 - M_current) % 2 != 0)){   // check for sanity
			state_is_valid(s, L, 0, 'e');
			if(verbose){
				printf("This state has M = %d (L = %d, dM_step = 2) which is beyond possible physical boundaries, there is something wrong with the simulation\n", M_current, L);
				getchar();
			}
		}
		if(N_states_to_save > 0){
			assert(states_to_save);
		}

		switch (interfaces_mode) {
			case mode_ID_M:
				OP_current = M_current;
				break;
			case mode_ID_CS:
				OP_current = biggest_cluster_sizes_current;
				break;
			default:
				assert(false);
		}

		if(verbose >= 2){
			printf("E=%lf, M=%d, OP_0=%d, OP_next=%d\n", E_current, M_current, OP_0, OP_next);
		}

		int ix, iy;
		double dE;
		int time_the_flip_took;
		*time_total = 0;

//		double Emin = -(2 + abs(h)) * L2;
//		double Emax = 2 * L2;
//		int default_spin_state = sgn(h);
		double E_tolerance = 1e-3;   // J=1
		long Nt_for_numerical_error = int(1e13 * E_tolerance / L2);
		// the error accumulates, so we need to recompute form scratch time to time
//      |E| ~ L2 => |dE_numerical| ~ L2 * 1e-13 => |dE_num_total| ~ sqrt(Nt * L2 * 1e-13) << E_tolerance => Nt << 1e13 * E_tolerance / L2

		while(1){
			// ----------- choose which to flip -----------
			time_the_flip_took = get_flip_point(s, e, mu, Temp, &ix, &iy, &dE);

			// --------------- compute time-dependent features ----------
			s[ix*L + iy] *= -1;

			*time_total += time_the_flip_took;

			E_current += dE;

			M_current += 2 * s[ix*L + iy];

			clear_clusters(cluster_element_inds, cluster_sizes, &N_clusters_current);
			uncheck_state(is_checked, L2);
			cluster_state_C(s, L, cluster_element_inds, cluster_sizes, &N_clusters_current, is_checked, default_spin_state);
			biggest_cluster_sizes_current = max(cluster_sizes, N_clusters_current);

			++(*Nt);

			if(verbose_BF){
//				if(*Nt % (Nt_max / 1000 + 1) == 0){
//					fflush(stdout);
//				}
				if(!(*Nt % 1000000)){
					if(Nt_max > 0){
						printf("BF run: %lf %%            \r", (double)(*Nt) / (Nt_max) * 100);
					} else {
						printf("BF run: Nt = %ld               \r", *Nt);
					}
					fflush(stdout);
				}
			}

			// ------------ update the OP --------------
			OP_prev = OP_current;
			switch (interfaces_mode) {
				case mode_ID_M:
					OP_current = M_current;
					break;
				case mode_ID_CS:
					OP_current = biggest_cluster_sizes_current;
					break;
				default:
					assert(false);
			}

			// ------------------ check for Fail ----------------
			// we need to exit before the state is modified, so it's not recorded
			if(OP_current < OP_0){
				if(verbose >= 3) printf("\nFail run, OP_mode = %d, OP_current = %d, OP_0 = %d\n", interfaces_mode, OP_current, OP_0);
				return 0;   // failed = gone to the initial state A
			}

			// -------------- check that error in E is negligible -----------
			// we need to do this since E is double so the error accumulated over steps
			if(*Nt % Nt_for_numerical_error == 0){
				double E_curent_real = comp_E(s, L, h);
				if(abs(E_current - E_curent_real) > E_tolerance){
					if(verbose >= 2){
						printf("\nE-error out of bound: E_current = %lf, dE = %lf, Nt = %ld, E_real = %lf\n", E_current, dE, *Nt, E_curent_real);
						print_E(&((*E)[*Nt - 10]), 10);
						print_S(s, L, 'r');
//						throw -1;
//						getchar();
					}
					E_current = E_curent_real;
				}
			}

			// ------------------ save timeevol ----------------
			if(OP_arr_len){
				if(*Nt >= *OP_arr_len){ // double the size of the time-index
					*OP_arr_len *= 2;
					if(time){
						*time = (int*) realloc (*time, sizeof(int) * *OP_arr_len);
						assert(*time);
					}
					if(E){
						*E = (double*) realloc (*E, sizeof(double) * *OP_arr_len);
						assert(*E);
					}
					if(M){
						*M = (int*) realloc (*M, sizeof(int) * *OP_arr_len);
						assert(*M);
					}
					if(biggest_cluster_sizes){
						*biggest_cluster_sizes = (int*) realloc (*biggest_cluster_sizes, sizeof(int) * *OP_arr_len);
						assert(*biggest_cluster_sizes);
					}
					if(h_A){
						*h_A = (int*) realloc (*h_A, sizeof(int) * *OP_arr_len);
						assert(*h_A);
					}

					if(verbose >= 2){
						printf("\nrealloced to %ld\n", *OP_arr_len);
					}
				}

				if(time) (*time)[*Nt - 1] = time_the_flip_took;
				if(E) (*E)[*Nt - 1] = E_current;
				if(M) (*M)[*Nt - 1] = M_current;
				if(biggest_cluster_sizes) (*biggest_cluster_sizes)[*Nt - 1] = biggest_cluster_sizes_current;
				if(h_A){
					if(*Nt == 1){   // *Nt is assumed to be >= 1
						(*h_A)[0] = (OP_current - OP_A > OP_B - OP_current ? 0 : 1);
					} else {   // *Nt > 1
						(*h_A)[*Nt - 1] = (*h_A)[*Nt - 2] == 1 ? (OP_current < OP_B ? 1 : 0) : (OP_current >= OP_A ? 0 : 1);
					}
				}

//				if((E_current < Emin * (1 + 1e-6)) || (E_current > Emax * (1 + 1e-6))){   // assuming Emin < 0, Emax > 0
//					printf("E_current = %lf, dE = %lf, Nt = %d, E = %lf\n", E_current, dE, *Nt, comp_E(s, L, h));
//					print_E(&((*E)[*Nt - 10]), 10);
//					print_S(s, L, 'r');
//					getchar();
//				}
			}
//			if(verbose >= 4) printf("done Nt=%d\n", *Nt-1);

			// ------------------- save the state if it's good (we don't want failed states) -----------------
//			printf("N_states_to_save = %d, max=%d\n", N_states_to_save, N_saved_states_max);
			if(N_states_to_save > 0){
				bool to_save_state = (*N_states_saved < N_saved_states_max) || (N_saved_states_max < 0);
				switch (save_state_mode) {
					case save_state_mode_Inside:
						to_save_state = to_save_state && (OP_current >= OP_min_save_state) && (OP_current < OP_max_save_state);
						break;
					case save_state_mode_Influx:
						to_save_state = to_save_state && (OP_current >= OP_max_save_state) && (OP_prev < OP_min_save_state);
						break;
					default:
						to_save_state = false;
						if(verbose){
							printf("WARNING:\nN_states_to_save > 0 provided, but wrong save_state_mode. Not saving states\n");
						}
				}
				if(to_save_state){
//					printf("saved state\n");
					memcpy(&(states_to_save[*N_states_saved * L2]), s, state_size_in_bytes);
					++(*N_states_saved);
				}
			}

			// ---------------- check exit ------------------
			// values of M_next are recorded, and states starting from M_next of the next stage are not recorded, so I have ...](... M accounting
			if(N_states_to_save > 0){
				if(*N_states_saved >= N_states_to_save){
					if(verbose >= 2) printf("Reached desired N_states_saved >= N_states_to_save (= %d)\n", N_states_to_save);
					return 1;
				}
			}
			if(Nt_max > 0){
				if(*Nt >= Nt_max){
					if(verbose){
						if(verbose >= 2) {
							printf("Reached desired Nt >= Nt_max (= %ld)\n", Nt_max);
						} else {
							printf("\n");
						}
					}

					return 1;
				}
			}
			if(OP_current >= OP_next){
				if(verbose >= 3) printf("Reached OP_next, OP_current = %d, OP_next = %d\n", OP_current, OP_next);
				return 1;
//				return OP_current == OP_next ? 1 : -1;
				// 1 == succeeded = reached the interface 'M == M_next'
				// -1 == overshoot = discard the trajectory
			}
		}
	}

	int get_OP_from_spinsup(int N_spins_up, int L2, int interface_mode, int default_spin_state)
	{
		switch (interface_mode) {
			case mode_ID_M:
				return (L2 - 2 * N_spins_up) * default_spin_state;
			case mode_ID_CS:
				return N_spins_up;
		}
	}

	int run_bruteforce_C(int L, double Temp, double h, long *time_total, int N_states, int *states,
						 long *OP_arr_len, long *Nt, double **E, int **M, int **biggest_cluster_sizes, int **h_A, int **time,
						 int interface_mode, int default_spin_state, int OP_A, int OP_B,
						 int OP_min_stop_state, int OP_max_stop_state, int *N_states_done,
						 int OP_min_save_state, int OP_max_save_state, int save_state_mode,
						 int N_spins_up_init, int verbose, long Nt_max, int *N_tries, int to_save_final_state,
						 int to_regenerate_init_state, long N_saved_states_max)
	/**
	 *
	 * @param L - see run_FFS
	 * @param Temp - see run_FFS
	 * @param h - see run_FFS
	 * @param time_total - the total physical time "passed" in the system
	 * @param N_states - int
	 * 	if > 0, then the run is going until the `N_states` states are saved. Saving happens based on other threshold parameters passed to the function
	 * 	if == 0, then ignored meaning it won't affect the stopping criteria, so the rub will be until Nt > Nt_max
	 * @param states - array containing all the states saved according to the passed threshold parameters
	 * @param OP_arr_len - see run_FFS
	 * @param Nt - the number of timesteps made
	 * @param E, M, biggest_cluster_sizes, time, h_A - see run_FFS
	 * @param interface_mode - see run_FFS
	 * @param default_spin_state - see run_FFS
	 * @param OP_A - see run_state
	 * @param OP_B - see run_state
	 * @param OP_min_stop_state, OP_max_stop_state - if OP goes outside of (OP_min_stop_state; OP_max_stop_state], then the run is restarted from a state randomly chosen from the already saved states
	 * @param N_states_done - the number of saved states
	 * @param OP_min_save_state - see run_state
	 * @param OP_max_save_state - see run_state
	 * @param save_state_mode - see run_state
	 * @param N_spins_up_init - the number of spins != default_spin_state in the initial configuration. I use =0 for our cases.
	 * 	if < 0, then such an N_spins_up_init is chosen so that the system starts with `OP == OP_min_stop_state` (the minimum value above the stopping occurs)
	 * @param verbose - see run_FFS
	 * @param Nt_max - if > 0, then the run is completed when Nt > Nt_max
	 * @param N_tries - The number of times the BF run got outside of (OP_min_stop_state; OP_max_stop_state] and thus was restarted
	 * @return error code (none implemented)
	 */
	{
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;

		assert(OP_max_stop_state - OP_min_stop_state >= 1);
		assert(OP_min_stop_state <= OP_min_save_state);
		assert(OP_max_save_state <= OP_max_stop_state);
		if(N_spins_up_init < 0){
			switch (interface_mode) {
				case mode_ID_M:
					N_spins_up_init = (L2 - (OP_min_stop_state + 1) * default_spin_state) / 2;
					break;
				case mode_ID_CS:
					N_spins_up_init = OP_min_stop_state + 1;
					break;
			}
		}

//		int OP_init = get_OP_from_spinsup(N_spins_up_init, L2, interface_mode, default_spin_state);
//		assert(OP_min_save_state <= OP_init);
//		assert(OP_init <= OP_max_save_state);

		int *cluster_element_inds = (int*) malloc(sizeof(int) * L2);
		int *cluster_sizes = (int*) malloc(sizeof(int) * L2);
		int *is_checked = (int*) malloc(sizeof(int) * L2);
		int *state_under_process = (int*) malloc(state_size_in_bytes);

		if(to_regenerate_init_state){
			generate_state(states, L, N_spins_up_init, interface_mode, default_spin_state, verbose);
//			print_S(states, L, '0'); 		getchar();
//			*N_states_done += 1;
			*N_states_done = 0;
		}
		int restart_state_ID;

		if(verbose){
			printf("running brute-force:\nL=%d  T=%lf  h=%lf  OP_mode=%d  OP\\in(%d;%d]  N_states_to_gen=%d  Nt_max=%ld  verbose=%d\n", L, Temp, h, interface_mode, OP_min_stop_state, OP_max_stop_state, N_states, Nt_max, verbose);
			switch (save_state_mode) {
				case save_state_mode_Inside:
					printf("N_spins_up_init:%d, OP_min_stop_state:%d, [OP_min_save_state; OP_max_save_state) = [%d; %d), OP_max_stop_state:%d\n", N_spins_up_init, OP_min_stop_state, OP_min_save_state, OP_max_save_state, OP_max_stop_state);
					break;
				case save_state_mode_Influx:
					printf("N_spins_up_init:%d, OP_min_stop_state:%d, OP_A = %d, OP_max_stop_state:%d\n", N_spins_up_init, OP_min_stop_state, OP_min_save_state, OP_max_stop_state);
					break;
			}
		}

		*N_tries = 0;
		*time_total = 0;
		long time_the_try_took;
		while(1){
			// initially start from the 0th state (the only one we have at the time). Next times start from a random state from the ones we have at the time
			if(*N_states_done > 0){
				restart_state_ID = gsl_rng_uniform_int(rng, *N_states_done);
				if(verbose >= 2){
					printf("generated %d states, restarting from state[%d]\n", *N_states_done, restart_state_ID);
				}
			} else {
				restart_state_ID = 0;
				if(verbose >= 2){
					printf("restarting from initial state\n");
				}
			}

			memcpy(state_under_process, &(states[L2 * restart_state_ID]), state_size_in_bytes);   // get a copy of the chosen init state

			run_state(state_under_process, L, Temp, h, &time_the_try_took,
					  OP_min_stop_state, OP_max_stop_state,
					  E, M, biggest_cluster_sizes, h_A, time, cluster_element_inds, cluster_sizes,
					  is_checked, Nt, OP_arr_len, interface_mode, default_spin_state,
					  -verbose, Nt_max, states, N_states_done, N_states,
					  OP_min_save_state, OP_max_save_state, save_state_mode, OP_A, OP_B, N_saved_states_max);

			++ (*N_tries);
			*time_total += time_the_try_took;

			if(N_states > 0){
				if(verbose)	{
					printf("brute-force done %lf              \n", (double)(*N_states_done) / N_states);
					fflush(stdout);
				}
				if(*N_states_done >= N_states) {
					if(verbose) printf("\n");
					-- (*N_tries);  // the last quit was (most likely) due to Nt>Nt_max, not due to OP>OP_B, so we need to substract it.
					break;
				}
			}
			if(Nt_max > 0){
				if(verbose)	{
					printf("brute-force done %lf              \r", (double)(*Nt) / Nt_max);
					fflush(stdout);
				}
				if(*Nt >= Nt_max) {
					-- (*N_tries);
					if(verbose) printf("\n");
					if(to_save_final_state){
						memcpy(&(states[L2 * *N_states_done]), state_under_process, state_size_in_bytes);
						++ (*N_states_done);
					}
					break;
				}
			}
//			if(Nt_max > 0) if(*Nt >= Nt_max) break;
		}

		free(cluster_element_inds);
		free(cluster_sizes);
		free(is_checked);
		free(state_under_process);

		return 0;
	}

	int get_equilibrated_state(int L, double Temp, double h, int *init_state, int interface_mode, int default_spin_state,
							   int OP_A, int OP_B, int verbose)
	{
		long time_total;
		long Nt_to_reach_OP_A = 0;
		int N_states_done = 0;
		int N_tries;
		long Nt = 0;

		// run "all spins down" until it reaches OP_A_thr
		run_bruteforce_C(L, Temp, h, &time_total, 1, init_state,
						 nullptr, &Nt_to_reach_OP_A, nullptr, nullptr, nullptr, nullptr, nullptr,
						 interface_mode, default_spin_state, -1, -1,
						 OP_min_default[interface_mode], OP_A,
						 &N_states_done, OP_A,
						 OP_A, save_state_mode_Influx, -1,
						 verbose, -1, &N_tries, 0, 1, -1);
		if(verbose > 0){
			printf("reached OP >= OP_A = %d in Nt = %ld MC steps\n", OP_A, Nt_to_reach_OP_A);
		}
		// replace the "all down" init state with the "at OP_A_thr" state
//			memcpy(init_states, &(init_states[L2]), sizeof(int) * L2);

		do{
			if(verbose > 0){
				printf("Attempting to simulate Nt = %ld MC steps towards the local optimum\n", Nt_to_reach_OP_A);
				if(N_tries > 0){
					printf("Previous attempt results in %d reaches of state B (OP_B = %d), so restating from the initial ~OP_A\n", N_tries, OP_B);
				}
			}
			// run it for the same amount of time it took to get to OP_A_thr the first time
			run_bruteforce_C(L, Temp, h, &time_total, -1, init_state,
							 nullptr, &Nt, nullptr, nullptr, nullptr, nullptr, nullptr,
							 interface_mode, default_spin_state, -1, -1,
							 OP_min_default[interface_mode], OP_B,
							 &N_states_done, -1,
							 -1, -1, -1,
							 verbose, Nt_to_reach_OP_A, &N_tries, 1, 0, -1);
		}while(N_tries > 0);
		// N_tries = 0 in the beginning of BF. If we went over the N_c, I want to restart because we might not have obtained enough statistic back around the optimum.

		return 0;
	}

	int get_init_states_C(int L, double *e, double *mu, long *time_total, int N_init_states, int *init_states, int mode, int OP_thr_save_state,
						  int interface_mode, int OP_A, int OP_B,
						  double **E, int **M, int **biggest_cluster_size, int **h_A, int **time,
						  long *Nt, long *OP_arr_len, int verbose)
	/**
	 *
	 * @param L - see run_FFS
	 * @param Temp - see run_FFS
	 * @param h - see run_FFS
	 * @param N_init_states - see run_FFS
	 * @param OP_thr_save_state - see 'run_state' description
	 * @param init_states - int*, assumed to be preallocated; The set of states to fill by generating them in this function
	 * @param verbose - see run_FFS
	 * @param mode - the mode of generation
	 * 		mode >=0: generate all spins -1, then randomly (uniform ix and iy \in [0; L^2-1]) set |mode| states to +1
	 * 		mode -1: all spins are 50/50 +1 or -1
	 * 		mode -2: A brute-force simulation is run until 'N_init_states' states with M < M_0 are saved
	 * 			The simluation is run in a following way:
	 * 			1. A state with 'all spins = -1' is generated and saved, then a MC simulation is run until 'M_current' reaches ~ 0
	 * 				(1 for odd L^2, 0 for even L^2. This is because M_step=2, so if L^2 is even, then all possible Ms are also even, and if L^2 is odd, all Ms are odd)
	 * 			2. During the simulation, the state is saved every time when M < M_0
	 * 			3. If we reach 'N_init_states' saved states before hitting 'M_current'~0, then the simulation just stops and the generation is complete
	 * 			4. If we reach 'M_current'~0 before obtaining 'N_init_states' saved states, the simulation is restarted randomly from a state \in states that were already generated and saved
	 * 			5. p. 4 in repeated until we obtain 'N_init_states' saved states
	 * @return - the Error code
	 */
	{
		int i;
		int L2 = L*L;
//		int state_size_in_bytes = sizeof(int) * L2;

		if(verbose){
			printf("generating states:\nN_init_states=%d, gen_mode=%d, OP_A <= %d, OP_mode=%d\n", N_init_states, mode, OP_thr_save_state, interface_mode);
		}

		if(mode >= -1){
			// generate N_init_states states in A
			// Here they are identical, but I think it's better to generate them accordingly to equilibrium distribution in A
			for(i = 0; i < N_init_states; ++i){
				generate_state(&(init_states[i * L2]), L, mode, interface_mode, verbose);
			}
		} else if(mode == -2){
			int N_states_done;
			int N_tries;
//			long Nt = 0;
			run_bruteforce_C(L, e, mu, time_total, N_init_states, init_states,
							 nullptr, Nt, nullptr, nullptr, nullptr, nullptr, nullptr,
							 interface_mode, 0, 0,
							 OP_min_default[interface_mode], OP_B,
							 &N_states_done, OP_min_default[interface_mode],
							 OP_thr_save_state, save_state_mode_Inside, -1,
							 verbose, -1, &N_tries, 0, 1, -1);
		} else if(mode == -3){
			int N_states_done;
			int N_tries = 0;

			get_equilibrated_state(L, e, mu, init_states, interface_mode, OP_thr_save_state, OP_B, verbose);

			*Nt = 0;   // forget anything we might have had
			*time_total = 0;
			N_states_done = 0;
			run_bruteforce_C(L, e, mu, time_total, N_init_states, init_states,
							 OP_arr_len, Nt, E, M, biggest_cluster_size, h_A, time,
							 interface_mode, OP_A, OP_B,
							 OP_min_default[interface_mode], OP_B,
							 &N_states_done, OP_thr_save_state,
							 OP_thr_save_state, save_state_mode_Influx, -1,
							 verbose, -1, &N_tries, 0, 0, -1);
			if(verbose > 0){
				if(N_tries > 0){
					printf("=================== WARNING ===============\nNumber of MC steps made for initial flux estimation: %ld\nNumber of times going to the B state (OP_B = %d): %d\n", *Nt, OP_B, N_tries);
				}
			}
		}

		return 0;
	}

	void cluster_state_C(const int *s, int L, int *cluster_element_inds, int *cluster_sizes, int *cluster_types, int *N_clusters, int *is_checked)
	{
		int i;
		int L2 = L * L;

		i = 0;
		*N_clusters = 0;
		int N_clustered_elements = 0;
		while(i < L2){
			if(!is_checked[i]){
				if(s[i] == background_specie_id) {
					is_checked[i] = -1;
				} else {
					cluster_types[*N_clusters] = s[i];
					add_to_cluster(s, L, is_checked, &(cluster_element_inds[N_clustered_elements]),
								   &(cluster_sizes[*N_clusters]), i, (*N_clusters) + 1,
								   s[i]);
					N_clustered_elements += cluster_sizes[*N_clusters];
					++(*N_clusters);
				}
			}
			++i;
		}

	}

	int add_to_cluster(const int* s, int L, int* is_checked, int* cluster, int* cluster_size, int pos, int cluster_label, int cluster_specie)
	{
		if(!is_checked[pos]){
			if(s[pos] == background_specie_id) {
				is_checked[pos] = -1;
			} else if(s[pos] == cluster_specie) {
				int L2 = L * L;
				is_checked[pos] = cluster_label;
				cluster[*cluster_size] = pos;
				++(*cluster_size);

				add_to_cluster(s, L, is_checked, cluster, cluster_size, md(pos - L, L2), cluster_label, cluster_specie);
				add_to_cluster(s, L, is_checked, cluster, cluster_size, pos % L == 0 ? pos + L - 1 : pos - 1, cluster_label, cluster_specie);
				add_to_cluster(s, L, is_checked, cluster, cluster_size, (pos + 1) % L == 0 ? pos - L + 1 : pos + 1, cluster_label, cluster_specie);
				add_to_cluster(s, L, is_checked, cluster, cluster_size, md(pos + L, L2), cluster_label, cluster_specie);
			}    // do nothing if s[pos] is other non-empty specie than cluster_specie
		}

		return 0;
	}

	int is_infinite_cluster(const int* cluster, const int* cluster_size, int L, char *present_rows, char *present_columns)
	{
		int i = 0;

		zero_array(present_rows, L);
		zero_array(present_columns, L);
		for(i = 0; i < (*cluster_size); ++i){
			present_rows[cluster[i] / L] = 1;
			present_columns[cluster[i] % L] = 1;
		}
		char cluster_is_infinite_x = 1;
		char cluster_is_infinite_y = 1;
		for(i = 0; i < L; ++i){
			if(!present_columns[i]) cluster_is_infinite_x = 0;
			if(!present_rows[i]) cluster_is_infinite_y = 0;
		}

		return cluster_is_infinite_x || cluster_is_infinite_y;
	}

	void uncheck_state(int *is_checked, int N)
	{
		for(int i = 0; i < N; ++i) is_checked[i] = 0;
	}

	void clear_clusters(int *clusters, int *cluster_sizes, int *N_clusters)
	{
//		int N_done = 0;
		for(int i = 0; i < *N_clusters; ++i){
//			for(int j = 0; j < cluster_sizes[i]; ++j){
//				clusters[j + N_done] = -1;
//			}
//			N_done += cluster_sizes[i];
			cluster_sizes[i] = 0;
		}
		*N_clusters = 0;
	}

	void clear_cluster(int* cluster, int *cluster_size)
	{
//		for(int i = 0; i < *cluster_size; ++i) cluster[i] = -1;
		*cluster_size = 0;
	}

	int comp_M(const int *state, int L2)
	/**
	 * Computes the amount of specie[1]
	 * @param state - the state to analyze
	 * @param L - the linear size of the lattice
	 * @return - the total amoint of s[1]
	 */
	{
		int i;
		int _M = 0;
		for(i = 0; i < L2; ++i) if(state[i] == main_specie_id) ++_M;
		return _M;
	}

	double comp_E(const int* state, int L, double *e, double *mu)
	/**
	 * Computes the Energy of the state 's' of the linear size 'L', immersed in the 'h' magnetic field;
	 * H = E/T = -\sum_{<ij>} s_i s_j e[s_i][s_j] - \sum_i mu[s_i]
	 * @param state - the state to analyze
	 * @param L - the linear size of the lattice
	 * @param e - interaction matrix
	 * @param mu - chemical potentials
	 * @return - the value of Energy of the state 'state'
	 */
	{
		int i, j;
		int L2 = L*L;

		double _E = 0;
		int s0;
		for(i = 0; i < L-1; ++i){
			for(j = 0; j < L-1; ++j){
				s0 = state[i*L + j] * N_species;
				_E += e[s0 + state[(i+1)*L + j]] + e[s0 + state[i*L + (j+1)]];
			}
			s0 = state[i*L + (L-1)] * N_species;
			_E += e[s0 + state[(i+1)*L + (L-1)]] + e[s0 + state[i*L + 0]];
		}
		for(j = 0; j < L-1; ++j){
			s0 = state[(L-1)*L + j] * N_species;
			_E += e[s0 + state[0*L + j]] + e[s0 + state[(L-1)*L + (j+1)]];
		}
		s0 = state[(L-1)*L + (L-1)] * N_species;
		_E += e[s0 + state[0*L + (L-1)]] + e[s0 + state[(L-1)*L + 0]];

		double _M = 0;
		for(i = 0; i < L2; ++i){
			_M += mu[state[i]];
		}

		return - _M - _E;    // e, mu > 0 -> we need to *(-1) because we search for a minimum
	}

	int generate_state(int *s, int L, int mode, int interface_mode, int default_spin_state, int verbose)
	/**
	 * Generates (fill with values) a single state 's' of size 'L' in a 'mode' way
	 * @param s - the state to fill
	 * @param L - the linear size of the state
	 * @param mode - the way to fill the state
	 * @return - the Error code
	 */
	{
		int i, j;
		int L2 = L*L;

		if(mode >= 0){   // generate |mode| spins UP, and the rest spins DOWN
			for(i = 0; i < L2; ++i) s[i] = default_spin_state;
			if(mode > 0){
				int N_down_spins = mode;
				assert(N_down_spins <= L2);

				switch (interface_mode) {
					case mode_ID_M:
					{
						int *indices_to_flip = (int*) malloc(sizeof(int) * L2);
						for(i = 0; i < L2; ++i) indices_to_flip[i] = i;
						int max = L2;
						int next_rand;
						int swap_var;
						for(i = 0; i < N_down_spins; ++i){   // generate N_down_spins uniformly distributed indices in [0; L2-1]
							next_rand = gsl_rng_uniform_int(rng, max);
							swap_var = indices_to_flip[next_rand];
							indices_to_flip[next_rand] = indices_to_flip[max - 1];
							indices_to_flip[max - 1] = swap_var;
							--max;

							s[swap_var] = -default_spin_state;
						}
						free(indices_to_flip);
					}
						break;
					case mode_ID_CS:
						for(i = 0; i < N_down_spins; ++i){
							s[i] = -default_spin_state;
						}
						break;
				}
			}
			int OP_init = get_OP_from_spinsup(mode, L2, interface_mode, default_spin_state);
			if(verbose){
				printf("generated state: N_spins_up=%d, OP_mode=%d, OP=%d\n", mode, interface_mode, OP_init);
			}
		} else if(mode == -1){   // random
			for(i = 0; i < L2; ++i) s[i] = gsl_rng_uniform_int(rng, 2) * 2 - 1;
		}

		return 0;
	}

	double get_dE(int *state, int L, double *e, double *mu, int ix, int iy, int *s_new)
	/**
	 * Computes the energy difference '(E_future_after_the_flip - E_current)'.
	 * H = -\sum_{<ij>} s_i s_j e[s_i][s_j] - \sum_i mu[s_i]
	 * @param state - the current state (before the flip)
	 * @param ix - the X index os the spin considered for a flip
	 * @param iy - the Y index os the spin considered for a flip
	 * @param s_new - the cyclic step from s_current to s_new in the space of species' ids
	 * @return - the values of a potential Energy change
	 */
// H =
	{
		int s = state[ix*L + iy];
		*s_new = (s + 1 + *s_new) % N_species;
		int s1 = state[md(ix + 1, L)*L + iy];
		int s2 = state[ix*L + md(iy + 1, L)];
		int s3 = state[md(ix - 1, L)*L + iy];
		int s4 = state[ix*L + md(iy - 1, L)];
		return (mu[s] + e[s * N_species + s1] + e[s * N_species + s2] + e[s * N_species + s3] + e[s * N_species + s4])
			  -(mu[*s_new] + e[*s_new * N_species + s1] + e[*s_new * N_species + s2] + e[*s_new * N_species + s3] + e[*s_new * N_species + s4]);
	}

	int get_flip_point(int *state, int L, double *e, double *mu, int *ix, int *iy, int *s_new, double *dE)
	/**
	 * Get the positions [*ix, *iy] of a spin to flip in a MC process
	 * @param ix - int*, the X index of the spin to be flipped (already decided)
	 * @param iy - int*, the Y index of the spin to be flipped (already decided)
	 * @param dE - the Energy difference necessary for the flip (E_flipped - E_current)
	 * @return - the number of tries it took to obtain a successful flip
	 */
	{
		int N_flip_tries = 0;
		do{
			*ix = gsl_rng_uniform_int(rng, L);
			*iy = gsl_rng_uniform_int(rng, L);
			*s_new = gsl_rng_uniform_int(rng, N_species - 1);

			*dE = get_dE(state, L, e, mu, *ix, *iy, *s_new);
			++N_flip_tries;
		}while(!(*dE <= 0 ? 1 : (gsl_rng_uniform(rng) < exp(- *dE) ? 1 : 0)));

		return N_flip_tries;
	}

	int init_rand_C(int my_seed)
	/**
	 * Sets up a new GSL randomg denerator seed
	 * @param my_seed - the new seed for GSL
	 * @return  - the Error code
	 */
	{
		// initialize random generator
		gsl_rng_env_setup();
		const gsl_rng_type* T = gsl_rng_default;
		rng = gsl_rng_alloc(T);
		gsl_rng_set(rng, my_seed);
//		srand(my_seed);

		seed = my_seed;
		return 0;
	}

	void print_E(const double *E, long Nt, char prefix, char suffix)
    {
        if(prefix > 0) printf("Es: %c\n", prefix);
        for(int i = 0; i < Nt; ++i) printf("%lf ", E[i]);
        if(suffix > 0) printf("%c", suffix);
    }

	void print_M(const int *M, long Nt, char prefix, char suffix)
	{
		if(prefix > 0) printf("Ms: %c\n", prefix);
		for(int i = 0; i < Nt; ++i) printf("%d ", M[i]);
		if(suffix > 0) printf("%c", suffix);
	}

    int print_S(const int *s, int L, char prefix)
    {
        int i, j;

        if(prefix > 0){
            printf("%c\n", prefix);
        }

        for(i = 0; i < L; ++i){
            for(j = 0; j < L; ++j){
                printf("%2d", s[i*L + j]);
            }
            printf("\n");
        }

        return 0;
    }

	int state_is_valid(const int *s, int L, int k, char prefix)
	{
		for(int i = 0; i < L*L; ++i) if(abs(s[i]) != 1) {
				printf("%d\n", k);
				print_S(s, L, prefix);
				return 0;
			}
		return 1;
	}

	int E_is_valid(const double *E, const double E1, const double E2, int N, int k, char prefix)
	{
		for(int i = 0; i < N; ++i) if((E[i] < E1) || (E[i] > E2)) {
				printf("%c%d\n", prefix, k);
				printf("pos: %d\n", i);
				print_E(&(E[std::max(i - 5, 0)]), std::min(i, 5), 'b');
				printf("E_err = %lf\n", E[i]);
//				print_S();
				print_E(&(E[std::min(i + 5, N - 1)]), std::min(N - i, 5), 'a');
				return 0;
			}
		return 1;
	}

	void print_e_matrix(double *e)
	{
		int ix, iy;
		for(ix = 0; ix < N_species; ++ix) {
			for(iy = 0; iy < N_species; ++iy){
				printf("%5.lf ", e[ix * N_species + iy]);
			}
			printf("\n");
		}
	}

	void print_mu_vector(double *mu)
	{
		for(int i = 0; i < N_species; ++i)
			printf("%5.lf ", mu[i]);
		printf("\n");
	}

}

