//
// Created by ypolyach on 10/27/21.
//

#include <gsl/gsl_rng.h>
#include <cmath>
#include <optional>

#include <pybind11/pytypes.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <Python.h>

#include <vector>
#include <random>
#include <set>
#include <algorithm>

namespace py = pybind11;
using namespace pybind11::literals;

#include "lattice_gas.h"

//void test_my(int k, py::array_t<long> *_Nt, py::array_t<double> *_probs, py::array_t<double> *_d_probs, int l)
//// A function for DEBUG purposes
//{
//	printf("checking  step %d\n", k);
//	py::buffer_info Nt_info = (*_Nt).request();
//	py::buffer_info probs_info = (*_probs).request();
//	py::buffer_info d_probs_info = (*_d_probs).request();
//	printf("n%d: %ld\n", k, Nt_info.shape[0]);
//	printf("p%d: %ld\n", k, probs_info.shape[0]);
//	printf("d%d: %ld\n", k, d_probs_info.shape[0]);
//}

void print_possible_move_modes()
{
	printf("flip: %d\nswap : %d\n", move_mode_flip, move_mode_swap);
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

py::int_ get_verbose()
{
	return lattice_gas::verbose_dafault;
}

py::dict get_move_modes()
{
	return py::dict("flip"_a=move_mode_flip, "swap"_a=move_mode_swap, "long_swap"_a=move_mode_long_swap);
}

//int compute_hA(py::array_t<int> *h_A, int *OP, long Nt, int OP_A, int OP_B)
//{
//	py::buffer_info h_A_info = (*h_A).request();
//	int *h_A_ptr = static_cast<int *>(h_A_info.ptr);
//	h_A_ptr[0] = OP[0] - OP_A <= OP_B - OP[0] ? 1 : 0;   // if closer to A at t=0, then 1, else 0
//	int i;
//	for(i = 1; i < Nt; ++i){
//		h_A_ptr[i] = h_A_ptr[i-1] == 1 ? (OP[i] <= OP_B ? 1 : 0) : (OP[i] > OP_A ? 0 : 1);
//	}
//
//	return 0;
//}

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

py::tuple cluster_state(py::array_t<int> state, std::optional<int> _verbose)
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

	py::array_t<int> cluster_types = py::array_t<int>(L2);
	py::buffer_info cluster_types_info = cluster_types.request();
	int *cluster_types_ptr = static_cast<int *>(cluster_types_info.ptr);

	py::array_t<int> is_checked = py::array_t<int>(L2);
	py::buffer_info is_checked_info = is_checked.request();
	int *is_checked_ptr = static_cast<int *>(is_checked_info.ptr);

	int N_clusters = L2;

	lattice_gas::clear_clusters(cluster_element_inds_ptr, cluster_sizes_ptr, &N_clusters);
	lattice_gas::uncheck_state(is_checked_ptr, L2);
	lattice_gas::cluster_state_C(state_ptr, L, cluster_element_inds_ptr, cluster_sizes_ptr, cluster_types_ptr, &N_clusters, is_checked_ptr);

	return py::make_tuple(cluster_element_inds, cluster_sizes, cluster_types);
}

py::tuple run_bruteforce(int move_mode, int L, py::array_t<double> e, py::array_t<double> mu, long Nt_max,
						 long N_saved_states_max, long save_states_stride, long stab_step,
						 std::optional<int> _N_spins_up_init, std::optional<int> _to_remember_timeevol,
						 std::optional<int> _OP_A, std::optional<int> _OP_B,
						 std::optional<int> _OP_min_save_state, std::optional<int> _OP_max_save_state,
						 std::optional<int> _OP_min, std::optional<int> _OP_max,
						 std::optional<int> _interface_mode, int save_state_mode,
						 std::optional< pybind11::array_t<int> > _init_state,
						 int to_use_smart_swap, int to_equilibrate, int to_start_only_state0,
						 std::optional<int> _verbose)
/**
 *
 * @param move_mode - type of MC moves
 * 		1 ('flip') - \mu-V-T ensemble, site flips
 * 		2 ('swap') - localN-V-T ensemble, local swaps
 * 		3 ('long_swap') - N-V-T ensemble, any range swaps
 * @param L - the side-size of the lattice
 * @param e - matrix NxN, particles interaction energies
 * @param mu - vector Nx1, particles chemical potentials
 * 		H = \sum_{<ij>} e[s_i][s_j] + \sum_i s_i mu[s_i]
 * 		mu_i = +inf   ==>   n_i = 0
 * 		T = 1
 * @param Nt_max - for many successful MC steps before halt
 * @param N_saved_states_max - after what number of saved states to halt
 * @param save_states_stride - time stride to save states time-evol
 * @param stab_step - the initial attempted time to equilibrate the system is "time_to_get_to_A_boundary" + "stab_step".
 * 		equilibration time will be reduced if the system goes to B in the initial equib_time.
 *
 * @param _N_spins_up_init - the number of up-spins in the initial frame. I always used 0 for my simulations
 * @param _to_remember_timeevol - whether to record the time dependence of OPs
 * @param _OP_A, _OP_B - the boundaries of A and B to compute h_A
 * @param _OP_min_save_state, _OP_max_save_state - save states within this region of OP
 * @param _OP_min, _OP_max - the boundaries at which the simulation is restarted. Here I always use [most_left_possible - 1; most_gight + 1] so the system never restarts.
 * @param _interface_mode - Magnetization or CS
 * @param _init_state - the state to start the run from
 * @param to_use_smart_swap - see run_state
 * @param to_equilibrate - whether to equilibrate the (given) initial state
 * @param _verbose - int number >= 0 or py::none(), shows how load is the process; If it's None (py::none()), then the default state 'verbose' is used
 *
 * @return :
 * 	(E, M, CS, h_A, time, k_AB)
 * 	E, M, CS, h_A, time - arrays [Nt]
 * 	k_AB = N_restarts / Nt_max
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
	assert(e_info.ndim == 1);
	assert(mu_info.ndim == 1);
	assert(e_info.shape[0] == N_species * N_species);
	assert(mu_info.shape[0] == N_species);

// -------------- check input ----------------
	assert(L > 0);
	int verbose = (_verbose.has_value() ? _verbose.value() : lattice_gas::verbose_dafault);
	int to_remember_timeevol = (_to_remember_timeevol.has_value() ? _to_remember_timeevol.value() : 1);
	int interface_mode = (_interface_mode.has_value() ? _interface_mode.value() : mode_ID_CS);
	assert((interface_mode >= 0) && (interface_mode < N_interface_modes));
	int OP_min = (_OP_min.has_value() ? _OP_min.value() : lattice_gas::OP_min_default[interface_mode]);
	int OP_max = (_OP_max.has_value() ? _OP_max.value() : lattice_gas::OP_max_default[interface_mode]);
	assert(OP_max > OP_min);
	int OP_A = (_OP_A.has_value() ? _OP_A.value() : OP_min);
	int OP_B = (_OP_B.has_value() ? _OP_B.value() : OP_max);
	int OP_min_save_state = (_OP_min_save_state.has_value() ? _OP_min_save_state.value() : OP_min);
	int OP_max_save_state = (_OP_max_save_state.has_value() ? _OP_max_save_state.value() : OP_max);
	int N_spins_up_init = (_N_spins_up_init.has_value() ? _N_spins_up_init.value() : -1);
	if(stab_step < 0) stab_step *= (-L2);

	py::buffer_info _init_state_info;
	int *_init_state_ptr = nullptr;
	if(_init_state.has_value()){
		_init_state_info = _init_state.value().request();
		_init_state_ptr = static_cast<int *>(_init_state_info.ptr);
		assert(_init_state.value().ndim() == 1);
		assert(_init_state.value().shape()[0] == L2);
	}

// ----------------- create return objects --------------
	long Nt = 0;
	long Nt_OP_saved = 0;
	int N_states_saved = 0;
	int N_launches;
	long OP_arr_len = 128;   // the initial value that will be doubling when necessary
//	py::array_t<int> state = py::array_t<int>(L2);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets
//	py::buffer_info state_info = state.request();
//	int *state_ptr = static_cast<int *>(state_info.ptr);

//	int *states = (int*) malloc(sizeof (int) * L2 * std::max((long)1, N_saved_states_max));
	if(N_saved_states_max == 0){
		N_saved_states_max = Nt_max / save_states_stride;
	}

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

	py::array_t<int> states = py::array_t<int>((N_saved_states_max > 0 ? N_saved_states_max : 3) * L2);
	py::buffer_info states_info = states.request();
	int *states_ptr = static_cast<int *>(states_info.ptr);

	N_states_saved = 0;
	lattice_gas::get_equilibrated_state(move_mode, L, e_ptr, mu_ptr, states_ptr, &N_states_saved,
										interface_mode, OP_A, OP_B, stab_step, _init_state_ptr, to_use_smart_swap,
										to_equilibrate, verbose);
	++N_states_saved;
	// N_states_saved is set to its initial values by default, so the equilibrated state is not saved
	// ++N prevents further processes from overwriting the initial state so it will be returned as states[0]


	lattice_gas::run_bruteforce_C(move_mode, L, e_ptr, mu_ptr, &time_total, N_saved_states_max, states_ptr,
							to_remember_timeevol ? &OP_arr_len : nullptr,
								  &Nt, &Nt_OP_saved, &_E, &_M, &_biggest_cluster_sizes, &_h_A, &_time,
								  interface_mode, OP_A, OP_B, OP_A > 1, to_start_only_state0,
								  OP_min, OP_max, &N_states_saved,
								  OP_min_save_state, OP_max_save_state,save_state_mode,
								  N_spins_up_init, verbose, Nt_max, &N_launches, 0,
								  (OP_A <= 1) && (!bool(_init_state_ptr)), save_states_stride,
								  to_use_smart_swap);

	int N_last_elements_to_print = std::min(Nt_OP_saved, (long)10);

	py::array_t<double> E;
	py::array_t<int> M;
	py::array_t<int> biggest_cluster_sizes;
	py::array_t<int> h_A;
	py::array_t<int> time;
	if(to_remember_timeevol){
		if(verbose >= 2){
			printf("Brute-force core done,  Nt = %ld, Nt_OP = %ld\n", Nt, Nt_OP_saved);
			lattice_gas::print_E(&(_E[Nt_OP_saved - N_last_elements_to_print]), N_last_elements_to_print, 'F');
			lattice_gas::print_M(&(_M[Nt_OP_saved - N_last_elements_to_print]), N_last_elements_to_print, 'F');
//		lattice_gas::print_biggest_cluster_sizes(&(_M[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
		}

		E = py::array_t<double>(Nt_OP_saved);
		py::buffer_info E_info = E.request();
		double *E_ptr = static_cast<double *>(E_info.ptr);
		memcpy(E_ptr, _E, sizeof(double) * Nt_OP_saved);
		free(_E);

		M = py::array_t<int>(Nt_OP_saved);
		py::buffer_info M_info = M.request();
		int *M_ptr = static_cast<int *>(M_info.ptr);
		memcpy(M_ptr, _M, sizeof(int) * Nt_OP_saved);
		free(_M);

		biggest_cluster_sizes = py::array_t<int>(Nt_OP_saved);
		py::buffer_info biggest_cluster_sizes_info = biggest_cluster_sizes.request();
		int *biggest_cluster_sizes_ptr = static_cast<int *>(biggest_cluster_sizes_info.ptr);
		memcpy(biggest_cluster_sizes_ptr, _biggest_cluster_sizes, sizeof(int) * Nt_OP_saved);
		free(_biggest_cluster_sizes);

		h_A = py::array_t<int>(Nt_OP_saved);
		py::buffer_info h_A_info = h_A.request();
		int *h_A_ptr = static_cast<int *>(h_A_info.ptr);
		memcpy(h_A_ptr, _h_A, sizeof(int) * Nt_OP_saved);
		free(_h_A);

		time = py::array_t<int>(Nt_OP_saved);
		py::buffer_info time_info = time.request();
		int *time_ptr = static_cast<int *>(time_info.ptr);
		memcpy(time_ptr, _time, sizeof(int) * Nt_OP_saved);
		free(_time);

		if(verbose >= 2){
			printf("internal memory for EMt freed\n");
			lattice_gas::print_E(&(E_ptr[Nt_OP_saved - N_last_elements_to_print]), N_last_elements_to_print, 'P');
			lattice_gas::print_M(&(M_ptr[Nt_OP_saved - N_last_elements_to_print]), N_last_elements_to_print, 'P');
			printf("exiting py::run_bruteforce\n");
		}

	}

//	free(states);

	return py::make_tuple(states, E, M, biggest_cluster_sizes, h_A, time, N_launches, time_total);
}

//int run_FFS_C(int move_mode, double *flux0, double *d_flux0, int L, const double *e, const double *mu, int *states,
//			  int *N_init_states, long *Nt, long *Nt_OP_saved, long stab_step,
//			  long *OP_arr_len, int *OP_interfaces, int N_OP_interfaces, double *probs, double *d_probs, double **E, int **M,
//			  int **biggest_cluster_sizes, int **time, int verbose, int init_gen_mode, int interface_mode,
//			  const int *init_state, int to_use_smart_swap);
py::tuple run_FFS(int move_mode, int L, py::array_t<double> e, py::array_t<double> mu,
				  pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces,
				  int to_remember_timeevol, int init_gen_mode, int interface_mode,  long stab_step,
				  std::optional< pybind11::array_t<int> > _init_state, int to_use_smart_swap,
				  std::optional<int> _verbose)
/**
 *
 * @param L - the side-size of the lattice
 * @param e - matrix NxN, particles interaction energies
 * @param mu - vector Nx1, particles chemical potentials
 * 		H = \sum_{<ij>} s_i s_j e[s_i][s_j] + \sum_i s_i mu[s_i]
 * 		mu_i = +inf   ==>   n_i = 0
 * 		T = 1
 * @param N_init_states - array of ints [N_OP_interfaces+2], how many states do I want on each interface
 * @param OP_interfaces - array of ints [N_OP_interfaces+2], contains values of M for interfaces. The map is [-L2; M_0](; M_1](...](; M_n-1](; L2]
 * @param to_remember_timeevol - T/F, determines whether the E and M evolution if stored
 * @param init_gen_mode - int, the way how the states in A are generated to be then simulated towards the M_0 interface
 * @param interface_mode - which order parameter is used for interfaces (i.e. causes exits and other influences); 0 - E, 1 - M, 2 - MaxClusterSize
 * @param _verbose - int number >= 0 or py::none(), shows how load is the process; If it's None (py::none()), then the default state 'verbose' is used
 * @return :
 * 	states - all the states in a 1D array. Mapping is the following:
 * 		states[0               ][0...L2-1] U states[1               ][0...L2-1] U ... U states[N_init_states[0]                 -1][0...L2-1]   U  ...\
 * 		states[N_init_states[0]][0...L2-1] U states[N_init_states[0]][0...L2-1] U ... U states[N_init_states[1]+N_init_states[0]-1][0...L2-1]   U  ...\
 * 		states[N_init_states[1]+N_init_states[0]][0...L2-1] U   ...\
 * 		... \
 * 		... states[sum_{k=0..N_OP_interfaces}(N_init_states[k])][0...L2-1]
 * 	states_parent_inds - indices of parents of states. Allows to back-trace any state to any-level parent state
 * 		structure:
 * 		states_parent_inds[N_init_states[1] ... (N_init_states[1]+N_init_states[2])] - parents of states at the 2nd interface
 * 		...
 * 	probs, d_probs - array of ints [N_OP_interfaces - 1], probs[i] - probability to go from i to i+1
 * 		probs[interface_ID]
 * 	Nt - array of ints [N_OP_interfaces], the number of MC-steps between each pair of interfaces
 * 		Nt[interface_ID]
 * 	Nt_OP_saved - array of ints [N_OP_interfaces], time-index used for time-evol records
 * 	flux0, d_flux0 - the flux from A to B
 * 	E, M, biggest_cluster_size, time - arrays of time-evol
 * 		E, M, CS, time: data from all of the simulations.
 * 		time - number of MC change-attempts the system spent in the corresponding state
 * 		Mapping - indices changes from last to first:
 * 			M[interface_ID][state_ID][time_ID]
 * 			Nt[interface_ID][state_ID] is used here
 * 			M[0][0][0...Nt[0][0]] U M[0][1][0...Nt[0][1]] U ...
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
	assert(e_info.ndim == 1);
	assert(mu_info.ndim == 1);
	assert(e_info.shape[0] == N_species * N_species);
	assert(mu_info.shape[0] == N_species);

	if(stab_step < 0) stab_step *= (-L2);

// -------------- check input ----------------
	int verbose = (_verbose.has_value() ? _verbose.value() : lattice_gas::verbose_dafault);

	if(verbose) {
		printf("C module:\nL=%d, to_remember_timeevol=%d, init_gen_mode=%d, interface_mode=%d, verbose=%d\n",
			   L, to_remember_timeevol, init_gen_mode, interface_mode, verbose);

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
	py::array_t<long> Nt_OP_saved = py::array_t<long>(N_OP_interfaces);
	py::array_t<double> probs = py::array_t<double>(N_OP_interfaces - 1);
	py::array_t<double> d_probs = py::array_t<double>(N_OP_interfaces - 1);
	py::buffer_info Nt_info = Nt.request();
	py::buffer_info Nt_OP_saved_info = Nt_OP_saved.request();
	py::buffer_info probs_info = probs.request();
	py::buffer_info d_probs_info = d_probs.request();
	long *Nt_ptr = static_cast<long *>(Nt_info.ptr);
	long *Nt_OP_saved_ptr = static_cast<long *>(Nt_OP_saved_info.ptr);
	double *probs_ptr = static_cast<double *>(probs_info.ptr);
	double *d_probs_ptr = static_cast<double *>(d_probs_info.ptr);

	int N_states_total = 0;
	for(i = 0; i < N_OP_interfaces; ++i) {
		N_states_total += N_init_states_ptr[i];
	}
	py::array_t<int> states = py::array_t<int>(N_states_total * L2);
	py::buffer_info states_info = states.request();
	int *states_ptr = static_cast<int *>(states_info.ptr);

	py::array_t<int> states_parent_inds = py::array_t<int>(N_states_total - N_init_states_ptr[0]);
	// 0th interface has no parents and we don't store anything
	py::buffer_info states_parent_inds_info = states_parent_inds.request();
	int *states_parent_inds_ptr = static_cast<int *>(states_parent_inds_info.ptr);

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

	py::buffer_info _init_state_info;
	int *_init_state_ptr = nullptr;
	if(_init_state.has_value()){
		_init_state_info = _init_state.value().request();
		_init_state_ptr = static_cast<int *>(_init_state_info.ptr);
		assert(_init_state.value().ndim() == 1);
		assert(_init_state.value().shape()[0] == L2);
	}

	if(verbose){
		printf("OP interfaces: \n");
		for(i = 0; i < N_OP_interfaces; ++i){
			printf("%d ", OP_interfaces_ptr[i]);
		}
		printf("\n");
	}

	lattice_gas::run_FFS_C(move_mode, &flux0, &d_flux0, L, e_ptr, mu_ptr, states_ptr, states_parent_inds_ptr,
						   N_init_states_ptr, Nt_ptr, Nt_OP_saved_ptr, stab_step,
						   to_remember_timeevol ? &OP_arr_len : nullptr,
						   OP_interfaces_ptr, N_OP_interfaces,
						   probs_ptr, d_probs_ptr,  to_remember_timeevol? &_E : nullptr,
						   to_remember_timeevol ? &_M : nullptr,
						   to_remember_timeevol ? &_biggest_cluster_sizes : nullptr,
						   to_remember_timeevol ? &_time : nullptr,
						   verbose, init_gen_mode, interface_mode, _init_state_ptr, to_use_smart_swap);

	if(verbose >= 2){
		printf("FFS core done\nNt: ");
	}
	long Nt_total = 0;
	long Nt_OP_saved_total = 0;
	for(i = 0; i < N_OP_interfaces; ++i) {
		if(verbose >= 2){
			printf("%ld ", Nt_ptr[i]);
		}
		Nt_total += Nt_ptr[i];
		Nt_OP_saved_total += Nt_OP_saved_ptr[i];
	}

	py::array_t<double> E;
    py::array_t<int> M;
	py::array_t<int> biggest_cluster_sizes;
	py::array_t<int> time;
    if(to_remember_timeevol){
		int N_last_elements_to_print = 10;
		if(verbose >= 2){
			lattice_gas::print_E(&(_E[Nt_OP_saved_total - N_last_elements_to_print]), N_last_elements_to_print, 'F');
			lattice_gas::print_M(&(_M[Nt_OP_saved_total - N_last_elements_to_print]), N_last_elements_to_print, 'F');
			printf("copying time evolution, Nt_total = %ld\n", Nt_OP_saved_total);
		}

		E = py::array_t<double>(Nt_OP_saved_total);
		py::buffer_info E_info = E.request();
		double *E_ptr = static_cast<double *>(E_info.ptr);
        memcpy(E_ptr, _E, sizeof(double) * Nt_OP_saved_total);
		free(_E);

		M = py::array_t<int>(Nt_OP_saved_total);
		py::buffer_info M_info = M.request();
		int *M_ptr = static_cast<int *>(M_info.ptr);
        memcpy(M_ptr, _M, sizeof(int) * Nt_OP_saved_total);
		free(_M);

		biggest_cluster_sizes = py::array_t<int>(Nt_OP_saved_total);
		py::buffer_info biggest_cluster_sizes_info = biggest_cluster_sizes.request();
		int *biggest_cluster_sizes_ptr = static_cast<int *>(biggest_cluster_sizes_info.ptr);
		memcpy(biggest_cluster_sizes_ptr, _biggest_cluster_sizes, sizeof(int) * Nt_OP_saved_total);
		free(_biggest_cluster_sizes);

		time = py::array_t<int>(Nt_OP_saved_total);
		py::buffer_info time_info = time.request();
		int *time_ptr = static_cast<int *>(time_info.ptr);
		memcpy(time_ptr, _time, sizeof(int) * Nt_OP_saved_total);
		free(_time);

		if(verbose >= 2){
			printf("data copied\n");
			printf("internal memory for EM freed\n");
			lattice_gas::print_E(E_ptr, Nt_OP_saved_total < N_last_elements_to_print ? Nt_OP_saved_total : N_last_elements_to_print, 'P');
			lattice_gas::print_M(M_ptr, Nt_OP_saved_total < N_last_elements_to_print ? Nt_OP_saved_total : N_last_elements_to_print, 'P');
		}
    }

	if(verbose >= 2){
		printf("exiting py::run_FFS\n");
	}
    return py::make_tuple(states, states_parent_inds, probs, d_probs, Nt, Nt_OP_saved, flux0, d_flux0, E, M, biggest_cluster_sizes, time);
}

namespace lattice_gas
{
	std::mt19937 *gen_mt19937;
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

	int run_FFS_C(int move_mode, double *flux0, double *d_flux0, int L, const double *e, const double *mu, int *states,
				  int *states_parent_inds, int *N_init_states, long *Nt, long *Nt_OP_saved, long stab_step,
				  long *OP_arr_len, int *OP_interfaces, int N_OP_interfaces, double *probs, double *d_probs, double **E, int **M,
				  int **biggest_cluster_sizes, int **time, int verbose, int init_gen_mode, int interface_mode,
				  const int *init_state, int to_use_smart_swap)
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
	 * @return - the Error code (none implemented yet)
	 *
  	 */
	{
		int i, j;
		int L2 = L*L;
		int state_size_in_bytes = L2 * sizeof(int);

// get the initial states; they should be sampled from the distribution in [-L^2; M_0), but they are all set to have M == -L^2 because then they all will fall into the local optimum and almost forget the initial state, so it's almost equivalent to sampling from the proper distribution if 'F(M_0) - F_min >~ T'
		long Nt_total = 0;
		long Nt_OP_saved_total = 0;
		long time_0;
		get_init_states_C(move_mode, L, e, mu, &time_0, N_init_states[0], states,
						  init_gen_mode, OP_interfaces[0], stab_step, interface_mode,
						  OP_interfaces[0], OP_interfaces[N_OP_interfaces - 1],
						  E, M, biggest_cluster_sizes, nullptr, time,
						  &Nt_total, &Nt_OP_saved_total, OP_arr_len, init_state, to_use_smart_swap,
						  verbose);
		Nt[0] = Nt_total;
		Nt_OP_saved[0] = Nt_OP_saved_total;

		*flux0 = (double)N_init_states[0] / time_0;   // [1/step]
		*d_flux0 = *flux0 / sqrt(N_init_states[0]);   // [1/step]; this works only if 'N_init_states[0] >> 1 <==>  (d_f/f << 1)'
		if(verbose){
			printf("Init states (N_states = %d, N_OP_saved = %ld, Nt = %ld) generated with mode %d\n", N_init_states[0], Nt_OP_saved[0], Nt[0], init_gen_mode);
			printf("flux0 = (%e +- %e) 1/step\n", *flux0, *d_flux0);
		}

		int N_states_analyzed = 0;
		long Nt_total_prev;
		long Nt_OP_saved_total_prev;
		for(i = 1; i < N_OP_interfaces; ++i){
			Nt_total_prev = Nt_total;
			Nt_OP_saved_total_prev = Nt_OP_saved_total;
			// run each FFS step starting from states saved on the previous step (at &(states[L2 * N_states_analyzed]))
			// and saving steps for the next step (to &(states[L2 * (N_states_analyzed + N_init_states[i - 1])])).
			// Keep track of E, M, CS, the timestep. You have `N_init_states[i - 1]` to choose from to start a trial run
			// and you need to generate 'N_init_states[i]' at the next interface. Use OP_A = OP_interfaces[0], OP_next = OP_interfaces[i].
			probs[i - 1] = process_step(move_mode, &(states[L2 * N_states_analyzed]),
									&(states_parent_inds[N_states_analyzed + N_init_states[i - 1] - N_init_states[0]]),
									&(states[L2 * (N_states_analyzed + N_init_states[i - 1])]),
									E, M, biggest_cluster_sizes, time, &Nt_total, &Nt_OP_saved_total,
									OP_arr_len, N_init_states[i - 1], N_init_states[i],
									L, e, mu, OP_interfaces[0], OP_interfaces[i],
									interface_mode, to_use_smart_swap, verbose); // OP_interfaces[0] - (i == 1 ? 1 : 0)
			//d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] / probs[i]));
			d_probs[i - 1] = probs[i - 1] / sqrt(N_init_states[i - 1] * (1 - probs[i - 1]));

			N_states_analyzed += N_init_states[i - 1];

			Nt[i] = Nt_total - Nt_total_prev;
			Nt_OP_saved[i] = Nt_OP_saved_total - Nt_OP_saved_total_prev;
			if(verbose){
				printf("-log10(p_%d) = (%lf +- %lf)\nNt[%d] = %ld; Nt_total = %ld, Nt_OP[%d] = %ld; Nt_OP_total = %ld\n",
					   i, -log(probs[i-1]) / log(10), d_probs[i-1] / probs[i-1] / log(10),
					   i, Nt[i], Nt_total, i, Nt_OP_saved[i], Nt_OP_saved_total);
				// this assumes p<<1
				if(verbose >= 2){
					int N_last_elements_to_print = 10;
					printf("\nstate[%d] beginning: ", i-1);
					for(j = 0; j < (Nt_OP_saved[i] > N_last_elements_to_print ? N_last_elements_to_print : Nt_OP_saved[i]); ++j)  printf("%d ", states[L2 * N_states_analyzed - N_init_states[i-1] + j]);
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
				printf("Nt_total = %ld\n", Nt_OP_saved_total);
				if(OP_arr_len){
					if(E) print_E(&((*E)[Nt_OP_saved_total - N_last_elements_to_print]), N_last_elements_to_print, 'f');
					if(M) print_M(&((*M)[Nt_OP_saved_total - N_last_elements_to_print]), N_last_elements_to_print, 'f');
				}
				printf("Nt:");
				for(i = 0; i < N_OP_interfaces; ++i) printf(" %ld", Nt[i]);
				printf("\nNt_OP:");
				for(i = 0; i < N_OP_interfaces; ++i) printf(" %ld", Nt_OP_saved[i]);
				printf("\n");
			}
		}

		return 0;
	}

	double process_step(int move_mode, int *init_states, int *states_parent_inds, int *next_states,
						double **E, int **M, int **biggest_cluster_sizes, int **time,
						long *Nt, long *Nt_OP_saved, long *OP_arr_len, int N_init_states, int N_next_states,
						int L, const double *e, const double *mu, int OP_0, int OP_next,
						int interfaces_mode, int to_use_smart_swap, int verbose)
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
		int OP_current;
		int *cluster_element_inds = (int*) malloc(sizeof(int) * L2);
		int *cluster_sizes = (int*) malloc(sizeof(int) * L2);
		int *cluster_types = (int*) malloc(sizeof(int) * L2);
		int *is_checked = (int*) malloc(sizeof(int) * L2);
		if(verbose){
			printf("doing step:(%d; %d]\n", OP_0, OP_next);
//			if(verbose >= 2){
//				printf("press any key to continue...\n");
//				getchar();
//			}
		}
		int init_state_to_process_ID;
//		int run_status;
		long OP_arr_len_old;
//		long Nt_old;
		long time_of_step_total;
		while(N_succ < N_next_states){
			init_state_to_process_ID = gsl_rng_uniform_int(rng, N_init_states);
			if(verbose >= 2){
				printf("state[%d] (id in set = %d):\n", N_succ, init_state_to_process_ID);
			}
			memcpy(state_under_process, &(init_states[init_state_to_process_ID * L2]), state_size_in_bytes);   // get a copy of the chosen init state
//			Nt_old = *Nt;
//			OP_arr_len_old = *OP_arr_len;

			run_state(move_mode, state_under_process, &OP_current, L, e, mu, &time_of_step_total, OP_0, OP_next,
					   E, M, biggest_cluster_sizes, nullptr, time,
					   cluster_element_inds, cluster_sizes, cluster_types, is_checked,
					   Nt, Nt_OP_saved, OP_arr_len, interfaces_mode, to_use_smart_swap, verbose);

			if(OP_current < OP_0){
				++N_runs;
				if(OP_arr_len){ --Nt_OP_saved; } // erase the last OP_arr element since it is from the state that was not saved
			} else if(OP_current >= OP_next) {
				// Interpolation is needed if 's' and 'OP' are continuous

				// Nt is not reinitialized to 0 and that's correct because it shows the total number of OPs datapoints
				++N_runs;
				++N_succ;
				if(next_states) {   // save the resulting system state for the next step
					memcpy(&(next_states[(N_succ - 1) * L2]), state_under_process, state_size_in_bytes);
					states_parent_inds[N_succ - 1] = init_state_to_process_ID;
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
			}

//			switch (run_status) {
//				case 0:  // reached < OP_A  => unsuccessful run
//					++N_runs;
//					break;
//				case 1:  // reached >=OP_next  => successful run
//					// Interpolation is needed if 's' and 'OP' are continuous
//
//					// Nt is not reinitialized to 0 and that's correct because it shows the total number of OPs datapoints
//					++N_runs;
//					++N_succ;
//					if(next_states) {   // save the resulting system state for the next step
//						memcpy(&(next_states[(N_succ - 1) * L2]), state_under_process, state_size_in_bytes);
//						states_parent_inds[N_succ - 1] = init_state_to_process_ID;
//					}
//					if(verbose) {
//						double progr = (double)N_succ/N_next_states;
//						if(verbose < 2){
//							if(N_succ % (N_next_states / 1000 + 1) == 0){
//								printf("%lf %%          \r", progr * 100);
//								fflush(stdout);
//							}
//						} else { // verbose == 1
//							printf("state %d saved for future, N_runs=%d\n", N_succ - 1, N_runs + 1);
//							printf("%lf %%\n", progr * 100);
//						}
//					}
//					break;
////				case -1:  // reached >OP_next => overshoot => discard the trajectory => revert all the "pointers to the current stage" to their previous values
////					*Nt = Nt_old;
////					*OP_arr_len = OP_arr_len_old;
////					break;
//				default:
//					printf("Wrong run_status = %d returned by process_state\n", run_status);
//					assert(false);
//			}
		}
		if(verbose >= 2) {
			printf("\n");
		}

		free(state_under_process);
		free(cluster_element_inds);
		free(cluster_sizes);
		free(cluster_types);
		free(is_checked);

		return (double)N_succ / N_runs;   // the probability P(i+1 | i) to go from i to i+1
	}

	bool is_potential_swap_position(int *state, int L, int ix, int iy)
	{
		int s_group[2 * dim + 1];
		get_spin_with_neibs(state, L, ix, iy, s_group);
		for(int j = 1; j <= 2 * dim; ++j){
			if(s_group[0] != s_group[j]){
				return true;
			}
		}

		return false;
	}

	void update_potential_position(int *state, int L, int pos, std::set< int > *positions){
		auto pos_pos = positions->find(pos);
		if(pos_pos == positions->end()){
			if(is_potential_swap_position(state, L, pos / L, pos % L)){
				positions->insert(pos);
			}
		} else {
			if(!is_potential_swap_position(state, L, pos / L, pos % L)){
				positions->erase(pos_pos);
			}
		}

	}

	void update_neib_potpos(int *state, int L, int ix, int iy, std::set< int > *positions)
	{
		int L2 = L*L;
		int pos = ix * L + iy;
		update_potential_position(state, L, md(ix + 1, L) * L + iy, positions);
		update_potential_position(state, L, md(ix - 1, L) * L + iy, positions);
		update_potential_position(state, L, ix * L + md(iy + 1, L), positions);
		update_potential_position(state, L, ix * L + md(iy - 1, L), positions);
	}

	void find_potential_swaps(int *state, int L, std::set< int > *positions)
	/*
	 * This algorithm is not optimal
	 * A more optimal way would be to cluster the state with a criterion "g_0 != g_neig"
	 */
	{
		int L2 = L*L;
		int i, j;
		int s_group[2 * dim + 1];
		for(i = 0; i < L2; ++i){
			if(is_potential_swap_position(state, L, i / L, i % L)){
				positions->insert(i);
			}
		}
	}

	int run_state(int move_mode, int *s, int *OP_current, int L, const double *e, const double *mu, long *time_total,
				  int OP_0, int OP_next, double **E, int **M, int **biggest_cluster_sizes, int **h_A, int **time,
				  int *cluster_element_inds, int *cluster_sizes, int *cluster_types, int *is_checked, long *Nt, long *Nt_OP_saved,
				  long *OP_arr_len, int interfaces_mode, int to_use_smart_swap, int verbose, int to_cluster, long Nt_max,
				  int *states_to_save, int *N_states_saved, int N_states_to_save, int OP_min_save_state, int OP_max_save_state,
				  int save_state_mode, int OP_A, int OP_B, long save_states_stride)
	/**
	 *	Run state saving states in (OP_min_save_state, OP_max_save_state) and saving OPs for the whole (OP_0; OP_next)
	 *
	 * @param move_mode - the mode of MC moves
	 * @param s - the current state the system under simulation is in
	 * @param OP_current - to be able to return the last value of the OP
	 * @param L - see run_FFS
	 * @param e - see run_FFS
	 * @param mu - see run_FFS
	 * @param time_total - the total time (including failed move attempts) passed
	 * @param OP_0 - see process_step
	 * @param OP_next - see process_step
	 * @param E, M, biggest_cluster_sizes - see run_FFS
	 * @param h_A - the history-dependent function, =1 for the states that came from A, =0 otherwise (i.e. for the states that came from B)
	 * @param time - number of MC attempts the system stayed at a given state (array Nt x 1)
	 * @param cluster_element_inds - array of ints, the indices of spins participating in clusters
	 * @param cluster_sizes - array of ints, sizes of clusters
	 * @param cluster_types - array of ints, types of found clusters
	 * @param is_checked - array of ints L^2 in size. Labels of spins necessary for clustering. Says which spins are already checked by the clustering procedure during the current clustering run
	 * `cluster_element_inds` are aligned with `cluster_sizes` - there are `cluster_sizes[0]` of indices making the 1st cluster, `cluster_sizes[1]` indices of the 2nd slucter, so on.
	 * @param Nt - see run_FFS
	 * @param Nt_OP_saved - number of saved time-evol states ('state' + 'E', 'M', etc.)
	 * @param OP_arr_len - see process_step
	 * @param interfaces_mode - see run_FFS
	 * @param to_use_smart_swap - whether to use the simplified energy difference for local and non-local swaps. !!!NOT IMPLEMENTED PROPERLY!!!
	 * @param verbose - see run_FFS
	 * @param Nt_max - int, default -1; It's it's > 0, then the simulation is stopped on '*Nt >= Nt_max'
	 * @param states_to_save - int*, default nullptr; The pointer to the set of states to save during the run
	 * @param N_states_saved - int*, default nullptr; The number of states already saved in 'states_to_save'
	 * @param N_states_to_save - int, default -1; If it's >0, the simulation is stopped when 'N_states_saved >= N_states_to_save'
	 * @param OP_min_save_state, OP_max_save_state - int, default 0; If(N_states_to_save > 0), states are saved when M == M_thr_save_state
	 * @param save_state_mode :
	 * 1 ("inside region") = save states that have OP in (OP_min_save_state; OP_max_save_state];
	 * 2 ("outflux") = save states that have `OP_current >= OP_min_save_state` and `OP_prev < OP_min_save_state`
	 * @param OP_A, OP_B - A dna B doundaries to compute h_A
	 * @param save_states_stride - this many successful flips between saving the full system state
	 *
	 * @return - the Error status (none implemented yet)
	 */
	{
		int L2 = L * L;
		int state_size_in_bytes = sizeof(int) * L2;
		int h_A_current = 0;   // TODO pass this as an argument to make it transferable between runs
		int OP_prev;
		int M_current = comp_M(s, L); // remember the 1st M;
		double E_current = comp_E(s, L, e, mu); // remember the 1st energy;
		int N_clusters_current = L2;   // so that all uninitialized cluster_sizes are set to 0
		int biggest_cluster_sizes_current = 0;
		bool verbose_BF = (verbose < 0);
		if(verbose_BF) verbose = -verbose;

		if((abs(M_current) > L2) || ((L2 - M_current) % OP_step[mode_ID_M] != 0)){   // check for sanity
			state_is_valid(s, L, 0, 'e');
			if(verbose){
				printf("This state has M = %d (L = %d, dM_step = 2) which is beyond possible physical boundaries, there is something wrong with the simulation\n", M_current, L);
				getchar();
			}
		}
		if(N_states_to_save > 0){
			assert(states_to_save);
		}

		if(to_cluster){
			clear_clusters(cluster_element_inds, cluster_sizes, &N_clusters_current);
			uncheck_state(is_checked, L2);
			cluster_state_C(s, L, cluster_element_inds, cluster_sizes, cluster_types, &N_clusters_current, is_checked);
			biggest_cluster_sizes_current = max(cluster_sizes, N_clusters_current);
		} else {
			biggest_cluster_sizes_current = 1;
		}

		switch (interfaces_mode) {
			case mode_ID_M:
				*OP_current = M_current;
				break;
			case mode_ID_CS:
				*OP_current = biggest_cluster_sizes_current;
				break;
			default:
				assert(false);
		}

		OP_prev = *OP_current;

		if(verbose >= 2){
			printf("E=%lf, M=%d, OP_0=%d, OP_next=%d\n", E_current, M_current, OP_0, OP_next);
		}

		int ix, iy, s_new;   // for flip moves
		int ix_new, iy_new, s_swap;   // for swap moves
		double dE;
		int time_the_flip_took;
		*time_total = 0;
		std::set< int > potential_swap_positions;
		if(to_use_smart_swap){
			find_potential_swaps(s, L, &potential_swap_positions);
		}

		double E_tolerance = 1e-5;   // J=1
		unsigned long long Nt_for_numerical_error = sqr(lround(1e13 * E_tolerance / L2));
		// the error accumulates, so we need to recompute form scratch time to time
//      |E| ~ L2 => |dE_numerical| ~ L2 * 1e-13 => |dE_num_total| ~ sqrt(Nt) * L2 * 1e-13 << E_tolerance => Nt << (1e13 * E_tolerance / L2)^2

		while(1){
			// ----------- decide on state change -----------
			switch (move_mode) {
				case move_mode_flip:
					time_the_flip_took = flip_move(s, L, e, mu, &ix, &iy, &s_new, &dE);
					break;
				case move_mode_swap:
					time_the_flip_took = swap_move(s, L, e, mu, &ix, &iy, &ix_new, &iy_new, &dE,
												   to_use_smart_swap ? &potential_swap_positions : nullptr);
					break;
				case move_mode_long_swap:
					time_the_flip_took = long_swap_move(s, L, e, mu, &ix, &iy, &ix_new, &iy_new, &dE);
					break;
			}

			// ------------------ save timeevol ----------------
			if(OP_arr_len){
				if((*Nt) % save_states_stride == 0){
					if(*Nt_OP_saved >= *OP_arr_len){ // double the size of the time-index
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

					if(time) (*time)[*Nt_OP_saved] = time_the_flip_took;
					if(E) (*E)[*Nt_OP_saved] = E_current;
					if(M) (*M)[*Nt_OP_saved] = M_current;
					if(biggest_cluster_sizes) (*biggest_cluster_sizes)[*Nt_OP_saved] = biggest_cluster_sizes_current;
					if(h_A) (*h_A)[*Nt_OP_saved] = h_A_current;
					++ (*Nt_OP_saved);
					if(verbose >= 4){ printf("Nt_OP_saved = %d\n", *Nt_OP_saved); }
				}

			}

			// ------------------- save the state if it's good (we don't want failed states) -----------------
			if(N_states_to_save > 0){
				bool to_save_state = ((*Nt) % save_states_stride == 0);
				if(to_save_state){
					switch (save_state_mode) {
						case save_state_mode_Inside:
							to_save_state = (*OP_current >= OP_min_save_state) && (*OP_current < OP_max_save_state);
							break;
						case save_state_mode_Influx:
							to_save_state = (*OP_current >= OP_max_save_state) && (OP_prev < OP_min_save_state);
							break;
						case save_state_mode_Outside:
							to_save_state = (*OP_current >= OP_max_save_state) || (*OP_current < OP_min_save_state);
							break;
						default:
							to_save_state = false;
							if(verbose){
								printf("WARNING:\nN_states_to_save = %d > 0 provided, but wrong save_state_mode = %d. Not saving states\n", N_states_to_save, save_state_mode);
								STP
							}
					}
				}
				if(to_save_state){
					memcpy(&(states_to_save[*N_states_saved * L2]), s, state_size_in_bytes);
					++(*N_states_saved);
				}
			}

			// ---------------- check BF exit ------------------
			// values of M_next are recorded, and states starting from M_next of the next stage are not recorded, so I have ...](... M accounting
			if(N_states_to_save > 0){
				if(*N_states_saved >= N_states_to_save){
					if(verbose >= 2) printf("Reached desired N_states_saved = %d >= N_states_to_save (= %d)\n", *N_states_saved, N_states_to_save);
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

			// ------------------ check FFS exit ----------------
			if(*OP_current < OP_0){   // gone to the initial state A
				if(verbose >= 3) printf("\nReached OP_0 = %d, OP_current = %d, OP_mode = %d\n", OP_0, *OP_current, interfaces_mode);
				return 0;
			} else if(*OP_current >= OP_next){
				if(verbose >= 3) printf("Reached OP_next = %d, *OP_current = %d, OP_mode = %d\n", OP_next, *OP_current, interfaces_mode);
				return 1;
//				return *OP_current == OP_next ? 1 : -1;
				// 1 == succeeded = reached the interface 'M == M_next'
				// -1 == overshoot = discard the trajectory
			}

			// ----------- modify stats -----------
			switch (move_mode) {
				case move_mode_flip:
					M_current += (s_new == main_specie_id ? 1 : (s[ix*L + iy] == main_specie_id ? -1 : 0));
					s[ix * L + iy] = s_new;
					break;
				case move_mode_swap:
					std::swap(s[ix * L + iy], s[ix_new * L + iy_new]);
					if(to_use_smart_swap){
						update_neib_potpos(s, L, ix, iy, &potential_swap_positions);
						update_neib_potpos(s, L, ix_new, iy_new, &potential_swap_positions);
					}
					break;
				case move_mode_long_swap:
					std::swap(s[ix * L + iy], s[ix_new * L + iy_new]);
					break;
			}

			*time_total += time_the_flip_took;
			E_current += dE;

			if(to_cluster){
				clear_clusters(cluster_element_inds, cluster_sizes, &N_clusters_current);
				uncheck_state(is_checked, L2);
				cluster_state_C(s, L, cluster_element_inds, cluster_sizes, cluster_types, &N_clusters_current, is_checked);
				biggest_cluster_sizes_current = max(cluster_sizes, N_clusters_current);
			} else {
				biggest_cluster_sizes_current = 1;
			}

			if(h_A){
//				h_A_prev = h_A_current;
				h_A_current = (*Nt == 0 ? (*OP_current - OP_A > OP_B - *OP_current ? 0 : 1) : (h_A_current == 1 ? (*OP_current < OP_B ? 1 : 0) : (*OP_current >= OP_A ? 0 : 1)));
			}

			++(*Nt);

			if(verbose_BF){
				if(!(*Nt % (81920000 / L2))){
					if(Nt_max > 0){
						printf("BF run: %lf %%, Nt_OP = %ld, CS = %d", (double)(*Nt) / (Nt_max) * 100, Nt_OP_saved ? *Nt_OP_saved : -1, biggest_cluster_sizes_current);
					} else {
						printf("BF run: Nt = %ld, Nt_OP = %ld, CS = %d", *Nt, Nt_OP_saved ? *Nt_OP_saved : -1, biggest_cluster_sizes_current);
					}

					qsort(cluster_sizes, N_clusters_current, sizeof(int), cmpfunc_decr<int>);
					printf(";       CSs:");
					for(int j = 0; j < std::min(10, N_clusters_current); ++j){
						printf(" %d,", cluster_sizes[j]);
					}
					printf("               \r");

					fflush(stdout);
				}
			}

			// ------------ update the OP --------------
			OP_prev = *OP_current;
			switch (interfaces_mode) {
				case mode_ID_M:
					*OP_current = M_current;
					break;
				case mode_ID_CS:
					*OP_current = biggest_cluster_sizes_current;
					break;
				default:
					printf("ERROR\n");
			}

			// -------------- check that error in E is negligible -----------
			// we need to do this since E is double so the error accumulated over steps
			if(*Nt % Nt_for_numerical_error == 0){
				double E_curent_real = comp_E(s, L, e, mu);
				if(abs(E_current - E_curent_real) > E_tolerance){
					if(verbose >= 2){
						printf("\nE-error out of bound: E_current = %lf, dE = %lf, Nt = %ld, E_real = %lf\n", E_current, dE, *Nt, E_curent_real);
						print_E(&((*E)[*Nt_OP_saved - 10]), 10);
						print_S(s, L, 'r');
//						throw -1;
//						getchar();
					}
					E_current = E_curent_real;
				}
			}
		}
	}

	int get_OP_from_spinsup(int N_spins_up, int L2, int interface_mode)
	{
		switch (interface_mode) {
			case mode_ID_M:
				return N_spins_up;
			case mode_ID_CS:
				return N_spins_up;
		}
	}

	int run_bruteforce_C(int move_mode, int L, const double *e, const double *mu, long *time_total, int N_states, int *states,
						 long *OP_arr_len, long *Nt, long *Nt_OP_saved, double **E, int **M, int **biggest_cluster_sizes, int **h_A, int **time,
						 int interface_mode, int OP_A, int OP_B, int to_cluster, int to_start_only_state0,
						 int OP_min_stop_state, int OP_max_stop_state, int *N_states_done,
						 int OP_min_save_state, int OP_max_save_state, int save_state_mode,
						 int N_spins_up_init, int verbose, long Nt_max, int *N_tries, int to_save_final_state,
						 int to_regenerate_init_state, long save_states_stride, int to_use_smart_swap)
	/**
	 *
	 * @param L - see run_FFS
	 * @param e - see run_FFS
	 * @param mu - see run_FFS
	 * @param time_total - the total physical time "passed" in the system
	 * @param N_states - int
	 * 	if > 0, then the run is going until the `N_states` states are saved. Saving happens based on other threshold parameters passed to the function
	 * 	if == 0, then ignored meaning it won't affect the stopping criteria, so the rub will be until Nt > Nt_max
	 * @param states - array containing all the states saved according to the passed threshold parameters
	 * @param OP_arr_len - see run_FFS
	 * @param Nt - the number of timesteps made
	 * @param E, M, biggest_cluster_sizes, time, h_A - see run_FFS
	 * @param interface_mode - see run_FFS
	 * @param OP_A - see run_state
	 * @param OP_B - see run_state
	 * @param OP_min_stop_state, OP_max_stop_state - if OP goes outside of (OP_min_stop_state; OP_max_stop_state], then the run is restarted from a state randomly chosen from the already saved states
	 * @param N_states_done - the number of saved states
	 * @param OP_min_save_state - see run_state
	 * @param OP_max_save_state - see run_state
	 * @param save_state_mode - see run_state
	 * @param N_spins_up_init - the number of cells == 1 in the initial configuration. I use =0 for our cases.
	 * 	if < 0, then such an N_spins_up_init is chosen so that the system starts with `OP == OP_min_stop_state` (the minimum value above the stopping occurs)
	 * @param verbose - see run_FFS
	 * @param Nt_max - if > 0, then the run is completed when Nt > Nt_max
	 * @param N_tries - The number of times the BF run got outside of [OP_min_stop_state; OP_max_stop_state) and thus was restarted
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
					N_spins_up_init = OP_min_stop_state + 1;
					break;
				case mode_ID_CS:
					N_spins_up_init = OP_min_stop_state + 1;
					break;
				default:
					printf("ERROR\n");
			}
		}

//		int OP_init = get_OP_from_spinsup(N_spins_up_init, L2, interface_mode);
//		assert(OP_min_save_state <= OP_init);
//		assert(OP_init <= OP_max_save_state);

		int *cluster_element_inds = (int*) malloc(sizeof(int) * L2);
		int *cluster_sizes = (int*) malloc(sizeof(int) * L2);
		int *cluster_types = (int*) malloc(sizeof(int) * L2);
		int *is_checked = (int*) malloc(sizeof(int) * L2);
		int *state_under_process = (int*) malloc(state_size_in_bytes);
		int OP_current;

		if(to_regenerate_init_state){
			generate_state(states, L, N_spins_up_init, interface_mode, verbose);
//			print_S(states, L, '0'); 		getchar();
//			*N_states_done += 1;
			*N_states_done = 0;
		}
		int restart_state_ID;

		if(verbose){
			printf("running brute-force:\nL=%d  to_cluster=%d  OP_mode=%d  OP\\in[%d;%d), [OP_min_save_state; OP_max_save_state) = [%d; %d)  N_spins_up_init=%d  N_states_to_gen=%d  Nt_max=%ld  stride=%ld  smart_swap=%d  verbose=%d\n",
				   L, to_cluster, interface_mode, OP_min_stop_state, OP_max_stop_state, OP_min_save_state, OP_max_save_state, N_spins_up_init, N_states, Nt_max, save_states_stride, to_use_smart_swap, verbose);
			print_e_matrix(e);
			print_mu_vector(mu);
			switch (save_state_mode) {
				case save_state_mode_Influx:
					printf("OP_A = %d\n", OP_min_save_state);
					break;
			}
		}

		*N_tries = 0;
		*time_total = 0;
		long time_the_try_took;
		while(1){
			// initially start from the 0th state (the only one we have at the time). Next times start from a random state from the ones we have at the time
			if((*N_states_done > 0) && (!to_start_only_state0)){
				restart_state_ID = gsl_rng_uniform_int(rng, *N_states_done);
				if(verbose >= 2){
					printf("generated %d states, restarting from state[%d]\n", *N_states_done, restart_state_ID);
//					print_S(&(states[L2 * restart_state_ID]), L, 't');
				}
			} else {
				restart_state_ID = 0;
				if(verbose >= 2){
					printf("restarting from initial state\n");
				}
			}

			memcpy(state_under_process, &(states[L2 * restart_state_ID]), state_size_in_bytes);   // get a copy of the chosen init state

			run_state(move_mode, state_under_process, &OP_current, L, e, mu, &time_the_try_took,
					  OP_min_stop_state, OP_max_stop_state,
					  E, M, biggest_cluster_sizes, h_A, time, cluster_element_inds, cluster_sizes, cluster_types,
					  is_checked, Nt, Nt_OP_saved, OP_arr_len, interface_mode, to_use_smart_swap,
					  -verbose, to_cluster, Nt_max, states, N_states_done, N_states,
					  OP_min_save_state, OP_max_save_state, save_state_mode, OP_A, OP_B, save_states_stride);

			++ (*N_tries);
			*time_total += time_the_try_took;

			if(N_states > 0){
//				printf("v: %d\n", verbose);
//				STP
				if(verbose > 0)	{
					printf("brute-force done %lf %%, N_tries = %d, N_states_done = %d              \n", 100 * (double)(*N_states_done) / N_states, *N_tries, *N_states_done);
					fflush(stdout);
				}
				if(*N_states_done >= N_states) {
					if(verbose) printf("\n");
					if(OP_current < OP_max_stop_state) --(*N_tries);
					break;
				}
			}
			if(Nt_max > 0){
				if(verbose > 0)	{
					printf("brute-force done %lf %%, N_tries = %d,  Nt = %ld             \r", 100 * (double)(*Nt) / Nt_max, *N_tries, *Nt);
					fflush(stdout);
				}
				if(*Nt >= Nt_max) {
					if(OP_current < OP_max_stop_state) --(*N_tries);
					if(verbose) printf("\n");
					if(to_save_final_state){
						memcpy(&(states[L2 * *N_states_done]), state_under_process, state_size_in_bytes);
						++ (*N_states_done);

//						printf("OP_c = %d", OP_current);
//						print_S(state_under_process, L, 'r');
//						getchar();
					}
					break;
				}
			}
//			if(Nt_max > 0) if(*Nt >= Nt_max) break;
		}

		free(cluster_element_inds);
		free(cluster_sizes);
		free(cluster_types);
		free(is_checked);
		free(state_under_process);

		return 0;
	}

	int get_equilibrated_state(int move_mode, int L, const double *e, const double *mu, int *state, int *N_states_done,
							   int interface_mode, int OP_A, int OP_B, long stab_step, const int *init_state,
							   int to_use_smart_swap, int to_equilibrate, int verbose)
	{
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;

		long time_total;
		long Nt_to_reach_OP_A = 0;
		int N_states_done_local = *N_states_done;
		int N_tries;
//		long Nt = 0;

		if(init_state){
			memcpy(state, init_state, state_size_in_bytes);
		}

//		print_S(&(state[0 * L2]), L, 'q');
//		getchar();

		if(to_equilibrate){
			run_bruteforce_C(move_mode, L, e, mu, &time_total, 1, state,
							 nullptr, &Nt_to_reach_OP_A, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
							 interface_mode, -1, -1, 1, 0,
							 OP_min_default[interface_mode], OP_A,
							 &N_states_done_local, OP_A,
							 OP_A, save_state_mode_Influx, -1,
							 verbose, -1, &N_tries, 0, ! bool(init_state),
							 1, to_use_smart_swap);
			if(verbose > 0){
				printf("reached OP >= OP_A = %d in Nt = %ld MC steps\n", OP_A, Nt_to_reach_OP_A);
			}
			// replace the "all down" init state with the "at OP_A_thr" state
//			memcpy(init_states, &(init_states[L2]), sizeof(int) * L2);

//		printf("N_states_done = %d", N_states_done);
//		print_S(&(state[(N_states_done - 1) * L2]), L, 'w');
//		getchar();

//		long N_steps_to_equil = 2 * Nt_to_reach_OP_A + stab_step;   // * 2 because we already have Nt = Nt_reach, and we want to run Nt_reach+stab_step new steps
			long N_steps_to_equil = Nt_to_reach_OP_A + stab_step;
			long Nt;
			N_tries = 0;
			do{
				*N_states_done = N_states_done_local;
				Nt = Nt_to_reach_OP_A;
				if(verbose > 0){
//				printf("Proc N_states_done = %d\n", N_states_done);
					printf("Attempting to simulate Nt = %ld MC steps towards the local optimum            \r", N_steps_to_equil);
					if(N_tries > 0){
						printf("Previous attempt results in %d reaches of state B (OP_B = %d), so restating from the initial ~OP_A\n", N_tries, OP_B);
					}
				}

				// run it for the same amount of time it took to get to OP_A_thr the first time + 10 sweeps
				run_bruteforce_C(move_mode, L, e, mu, &time_total, -1, state,
								 nullptr, &Nt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
								 interface_mode, -1, -1, 1, 0,
								 OP_min_default[interface_mode], OP_B,
								 N_states_done, -1,
								 -1, -1, -1,
								 0, N_steps_to_equil, &N_tries, 1, 0,
								 1, to_use_smart_swap);
				if(N_tries == 0){
					if(get_max_CS(&(state[(*N_states_done - 1) * L2]), L) < OP_A){
						memcpy(&(state[(N_states_done_local - 1) * L2]), &(state[(*N_states_done - 1) * L2]), state_size_in_bytes);
						*N_states_done = N_states_done_local;
						break;
					} else {
						N_steps_to_equil = std::max(lround(N_steps_to_equil * 0.5), (long)(L2 * 1));
					}
				} else {
					N_steps_to_equil = lround(N_steps_to_equil * 0.9 / N_tries);
					// The fact that N_tries>0 means that system nucleated
					// Nucleations after Nt_to_reach_OP_A is unlikely if OP_A < N*
					// Thus if stab_step << Nt_to_reach_OP_A, then most likely N_tries = 0
					// So N_tries > 0 ==> stab_step >~ Nt_to_reach_OP_A
					// We need to find maximum N_steps_to_equil for which there is a good chance to get N_tries=0
					// Such N_steps_to_equil is ~N_tries smaller than current N_steps_to_equil
					//// but *1.5 go not get always down but looks for a maximum possible time
				}
			}while(true);

			printf("Equilibrated state generated                                                \n");
			// N_tries = 0 in the beginning of BF. If we went over the N_c, I want to restart because we might not have obtained enough statistic back around the optimum.
		}

		return 0;
	}

	int get_init_states_C(int move_mode, int L, const double *e, const double *mu, long *time_total, int N_init_states,
						  int *init_states, int mode, int OP_thr_save_state, long stab_step,
						  int interface_mode, int OP_A, int OP_B,
						  double **E, int **M, int **biggest_cluster_size, int **h_A, int **time,
						  long *Nt, long *Nt_OP_saved, long *OP_arr_len, const int *init_state, int to_use_smart_swap,
						  int verbose)
	/**
	 *
	 * @param L - see run_FFS
	 * @param e - see run_FFS
	 * @param mu - see run_FFS
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
		int state_size_in_bytes = sizeof(int) * L2;

		if(verbose){
			printf("generating states:\nN_init_states=%d, gen_mode=%d, OP_A <= %d, OP_mode=%d\n", N_init_states, mode, OP_thr_save_state, interface_mode);
		}

		if((mode >= gen_init_state_mode_SinglePiece) || (mode == gen_init_state_mode_Random)){
			// generate N_init_states states in A
			// Here they are identical, but I think it's better to generate them accordingly to equilibrium distribution in A
			for(i = 0; i < N_init_states; ++i){
				generate_state(&(init_states[i * L2]), L, mode, interface_mode, verbose);
			}
		} else if(mode == gen_init_state_mode_Inside){
			int N_states_done;
			int N_tries;
//			long Nt = 0;

			if(init_state){
				memcpy(init_states, init_state, state_size_in_bytes);
			}

			run_bruteforce_C(move_mode, L, e, mu, time_total, N_init_states, init_states,
							 nullptr, Nt, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
							 interface_mode, 0, 0, 1, 0,
							 OP_min_default[interface_mode], OP_B,
							 &N_states_done, OP_min_default[interface_mode],
							 OP_thr_save_state, save_state_mode_Inside, -1,
							 verbose, -1, &N_tries, 0, ! bool(init_state),
							 1, to_use_smart_swap);
		} else if(mode == gen_init_state_mode_Influx){
			int N_states_done = 0;
			int N_tries = 0;

			get_equilibrated_state(move_mode, L, e, mu, init_states, &N_states_done, interface_mode,
								   OP_thr_save_state, OP_B, stab_step, init_state, to_use_smart_swap,
								   1, verbose);

			*Nt = 0;   // forget anything we might have had
			N_states_done = 0;  // this is =1 because one state in [...; OP_interface[0]) is saved to start outflux generation from it
			*Nt_OP_saved = 0;
			*time_total = 0;

//			printf("N_states_done = %d\n", N_states_done);
//			print_S(&(init_states[0 * L2]), L, 'i');
//			print_S(&(init_states[1 * L2]), L, 'i');
//			print_S(&(init_states[2 * L2]), L, 'i');
//			getchar();

			run_bruteforce_C(move_mode, L, e, mu, time_total, N_init_states, init_states,
							 OP_arr_len, Nt, Nt_OP_saved, E, M, biggest_cluster_size, h_A, time,
							 interface_mode, OP_A, OP_B, 1, 0,
							 OP_min_default[interface_mode], OP_B,
							 &N_states_done, OP_thr_save_state,
							 OP_thr_save_state, save_state_mode_Influx, -1,
							 verbose, -1, &N_tries, 0, 0, 1,
							 to_use_smart_swap);
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
//				} else if(s[i] == main_specie_id) {
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

//	int is_infinite_cluster(const int* cluster, const int* cluster_size, int L, char *present_rows, char *present_columns)
//	{
//		int i = 0;
//
//		zero_array(present_rows, L);
//		zero_array(present_columns, L);
//		for(i = 0; i < (*cluster_size); ++i){
//			present_rows[cluster[i] / L] = 1;
//			present_columns[cluster[i] % L] = 1;
//		}
//		char cluster_is_infinite_x = 1;
//		char cluster_is_infinite_y = 1;
//		for(i = 0; i < L; ++i){
//			if(!present_columns[i]) cluster_is_infinite_x = 0;
//			if(!present_rows[i]) cluster_is_infinite_y = 0;
//		}
//
//		return cluster_is_infinite_x || cluster_is_infinite_y;
//	}

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

	double comp_E(const int* state, int L, const double *e, const double *mu)
	/**
	 * Computes the Energy of the state 's' of the linear size 'L', immersed in the 'h' magnetic field;
	 * H = E/T = \sum_{<ij>} s_i s_j e[s_i][s_j] + \sum_i mu[s_i]
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

		return _M + _E;    // e, mu > 0 -> we need to *(-1) because we search for a minimum
	}

	int generate_state(int *s, int L, int mode, int interface_mode, int verbose)
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

		if(mode >= gen_init_state_mode_SinglePiece){   // generate |mode| spins UP, and the rest spins DOWN
			for(i = 0; i < L2; ++i) s[i] = background_specie_id;
			if(mode > gen_init_state_mode_SinglePiece){
				int N_down_spins = mode - gen_init_state_mode_SinglePiece;
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

							s[swap_var] = main_specie_id;
						}
						free(indices_to_flip);
					}
						break;
					case mode_ID_CS:
						for(i = 0; i < N_down_spins; ++i){
							s[i] = main_specie_id;
						}
						break;
				}
			}
			int OP_init = get_OP_from_spinsup(mode, L2, interface_mode);
			if(verbose){
				printf("generated state: N_spins_up=%d, OP_mode=%d, OP=%d\n", mode, interface_mode, OP_init);
			}
		} else if(mode == gen_init_state_mode_Random){   // random
			for(i = 0; i < L2; ++i) s[i] = gsl_rng_uniform_int(rng, N_species);
		}

		return 0;
	}

	double new_spin_energy(const double *e, const double *mu, const int *s_neibs, int s_new)
	{
		int s_ix = s_new * N_species;
		return mu[s_new] + (e[s_ix + s_neibs[1]] + e[s_ix + s_neibs[2]] + e[s_ix + s_neibs[3]] + e[s_ix + s_neibs[4]]);
	}

	void get_spin_with_neibs(const int *state, int L, int ix, int iy, int *s_group)
	{
		s_group[0] = state[ix * L + iy];
		s_group[1] = state[md(ix + 1, L)*L + iy];
		s_group[2] = state[ix*L + md(iy + 1, L)];
		s_group[3] = state[md(ix - 1, L)*L + iy];
		s_group[4] = state[ix*L + md(iy - 1, L)];
	}

	double long_swap_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int ix_new, int iy_new)
	{
		int s1[2 * dim + 1];
		int s2[2 * dim + 1];
		get_spin_with_neibs(state, L, ix, iy, s1);
		get_spin_with_neibs(state, L, ix_new, iy_new, s2);

		return (new_spin_energy(e, mu, s1, s2[0]) + new_spin_energy(e, mu, s2, s1[0])) -
			   (new_spin_energy(e, mu, s1, s1[0]) + new_spin_energy(e, mu, s2, s2[0]));
	}

	double short_swap_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int ix_new, int iy_new)
	{
		assert(state);
		int L2 = L*L;
		int R = L/2;
		int dx = mds(ix_new - ix, R);
		int dy = mds(iy_new - iy, R);
//		int pos = ix * L + iy;
//		int pos_new = ix_new * L + iy_new;
//		double dE;
		int s_ix = state[ix * L + iy] * N_species;
		int snew_ix = state[ix_new * L + iy_new] * N_species;
		int s_neibs[4 * dim - 2];

		if(dx != 0){
			s_neibs[0] = state[md(ix_new + dx, L) * L + iy_new];
			s_neibs[1] = state[ix_new * L + md(iy_new + 1, L)];
			s_neibs[2] = state[ix_new * L + md(iy_new - 1, L)];
			s_neibs[3] = state[md(ix - dx, L) * L + iy];
			s_neibs[4] = state[ix * L + md(iy + 1, L)];
			s_neibs[5] = state[ix * L + md(iy - 1, L)];
		} else if(dy != 0) {
			s_neibs[0] = state[md(ix_new + 1, L) * L + iy_new];
			s_neibs[1] = state[md(ix_new - 1, L) * L + iy_new];
			s_neibs[2] = state[ix_new * L + md(iy_new + dy, L)];
			s_neibs[3] = state[md(ix + 1, L) * L + iy];
			s_neibs[4] = state[md(ix - 1, L) * L + iy];
			s_neibs[5] = state[ix * L + md(iy - dy, L)];
		} else {
			fprintf(stderr, "ERROR: dx = %d, dy = %d\n(ix, iy) = (%d, %d); (ix, iy)_new = (%d, %d)\nFor short_swap_dE\nAborting\n",
					dx, dy, ix, iy, ix_new, iy_new);
			assert(false);
		}

//		printf("%lf, %lf, %lf, %lf\n",
//			   e[s_ix + s_neibs[0]] + e[s_ix + s_neibs[1]] + e[s_ix + s_neibs[2]],
//			   e[snew_ix + s_neibs[3]] + e[snew_ix + s_neibs[4]] + e[snew_ix + s_neibs[5]],
//			   e[snew_ix + s_neibs[0]] + e[snew_ix + s_neibs[1]] + e[snew_ix + s_neibs[2]],
//			   e[s_ix + s_neibs[3]] + e[s_ix + s_neibs[4]] + e[s_ix + s_neibs[5]]);

		return (e[s_ix + s_neibs[0]] + e[s_ix + s_neibs[1]] + e[s_ix + s_neibs[2]] +
			  e[snew_ix + s_neibs[3]] + e[snew_ix + s_neibs[4]] + e[snew_ix + s_neibs[5]]) -
			 (e[snew_ix + s_neibs[0]] + e[snew_ix + s_neibs[1]] + e[snew_ix + s_neibs[2]] +
			  e[s_ix + s_neibs[3]] + e[s_ix + s_neibs[4]] + e[s_ix + s_neibs[5]]);
	}

	double swap_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int ix_new, int iy_new)
	{
		if(abs(ix - ix_new) + abs(iy - iy_new) > 1){
			return long_swap_mode_dE(state, L, e, mu, ix, iy, ix_new, iy_new);
		} else {
			return short_swap_mode_dE(state, L, e, mu, ix, iy, ix_new, iy_new);
		}
	}

	double flip_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int s_new)
	{
		int s[2 * dim + 1];
		get_spin_with_neibs(state, L, ix, iy, s);

		return new_spin_energy(e, mu, s, s_new) - new_spin_energy(e, mu, s, s[0]);
	}

	int long_swap_move(const int *state, uint L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new, double *dE)
	{
		int N_flip_tries = 0;
		uint L2 = L * L;
		ulong total_range = L2 * (L2 - 1);
		ulong rnd;
		uint pos, shift;
		bool to_swap = false;

		do{
			rnd = gsl_rng_uniform_int(rng, total_range);

			pos = rnd % L2;
			*iy = pos % L;
			*ix = pos / L;

			pos = (pos + rnd / L2 + 1) % L2;
			*iy_new = pos % L;
			*ix_new = pos / L;

			to_swap = (state[*ix * L + *iy] != state[pos]);
			if(to_swap){
				*dE = swap_mode_dE(state, L, e, mu, *ix, *iy, *ix_new, *iy_new);
				to_swap = (*dE <= 0 ? true : (gsl_rng_uniform(rng) < exp(- *dE)));
			}

			++N_flip_tries;
		}while(!to_swap);

		return N_flip_tries;
	}

	int swap_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new,
				  double *dE, const std::set< int > *swap_positions)
	{
		int N_flip_tries = 0;
		int L2 = L * L;
		int total_range;
		int rnd, pos, direction;
		bool to_swap = false;

		assert(!swap_positions);

		if(swap_positions){
			int sgn;
			total_range = dim * 2;
			do{
				std::sample( swap_positions->begin(), swap_positions->end(), &pos, 1, *gen_mt19937);
				direction = gsl_rng_uniform_int(rng, total_range);
				sgn = (direction % 2) * 2 - 1;
				direction = direction % dim;

				*iy = pos % L;
				*ix = pos / L;

				*ix_new = md(*ix + sgn * (direction == 0), L);
				*iy_new = md(*iy + sgn * (direction == 1), L);

				to_swap = (state[pos] != state[*ix_new * L + *iy_new]);
				if(to_swap){
					*dE = swap_mode_dE(state, L, e, mu, *ix, *iy, *ix_new, *iy_new);
					to_swap = (*dE <= 0 ? true : (gsl_rng_uniform(rng) < exp(- *dE)));
				}

				++N_flip_tries;
			}while(!to_swap);
		} else {
			total_range = L2 * dim;
			do{
				assert(L>0);
				rnd = gsl_rng_uniform_int(rng, total_range);
				direction = rnd / L2;   // we try only x+1 and y+1 flips (and not x-1, y-1), since it covers all the possible flips
				pos = rnd % L2;

				*iy = pos % L;
				*ix = pos / L;

				*ix_new = md(*ix + (direction == 0), L);
				*iy_new = md(*iy + (direction == 1), L);

				to_swap = (state[pos] != state[*ix_new * L + *iy_new]);
				if(to_swap){
					*dE = swap_mode_dE(state, L, e, mu, *ix, *iy, *ix_new, *iy_new);
					to_swap = (*dE <= 0 ? true : (gsl_rng_uniform(rng) < exp(- *dE)));
				}

				++N_flip_tries;

//				printf("swap_old = %d, dE = %lf, s = %d, s_new = %d\n", state[pos] != state[*ix_new * L + *iy_new],
//					   *dE, state[*ix * L + *iy], state[*ix_new * L + *iy_new]);
			}while(!to_swap);
		}

//		if(state[*ix * L + *iy] == state[*ix_new * L + *iy_new]){
//			printf("ERR:\n");
//			printf("to_swap: %d; s = %d, s_new = %d, \n", to_swap, state[*ix * L + *iy], state[*ix_new * L + *iy_new]);
//			printf("to_swap_old: %d; s = %d, s_new = %d, \n", state[*ix * L + *iy] != state[pos], state[*ix * L + *iy], state[pos]);
//			assert(getchar() != 'e');
//		}

		return N_flip_tries;
	}

	int flip_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *s_new, double *dE)
	/**
	 * Get the positions [*ix, *iy] of a spin to flip in a MC process
	 * @param ix - int*, the X index of the spin to be flipped (already decided)
	 * @param iy - int*, the Y index of the spin to be flipped (already decided)
	 * @param dE - the Energy difference necessary for the flip (E_flipped - E_current)
	 * @return - the number of tries it took to obtain a successful flip
	 */
	{
		int N_flip_tries = 0;
		int L2 = L * L;
		int total_range = L2 * (N_species - 1);
		int rnd, pos;
		do{
			rnd = gsl_rng_uniform_int(rng, total_range);
			pos = rnd % L2;
			*s_new = (state[pos] + 1 + rnd / L2) % N_species;
//			*s_new = 1 - state[*ix];
			*iy = pos % L;
			*ix = pos / L;

//			*iy = gsl_rng_uniform_int(rng, total_range);
//			*ix = (*iy) % L2;
//			*s_new = (state[*ix] + 1 + (*iy) / L2) % N_species;
////			*s_new = 1 - state[*ix];
//			*iy = *ix % L;
//			*ix = *ix / L;

//			*ix = gsl_rng_uniform_int(rng, L2);
//			*s_new = (state[*ix] + 1 + gsl_rng_uniform_int(rng, N_species - 1)) % N_species;
//			*iy = (*ix) % L;
//			*ix = *ix / L;

//			*ix = gsl_rng_uniform_int(rng, L);
//			*iy = gsl_rng_uniform_int(rng, L);
//			*s_new = (state[*ix * L + *iy] + 1 + gsl_rng_uniform_int(rng, N_species - 1)) % N_species;

			*dE = flip_mode_dE(state, L, e, mu, *ix, *iy, *s_new);
			++N_flip_tries;
		}while(!(*dE <= 0 ? 1 : (gsl_rng_uniform(rng) < exp(- *dE))));

		return N_flip_tries;
	}

	int get_max_CS(int *state, int L)
	{
		int L2 = L * L;

		int *cluster_element_inds = (int*) malloc(sizeof(int) * L2);
		int *cluster_sizes = (int*) malloc(sizeof(int) * L2);
		int *cluster_types = (int*) malloc(sizeof(int) * L2);
		int *is_checked = (int*) malloc(sizeof(int) * L2);

		int N_clusters = L2;

		clear_clusters(cluster_element_inds, cluster_sizes, &N_clusters);
		uncheck_state(is_checked, L2);
		cluster_state_C(state, L, cluster_element_inds, cluster_sizes, cluster_types, &N_clusters, is_checked);
		int biggest_CS = max(cluster_sizes, N_clusters);

		free(cluster_element_inds);
		free(cluster_sizes);
		free(cluster_types);
		free(is_checked);

		return biggest_CS;
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

		gen_mt19937 = new std::mt19937( std::random_device{}() );

		seed = my_seed;
		return 0;
	}

	void print_E(const double *E, long Nt, char prefix, char suffix)
    {
        if(prefix > 0) printf("Es:  %c\n", prefix);
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

	void print_e_matrix(const double *e)
	{
		int ix, iy;
		printf("e:\n");
		for(ix = 0; ix < N_species; ++ix) {
			for(iy = 0; iy < N_species; ++iy){
				printf("%5lf ", e[ix * N_species + iy]);
			}
			printf("\n");
		}
	}

	void print_mu_vector(const double *mu)
	{
		printf("mu: ( ");
		for(int i = 0; i < N_species; ++i)
			printf("%5lf ", mu[i]);
		printf(")\n");
	}

	int is_neib_x(int L, int i, int j)
	{
		int d = abs(i - j);
		return std::min(d, L-d) <= 1;
	}

	int is_neib(int L, int ix1, int iy1, int ix2, int iy2, int mode=1)
	{
		if(mode == 1){
			/*
			 * visual neibrs:
			 * ***
			 * *o*
			 * ***
			 */
			return is_neib_x(L, ix1, ix2) && is_neib_x(L, iy1, iy2);
		}
	}

	int *copy_state(const int *state, int L2)
	{
		int state_size_in_bytes = L2 * sizeof (int);

		int *new_state = (int*) malloc(state_size_in_bytes);
		memcpy(new_state, state, state_size_in_bytes);
		return new_state;
	}

	void shift_state(int *state, int L, int dx, int dy)
	{
		int L2 = L*L;

		int *old_state = copy_state(state, L2);

		int i,j;
		for(i = 0; i < L; ++i){
			for(j = 0; j < L; ++j){
				state[i * L + j] = old_state[L * md(i - dx, L) + md(j - dy, L)];
			}
		}

		free(old_state);
	}

	void print_env2(const int *state, int L, int ix1, int iy1, int ix2, int iy2)
	{
		int L2 = L*L;
		int *state_to_print = copy_state(state, L2);

		if(abs(ix1 - ix2) > 1){
			shift_state(state_to_print, L, 1, 0);
			ix1 = md(ix1 + 1, L);
			ix2 = md(ix2 + 1, L);
		}
		if(abs(iy1 - iy2) > 1){
			shift_state(state_to_print, L, 0, 1);
			iy1 = md(iy1 + 1, L);
			iy2 = md(iy2 + 1, L);
		}

		if(ix2 < ix1) std::swap(ix1, ix2);
		if(iy2 < iy1) std::swap(iy1, iy2);
		int dx = ix2 - ix1;
		int dy = iy2 - iy1;

		assert((dx >= 0) && (dx <= 1));
		assert((dy >= 0) && (dy <= 1));

		int group1[2 * dim + 1];
		int group2[2 * dim + 1];

		get_spin_with_neibs(state_to_print, L, ix1, iy1, group1);
		get_spin_with_neibs(state_to_print, L, ix2, iy2, group2);

		if(dx > 0){
			printf("  %d  \n%d %d %d\n%d %d %d\n  %d  \n", group1[3], group1[4], group1[0], group1[2], group2[4], group2[0], group2[2], group2[1]);
		} else if(dy > 0){
			printf("  %d %d  \n%d %d %d %d\n  %d %d  \n", group1[3], group2[3], group1[4], group1[0], group2[0], group2[2], group1[1], group2[1]);
		}

	}

	int md(int i, int L){ return i >= 0 ? (i < L ? i : i - L) : (L + i); }   // i mod L for i \in [-L; 2L-1]
//	int mds(int i, int R){ return i < R ? (i > -R ? i : i + 2*R) : (i - 2*R); }
	int mds(int i, int R){ return md(i + R, 2 * R) - R; };
}

/*
 * //					if((s[L2-L] != s[L2-L + 1]) || (s[L2-L] != s[L2-L + L-1]) || (s[L2-L] != s[L2-L + L]) || (s[L2-L] != s[L2-L - L])){
					if((s[L2-3*L+2] != s[L2-3*L+2 + 1]) || (s[L2-3*L+2] != s[L2-3*L+2 - 1]) || (s[L2-3*L+2] != s[L2-3*L+2 + L]) || (s[L2-3*L+2] != s[L2-3*L+2 - L])){
						print_S(s, L, 't');
						print_env2(s, L, L-3,2, L-2,2);
						print_env2(s, L, L-3,2, L-3,3);
						print_env2(s, L, L-3,2, L-4, 2);
						print_env2(s, L, L-3,2, L-3,1);

						int *bfr = copy_state(s, L2);
						shift_state(bfr, L, 5, 0);
						print_S(bfr, L, 'b');
						print_env2(bfr, L, 2,2, 3,2);
						print_env2(bfr, L, 2,2, 2,3);
						print_env2(bfr, L, 2,2, 1, 2);
						print_env2(bfr, L, 2,2, 2,1);
						free(bfr);

						bfr = copy_state(s, L2);
						shift_state(bfr, L, 2, -2);
						print_S(bfr, L, '1');
						print_env2(bfr, L, L-1,0, 0,0);
						print_env2(bfr, L, L-1,0, L-1,1);
						print_env2(bfr, L, L-1,0, L-2, 0);
						print_env2(bfr, L, L-1,0, L-1,L-1);
						free(bfr);

						bfr = copy_state(s, L2);
						shift_state(bfr, L, 3, -2);
						print_S(bfr, L, '2');
						print_env2(bfr, L, 0,0, 1,0);
						print_env2(bfr, L, 0,0, 0,1);
						print_env2(bfr, L, 0,0, L-1, 0);
						print_env2(bfr, L, 0,0, 0,L-1);
						free(bfr);

						bfr = copy_state(s, L2);
						shift_state(bfr, L, 3, -3);
						print_S(bfr, L, '3');
						print_env2(bfr, L, 0,L-1, 1,L-1);
						print_env2(bfr, L, 0,L-1, 0,0);
						print_env2(bfr, L, 0,L-1, L-1, L-1);
						print_env2(bfr, L, 0,L-1, 0,L-2);
						free(bfr);

						bfr = copy_state(s, L2);
						shift_state(bfr, L, 2, -3);
						print_S(bfr, L, '4');
						print_env2(bfr, L, L-1,L-1, 0,L-1);
						print_env2(bfr, L, L-1,L-1, L-1,0);
						print_env2(bfr, L, L-1,L-1, L-2, L-1);
						print_env2(bfr, L, L-1,L-1, L-1,L-2);
						free(bfr);
						STP
					}

 */

/*
 * 		if(dx == 1){
			dE = (e[snew_ix + state[md(ix_new + 1, L) * L + iy_new]] +
				  e[snew_ix + state[ix_new * L + md(iy_new + 1, L)]] +
				  e[snew_ix + state[ix_new * L + md(iy_new - 1, L)]] +
				  e[s_ix + state[md(ix - 1, L) * L + iy]] +
				  e[s_ix + state[ix * L + md(iy + 1, L)]] +
				  e[s_ix + state[ix * L + md(iy - 1, L)]]) -
				 (e[s_ix + state[md(ix_new + 1, L) * L + iy_new]] +
				  e[s_ix + state[ix_new * L + md(iy_new + 1, L)]] +
				  e[s_ix + state[ix_new * L + md(iy_new - 1, L)]] +
				  e[snew_ix + state[md(ix - 1, L) * L + iy]] +
				  e[snew_ix + state[ix * L + md(iy + 1, L)]] +
				  e[snew_ix + state[ix * L + md(iy - 1, L)]]);
		} else if(dx == -1) {
			dE = (e[snew_ix + state[md(ix_new - 1, L) * L + iy_new]] +
				  e[snew_ix + state[ix_new * L + md(iy_new + 1, L)]] +
				  e[snew_ix + state[ix_new * L + md(iy_new - 1, L)]] +
				  e[s_ix + state[md(ix + 1, L) * L + iy]] +
				  e[s_ix + state[ix * L + md(iy + 1, L)]] +
				  e[s_ix + state[ix * L + md(iy - 1, L)]]) -
				 (e[s_ix + state[md(ix_new - 1, L) * L + iy_new]] +
				  e[s_ix + state[ix_new * L + md(iy_new + 1, L)]] +
				  e[s_ix + state[ix_new * L + md(iy_new - 1, L)]] +
				  e[snew_ix + state[md(ix + 1, L) * L + iy]] +
				  e[snew_ix + state[ix * L + md(iy + 1, L)]] +
				  e[snew_ix + state[ix * L + md(iy - 1, L)]]);
		} else if(dy == 1){
			dE = (e[snew_ix + state[md(ix_new + 1, L) * L + iy_new]] +
				  e[snew_ix + state[md(ix_new - 1, L) * L + iy_new]] +
				  e[snew_ix + state[ix_new * L + md(iy_new + 1, L)]] +
				  e[s_ix + state[md(ix + 1, L) * L + iy]] +
				  e[s_ix + state[md(ix - 1, L) * L + iy]] +
				  e[s_ix + state[ix * L + md(iy - 1, L)]]) -
				 (e[s_ix + state[md(ix_new + 1, L) * L + iy_new]] +
				  e[s_ix + state[md(ix_new - 1, L) * L + iy_new]] +
				  e[s_ix + state[ix_new * L + md(iy_new + 1, L)]] +
				  e[snew_ix + state[md(ix + 1, L) * L + iy]] +
				  e[snew_ix + state[md(ix - 1, L) * L + iy]] +
				  e[snew_ix + state[ix * L + md(iy - 1, L)]]);
		} else if(dy == -1){
			dE = (e[snew_ix + state[md(ix_new + 1, L) * L + iy_new]] +
				  e[snew_ix + state[md(ix_new - 1, L) * L + iy_new]] +
				  e[snew_ix + state[ix_new * L + md(iy_new - 1, L)]] +
				  e[s_ix + state[md(ix + 1, L) * L + iy]] +
				  e[s_ix + state[md(ix - 1, L) * L + iy]] +
				  e[s_ix + state[ix * L + md(iy + 1, L)]]) -
				 (e[s_ix + state[md(ix_new + 1, L) * L + iy_new]] +
				  e[s_ix + state[md(ix_new - 1, L) * L + iy_new]] +
				  e[s_ix + state[ix_new * L + md(iy_new - 1, L)]] +
				  e[snew_ix + state[md(ix + 1, L) * L + iy]] +
				  e[snew_ix + state[md(ix - 1, L) * L + iy]] +
				  e[snew_ix + state[ix * L + md(iy + 1, L)]]);

 */