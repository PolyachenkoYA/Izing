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

namespace py = pybind11;

#include "Izing.h"

void test_my(int k, py::array_t<int> *_Nt, py::array_t<double> *_probs, py::array_t<double> *_d_probs, int l)
// A function for DEBUG purposes
{
	printf("checking step %d\n", k);
	py::buffer_info Nt_info = (*_Nt).request();
	py::buffer_info probs_info = (*_probs).request();
	py::buffer_info d_probs_info = (*_d_probs).request();
	printf("n%d: %d\n", k, Nt_info.shape[0]);
	printf("p%d: %d\n", k, probs_info.shape[0]);
	printf("d%d: %d\n", k, d_probs_info.shape[0]);
}

py::int_ init_rand(int my_seed)
{
	Izing::init_rand_C(my_seed);
	return 0;
}

py::int_ get_seed()
{
	return Izing::seed;
}

py::int_ set_verbose(int new_verbose)
{
	Izing::verbose_dafault = new_verbose;
	return 0;
}

py::tuple run_bruteforce(int L, double Temp, double h, int Nt_max, std::optional<int> _verbose)
/**
 *
 * @param L - the side-size of the lattice
 * @param Temp - temperature of the system; units=J, so it's actually T/J
 * @param h - magnetic-field-induced multiplier; unit=J, so it's h/J
 * @param Nt_max - for many succesful MC steps I want
 * @param _verbose - int number >= 0 or py::none(), shows how load is the process; If it's None (py::none()), then the default state 'verbose' is used
 * @return :
 * 	E, M - double arrays [Nt], Energy and Magnetization data from all the simulations
 */
{
	int i, j;
	int L2 = L*L;

// -------------- check input ----------------
	assert(L > 0);
	assert(Temp > 0);
	int verbose = (_verbose.has_value() ? _verbose.value() : Izing::verbose_dafault);

// ----------------- create return objects --------------
	int Nt = 0;
	int M_arr_len = 128;   // the initial value that will be doubling when necessary
	py::array_t<int> state = py::array_t<int>(L2);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets
	py::buffer_info state_info = state.request();
	int *state_ptr = static_cast<int *>(state_info.ptr);

	double *_E;
	int *_M;
	_E = (double*) malloc(sizeof(double) * M_arr_len);
	_M = (int*) malloc(sizeof(int) * M_arr_len);

	if(verbose){
		printf("using: L=%d  T=%lf  h=%lf  verbose=%d\n", L, Temp, h, verbose);
	}

	Izing::get_init_states_C(L, Temp, h, 1, L2, state_ptr, verbose, 0); // allocate all spins = -1

	Izing::run_state(state_ptr, L, Temp, h, -L2-1, L2+1, &_E, &_M, &Nt, &M_arr_len, true, verbose, Nt_max, nullptr);

	int N_last_elements_to_print = std::min(Nt, 10);
	if(verbose >= 2){
		printf("Brute-force core done\n");
		Izing::print_E(&(_E[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
		Izing::print_M(&(_M[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
	}

	py::array_t<double> E = py::array_t<double>(Nt);
	py::buffer_info E_info = E.request();
	double *E_ptr = static_cast<double *>(E_info.ptr);
	memcpy(E_ptr, _E, sizeof(double) * Nt);
	free(_E);

	py::array_t<int> M = py::array_t<double>(Nt);
	py::buffer_info M_info = M.request();
	int *M_ptr = static_cast<int *>(M_info.ptr);
	memcpy(M_ptr, _M, sizeof(int) * Nt);
	free(_M);

	if(verbose >= 2){
		printf("internal memory for EM freed\n");
		Izing::print_E(&(E_ptr[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'P');
		Izing::print_M(&(M_ptr[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'P');
		printf("exiting py::run_bruteforce\n");
	}

	return py::make_tuple(E, M);
}

// int run_FFS_C(double *flux0, double *d_flux0, int L, double Temp, double h, int *states, int *N_init_states, int *Nt, int *M_arr_len, int *M_interfaces, int N_M_interfaces, double *probs, double *d_probs, double **E, int **M, int to_remember_EM, int verbose)
py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> M_interfaces, int to_get_EM, int init_gen_mode, std::optional<int> _verbose)
/**
 *
 * @param L - the side-size of the lattice
 * @param Temp - temperature of the system; units=J, so it's actually T/J
 * @param h - magnetic-field-induced multiplier; unit=J, so it's h/J
 * @param N_init_states - array of ints [N_M_interfaces+2], how many states do I want on each interface
 * @param M_interfaces - array of ints [N_M_interfaces+2], contains values of M for interfaces. The map is [-L2; M_0](; M_1](...](; M_n-1](; L2]
 * @param to_get_EM - T/F, determines whether the E and M evolution if stored
 * @param init_gen_mode - int, the way how the states in A are generated to be then simulated towards the M_0 interface
 * @param _verbose - int number >= 0 or py::none(), shows how load is the process; If it's None (py::none()), then the default state 'verbose' is used
 * @return :
 * 	flux0, d_flux0 - the flux from A to M_0
 * 	Nt - array in ints [N_M_interfaces + 1], the number of MC-steps between each pair of interfaces
 * 		Nt[interface_ID]
 * 	M_arr_len - memory allocated for storing E and M data
 * 	probs, d_probs - array of ints [N_M_interfaces - 1], probs[i] - probability to go from i to i+1
 * 		probs[interface_ID]
 * 	states - all the states in a 1D array. Mapping is the following:
 * 		states[0               ][0...L2-1] U states[1               ][0...L2-1] U ... U states[N_init_states[0]                 -1][0...L2-1]   U  ...\
 * 		states[N_init_states[0]][0...L2-1] U states[N_init_states[0]][0...L2-1] U ... U states[N_init_states[1]+N_init_states[0]-1][0...L2-1]   U  ...\
 * 		states[N_init_states[1]+N_init_states[0]][0...L2-1] U   ...\
 * 		... \
 * 		... states[sum_{k=0..N_M_interfaces}(N_init_states[k])][0...L2-1]
 * 	E, M - Energy and Magnetization data from all the simulations. Mapping - indices changes from last to first:
 * 		M[interface_ID][state_ID][time_ID]
 * 		Nt[interface_ID][state_ID] is used here
 * 		M[0][0][0...Nt[0][0]] U M[0][1][0...Nt[0][1]] U ...
 */
{
	int i, j;
	int L2 = L*L;
	int state_size_in_bytes = L2 * sizeof(int);

// -------------- check input ----------------
	assert(L > 0);
	assert(Temp > 0);
	py::buffer_info M_interfaces_info = M_interfaces.request();
	py::buffer_info N_init_states_info = N_init_states.request();
	int *M_interfaces_ptr = static_cast<int *>(M_interfaces_info.ptr);
	int *N_init_states_ptr = static_cast<int *>(N_init_states_info.ptr);
	assert(M_interfaces_info.ndim == 1);
	assert(N_init_states_info.ndim == 1);

	int N_M_interfaces = M_interfaces_info.shape[0] - 2;
	assert(N_M_interfaces + 2 == N_init_states_info.shape[0]);
	for(i = 0; i <= N_M_interfaces; ++i) {
		assert(M_interfaces_ptr[i+1] > M_interfaces_ptr[i]);
		assert((M_interfaces_ptr[i+1] - M_interfaces_ptr[1]) % 2 == 0);   // M_step = 2, so there must be integer number of M_steps between all the M-s on interfaces
	}
	int verbose = (_verbose.has_value() ? _verbose.value() : Izing::verbose_dafault);

// ----------------- create return objects --------------
	double flux0, d_flux0;
	int M_arr_len = 128;   // the initial value that will be doubling when necessary
// [(-L2)---M_0](---M_1](---...---M_n-2](---M_n-1](---L2]
//        A       1       2 ...n-1       n-1        B
//        0       1       2 ...n-1       n-1       n
	py::array_t<int> Nt = py::array_t<int>(N_M_interfaces + 1);
	py::array_t<double> probs = py::array_t<double>(N_M_interfaces + 1);
	py::array_t<double> d_probs = py::array_t<double>(N_M_interfaces + 1);
	py::buffer_info Nt_info = Nt.request();
	py::buffer_info probs_info = probs.request();
	py::buffer_info d_probs_info = d_probs.request();
	int *Nt_ptr = static_cast<int *>(Nt_info.ptr);
	double *probs_ptr = static_cast<double *>(probs_info.ptr);
	double *d_probs_ptr = static_cast<double *>(d_probs_info.ptr);

	int N_states_total = 0;
	for(i = 0; i < N_M_interfaces + 2; ++i) {
		N_states_total += N_init_states_ptr[i];
	}
	py::array_t<int> states = py::array_t<int>(N_states_total * L2);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets
	py::buffer_info states_info = states.request();
	int *states_ptr = static_cast<int *>(states_info.ptr);

    double *_E;
    int *_M;
    if(to_get_EM){
        _E = (double*) malloc(sizeof(double) * M_arr_len);
        _M = (int*) malloc(sizeof(int) * M_arr_len);
    }

	if(verbose){
		printf("using: L=%d  T=%lf  h=%lf  EM=%d  gm=%d  v=%d\n", L, Temp, h, to_get_EM, init_gen_mode, verbose);
		for(i = 1; i <= N_M_interfaces; ++i){
			printf("%d ", M_interfaces_ptr[i]);
		}
		printf("\n");
	}

	Izing::run_FFS_C(&flux0, &d_flux0, L, Temp, h, states_ptr, N_init_states_ptr,
					 Nt_ptr, &M_arr_len, M_interfaces_ptr, N_M_interfaces,
					 probs_ptr, d_probs_ptr, &_E, &_M, to_get_EM, verbose, init_gen_mode);

	if(verbose >= 2){
		printf("FFS core done\nNt: ");
	}
	int Nt_total = 0;
	for(i = 0; i < N_M_interfaces + 1; ++i) {
		if(verbose >= 2){
			printf("%d ", Nt_ptr[i]);
		}
		Nt_total += Nt_ptr[i];
	}
	int N_last_elements_to_print = 10;
	if(verbose >= 2){
		Izing::print_E(&(_E[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'F');
		Izing::print_M(&(_M[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'F');
	}

	py::array_t<double> E;
    py::array_t<int> M;
    if(to_get_EM){
		if(verbose >= 2){
			printf("allocating EM, Nt_total = %d\n", Nt_total);
		}
        E = py::array_t<double>(Nt_total);
        M = py::array_t<double>(Nt_total);
        py::buffer_info E_info = E.request();
        py::buffer_info M_info = M.request();
        double *E_ptr = static_cast<double *>(E_info.ptr);
		int *M_ptr = static_cast<int *>(M_info.ptr);

		if(verbose >= 2){
			printf("numpy arrays created\n", Nt_total);
		}

        memcpy(E_ptr, _E, sizeof(double) * Nt_total);

		if(verbose >= 2){
			printf("E copied\n", Nt_total);
		}

        memcpy(M_ptr, _M, sizeof(int) * Nt_total);

		if(verbose >= 2){
			printf("data copied\n", Nt_total);
		}

        free(_E);
        free(_M);

		if(verbose >= 2){
			printf("internal memory for EM freed\n");
			Izing::print_E(E_ptr, Nt_total < 10 ? Nt_total : 10, 'P');
			Izing::print_M(M_ptr, Nt_total < 10 ? Nt_total : 10, 'P');
		}
    }

	if(verbose >= 2){
		printf("exiting py::run_FFS\n");
	}
    return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M);
}

namespace Izing
{
    gsl_rng *rng;
    int seed;
    int verbose_dafault;

	int run_FFS_C(double *flux0, double *d_flux0, int L, double Temp, double h, int *states, int *N_init_states, int *Nt,
				  int *M_arr_len, int *M_interfaces, int N_M_interfaces, double *probs, double *d_probs, double **E, int **M,
				  int to_remember_EM, int verbose, int init_gen_mode)
	/**
	 *
	 * @param flux0 - see run_FFS description (return section)
	 * @param d_flux0 - run_FFS (return section)
	 * @param L - run_FFS
	 * @param Temp - run_FFS
	 * @param h - run_FFS
	 * @param states - run_FFS (return section)
	 * @param N_init_states - run_FFS
	 * @param Nt - run_FFS (return section)
	 * @param M_arr_len - see 'double step_process(...)' description
	 * @param M_interfaces - run_FFS
	 * @param N_M_interfaces - int = len(M_interfaces) - 2, the number of non-trivial (all expect -L2-1 and +L2) interfaces
	 * @param probs - run_FFS (return section)
	 * @param d_probs - run_FFS (return section)
	 * @param E - run_FFS (return section)
	 * @param M - run_FFS (return section)
	 * @param to_remember_EM - run_FFS
	 * @param verbose - run_FFS
	 * @param init_gen_mode - run_FFS
	 * @return - the Error code (none implemented yet)
	 */
	{
		int i, j;
		int Nt_total_prev;
// get the initial states; they should be sampled from the distribution in [-L^2; M_0), but they are all set to have M == -L^2 because then they all will fall into the local optimum and almost forget the initial state, so it's almost equivalent to sampling from the proper distribution if 'F(M_0) - F_min >~ T'
		get_init_states_C(L, Temp, h, N_init_states[0], M_interfaces[1], states, verbose, init_gen_mode); // allocate all spins = -1
		if(verbose){
			printf("Init states generated with mode %d\n", init_gen_mode);
		}

		int N_states_analyzed = 0;
		int Nt_total = 0;
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;

		for(i = 0; i <= N_M_interfaces; ++i){
			Nt_total_prev = Nt_total;
			probs[i] = process_step(&(states[L2 * N_states_analyzed]),
									i < N_M_interfaces ? &(states[L2 * (N_states_analyzed + N_init_states[i])]) : nullptr,
									E, M, &Nt_total, M_arr_len, N_init_states[i], N_init_states[i+1],
									L, Temp, h, M_interfaces[i], M_interfaces[i+1],
									i < N_M_interfaces, to_remember_EM, verbose);
			//d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] / probs[i]));
			d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] * (1 - probs[i])));

			N_states_analyzed += N_init_states[i];

			Nt[i] = Nt_total - Nt_total_prev;
			if(i == 0){
				// we know that 'probs[0] == 1' because M_0 = -L2-1 for run[0]. Thus we can compute the flux
				*flux0 = (double)N_init_states[0] / Nt[0];
				*d_flux0 = *flux0 / sqrt(Nt[0]);   // TODO: use 'Nt/memory_time' instead of 'Nt'
			}

			if(verbose){
				if(i == 0){
					printf("flux0 = (%e +- %e) 1/step\n", *flux0, *d_flux0);
				} else {
					printf("-ln(p_%d) = (%lf +- %lf)\n", i, -log(probs[i]), d_probs[i] / probs[i]);   // this assumes p<<1
				}
				printf("Nt[%d] = %d; Nt_total = %d\n", i, Nt[i], Nt_total);
				if(verbose >= 2){
					if(i < N_M_interfaces){
						printf("\nstate[%d] beginning: ", i);
						for(j = 0; j < (Nt[i] > 10 ? 10 : Nt[i]); ++j)  printf("%d ", states[L2 * N_states_analyzed - N_init_states[i] + j]);
					}
					printf("\n");
				}
			}
		}

		double ln_k_AB = log(*flux0 * 1);   // flux has units = 1/time; Here, time is in steps, so it's not a problem. But generally speaking it's not clear what time to use here.
		double d_ln_k_AB = Izing::sqr(*d_flux0 / *flux0);
		for(i = 1; i < N_M_interfaces; ++i){   // we don't need the last prob since it's a P from M=M_last to M=L2
			ln_k_AB += log(probs[i]);
			d_ln_k_AB += Izing::sqr(d_probs[i] / probs[i]);   // this assumes dp/p << 1,
		}
		d_ln_k_AB = sqrt(d_ln_k_AB);

		if(verbose){
			printf("-ln(k_AB * [1 step]) = (%lf +- %lf)\n", - ln_k_AB, d_ln_k_AB);
			if(verbose >= 2){
				int N_last_elements_to_print = 10;
				printf("Nt_total = %d\n", Nt_total);
				print_E(&((*E)[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'f');
				print_M(&((*M)[Nt_total - N_last_elements_to_print]), N_last_elements_to_print, 'f');
				printf("Nt: ");
				for(i = 0; i < N_M_interfaces + 1; ++i) printf("%d ", Nt[i]);
				printf("\n");
			}
		}

		return 0;
	}

	double process_step(int *init_states, int *next_states, double **E, int **M, int *Nt, int *M_arr_len,
						int N_init_states, int N_next_states, int L, double Temp, double h, int M_0, int M_next,
						int to_save_next_states, bool to_remember_EM, int verbose)
	/**
	 *
	 * @param init_states - are assumed to contain 'N_init_states * state_size_in_bytes' ints representing states to start simulations from
	 * @param next_states - are assumed to be allocated to have 'N_init_states * state_size_in_bytes' ints
	 * @param E - array of Energy values for all the runs, joined consequently; Is assumed to be preallocated with *M_arr_len of doubles
	 * @param M - array of Magnetic moment values for all the runs, joined consequently; Is assumed to be preallocated with *M_arr_len of doubles
	 * @param Nt - total number of simulation steps in this 'i -> i+1' part of the simulation
	 * @param M_arr_len - size allocated for M and E arrays
	 * @param N_init_states - the number of states with M==M_next to generate; the simulation is terminated when this number is reached
	 * @param L - the side-size of the lattice
	 * @param Temp - temperature of the system; units=J, so it's actually T/J
	 * @param h - magnetic-field-induced multiplier; unit=J, so it's h/J
	 * @param M_0 - the lower-border to stop the simulations at. If a simulation reaches this M==M_0, it's terminated and discarded
	 * @param M_next - the upper-border to stop the simulations at. If a simulation reaches this M==M_0, it's stored to be a part of a init_states set of states for the next FFS step
	 * @param to_save_next_states - whether to use *next_states to store states with M = M_next. It's used to disable saving for M = +L2 in the last step.
	 * @param to_remember_EM - T/F, determines whether the E and M evolution if stored
	 * @param verbose - T/F, shows the progress
	 * @return - the fraction of successful runs (the runs that reached M==M_next)
	 */
	{
		int i;
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;

		int N_succ = 0;
		int N_runs = 0;
		int *state_under_process = (int*) malloc(state_size_in_bytes);
		if(verbose){
			printf("doing step:(%d; %d]\n", M_0, M_next);
//			if(verbose >= 2){
//				printf("press any key to continue...\n");
//				getchar();
//			}
		}
		int init_state_to_process_ID;
		while(N_succ < N_next_states){
			init_state_to_process_ID = gsl_rng_uniform_int(rng, N_init_states);
			if(verbose >= 2){
				printf("state[%d] (id in set = %d):\n", N_succ, init_state_to_process_ID);
			}
			memcpy(state_under_process, &(init_states[init_state_to_process_ID * L2]), state_size_in_bytes);   // get a copy of the chosen init state
			if(run_state(state_under_process, L, Temp, h, M_0, M_next, E, M, Nt, M_arr_len, to_remember_EM, verbose)){   // run it until it reaches M_0 or M_next
				// Nt is not reinitialized to 0 and that's correct because it shows the total number of EM datapoints
				// the run reached M_next
				++N_succ;
				if(to_save_next_states) {
					memcpy(&(next_states[(N_succ - 1) * L2]), state_under_process, state_size_in_bytes);   // save the resulting system state for the next step
				}
				if(verbose) {
					if(verbose >= 2){
						printf("state %d saved for future\n", N_succ - 1);
						printf("%lf %%\n", (double)N_succ/N_next_states * 100);
					} else { // verbose == 1
						if(N_succ % (N_next_states / 1000 + 1) == 0){
							printf("%lf %%          \r", (double)N_succ/N_next_states * 100);
							fflush(stdout);
						}
					}
				}
			}
			++N_runs;
		}
		if(verbose >= 2) {
			printf("\n");
		}

		free(state_under_process);

		return (double)N_succ / N_runs;   // the probability P(i+1 | i) to go from i to i+1
	}

	int run_state(int *s, int L, double Temp, double h, int M_0, int M_next, double **E, int **M, int *Nt, int *M_arr_len,
				  bool to_remember_EM, int verbose, int Nt_max, int* states_to_save, int *N_states_saved, int N_states_to_save, int M_thr_save_state)
	/**
	 *
	 * @param s - the current state the system under simulation is in
	 * @param L - see run_FFS
	 * @param Temp - see run_FFS
	 * @param h - see run_FFS
	 * @param M_0 - the interface such that A = {state \in {all states} : M(state) < M_0}
	 * @param M_next - the interface the the simulation is currently tring to reach. The simulation is stopped when M = M_next. It's assumed M_0 < M_next
	 * @param E - see run_FFS
	 * @param M - see run_FFS
	 * @param Nt - see run_FFS
	 * @param M_arr_len - see process_step function description
	 * @param to_remember_EM - see run_FFS
	 * @param verbose - see run_FFS
	 * @param Nt_max - int, default -1; It's it's > 0, then the simulation is stopped on '*Nt >= Nt_max'
	 * @param states_to_save - int*, default nullptr; The pointer to the set of states to save during the run
	 * @param N_states_saved - int*, default nullptr; The number of states already saved in 'states_to_save'
	 * @param N_states_to_save - int, default -1; If it's >0, the simulation is stopped when 'N_states_saved >= N_states_to_save'
	 * @param M_thr_save_state - int, default 0; If(N_states_to_save > 0), states are saved when M == M_thr_save_state
	 * @return - the Error status (none implemented yet)
	 */
	{
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;
		int M_current = comp_M(s, L); // remember the 1st M;
		double E_current = comp_E(s, L, h); // remember the 1st energy;

		if((abs(M_current) > L2) || ((L2 - M_current) % 2 != 0)){   // check for sanity
			state_is_valid(s, L, 0, 'e');
			if(verbose){
				printf("This state has M = %d (L = %d, dM_step = 2) which is beyond possible physical boundaries, there is something wrong with the simulation\n", M_current, L);
			}
		}
		if(N_states_to_save > 0){
			assert(states_to_save);
		}

		if(verbose >= 2){
			printf("E=%lf, M=%d\n", E_current, M_current);
		}

		int ix, iy;
		double dE;

		double Emin = -(2 + abs(h)) * L2;
		double Emax = 2 * L2;
		double E_tolerance = 1e-3;   // J=1
		int Nt_for_numerical_error = int(1e13 * E_tolerance / L2);
		// the error accumulates, so we need to recompute form scratch time to time
//      |E| ~ L2 => |dE_numerical| ~ L2 * 1e-13 => |dE_num_total| ~ sqrt(Nt * L2 * 1e-13) << E_tolerance => Nt << 1e13 * E_tolerance / L2

		while(1){
			// ----------- choose which to flip -----------
//			if(verbose >= 3) printf("doing Nt=%d\n", *Nt);
			get_flip_point(s, L, h, Temp, &ix, &iy, &dE);
//			if(verbose >= 3) printf("flip done\n");

			// ----------- modify state -----------
			++(*Nt);
			s[ix*L + iy] *= -1;
			M_current += 2 * s[ix*L + iy];
//			E_current += dE;
			if(*Nt % Nt_for_numerical_error){
				E_current += dE;
			} else {
				double E_curent_real = comp_E(s, L, h);
				if(abs(E_current - E_curent_real) > E_tolerance){
					if(verbose >= 2){
						printf("E_current = %lf, dE = %lf, Nt = %d, E_real = %lf\n", E_current, dE, *Nt, E_curent_real);
						print_E(&((*E)[*Nt - 10]), 10);
						print_S(s, L, 'r');
						throw -1;
//						getchar();
					}
					E_current = E_curent_real;
				}
			}
//			if(verbose >= 3) printf("state modified\n");

			// ------------------ save EM ----------------
			if(to_remember_EM){
				if(*Nt >= *M_arr_len){ // double the size of the time-index
					*M_arr_len *= 2;
					*E = (double*) realloc (*E, sizeof(double) * *M_arr_len);
					*M = (int*) realloc (*M, sizeof(int) * *M_arr_len);
					assert(*E);
					assert(*M);
					if(verbose >= 2){
						printf("realloced to %d\n", *M_arr_len);
					}
				}

				(*E)[*Nt - 1] = E_current;
				(*M)[*Nt - 1] = M_current;

//				if((E_current < Emin * (1 + 1e-6)) || (E_current > Emax * (1 + 1e-6))){   // assuming Emin < 0, Emax > 0
//					printf("E_current = %lf, dE = %lf, Nt = %d, E = %lf\n", E_current, dE, *Nt, comp_E(s, L, h));
//					print_E(&((*E)[*Nt - 10]), 10);
//					print_S(s, L, 'r');
//					getchar();
//				}
			}
//			if(verbose >= 3) printf("done Nt=%d\n", *Nt-1);

			// ------------------- save the state if it's good (we don't want exit states) -----------------
			if(N_states_to_save > 0){
				if(M_current < M_thr_save_state){
					memcpy(&(states_to_save[*N_states_saved * L2]), s, state_size_in_bytes);
					++ *N_states_saved;
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
					if(verbose >= 2) printf("Reached desired Nt >= Nt_max (= %d)\n", Nt_max);
					return 1;
				}
			}
			if(M_current == M_0){
				if(verbose >= 2) printf("Fail run\n");
				return 0;   // failed = gone to the initial state A
			} else if(M_current == M_next){
				if(verbose >= 2) printf("Success run\n");
				return 1;   // succeeded = reached the interface 'M == M_next'
			}
		}
	}

	int get_init_states_C(int L, double Temp, double h, int N_init_states, int M_0, int *init_states, int verbose, int mode)
	/**
	 *
	 * @param L - see run_FFS
	 * @param Temp - see run_FFS
	 * @param h - see run_FFS
	 * @param N_init_states - see run_FFS
	 * @param M_0 - see 'run_state' description
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

		if(mode >= -1){
			// generate N_init_states states in A
			// Here they are identical, but I think it's better to generate them accordingly to equilibrium distribution in A
			for(i = 0; i < N_init_states; ++i){
				generate_state(&(init_states[i * L2]), L, mode);
			}
		} else if(mode == -2){
			generate_state(init_states, L, 0);
			int N_states_done = 1;
			int Nt = 0;
			int restart_state_ID;

			while(N_states_done < N_init_states){
				restart_state_ID = gsl_rng_uniform_int(rng, N_states_done);
				if(verbose >= 2){
					printf("generated %d states, restarting from state[%d]", N_states_done, restart_state_ID);
				}
				run_state(&(init_states[L2 * restart_state_ID]), L, Temp, h, -L2-1, L2 % 2, nullptr, nullptr,
						  &Nt,nullptr, false, verbose, -1, init_states,
						  &N_states_done, N_init_states, M_0);
			}
		}

		return 0;
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

	int comp_M(const int *s, int L)
	/**
	 * Computes the Magnetization of the state 's' of the linear size 'L'
	 * @param s - the state to analyze
	 * @param L - the linear size of the lattice
	 * @return - the value of Magnetization of the state 's'
	 */
	{
		int i, j;
		int _M = 0;
		int L2 = L*L;
		for(i = 0; i < L2; ++i) _M += s[i];
		return _M;
	}

	double comp_E(const int* s, int L, double h)
	/**
	 * Computes the Energy of the state 's' of the linear size 'L', immersed in the 'h' magnetic field;
	 * E = E/J = - \sum_{i, <j>}(s_i * s_j) - h * sum_{i}(s_i)
	 * @param s - the state to analyze
	 * @param L - the linear size of the lattice
	 * @return - the value of Energy of the state 's'
	 */
	{
		int i, j;
		double _E = 0;
		for(i = 0; i < L-1; ++i){
			for(j = 0; j < L-1; ++j){
				_E += s[i*L + j] * (s[(i+1)*L + j] + s[i*L + (j+1)]);
			}
			_E += s[i*L + (L-1)] * (s[(i+1)*L + (L-1)] + s[i*L + 0]);
		}
		for(j = 0; j < L-1; ++j){
			_E += s[(L-1)*L + j] * (s[0*L + j] + s[(L-1)*L + (j+1)]);
		}
		_E += s[(L-1)*L + (L-1)] * (s[0*L + (L-1)] + s[(L-1)*L + 0]);

		int _M = comp_M(s, L);
		_E *= -1;   // J > 0 -> {++ > +-} -> we need to *(-1) because we search for a minimum
		// energy is measured in [E]=J, so J==1, or we store E/J value

		return - h * _M + _E;
	}

	int generate_state(int *s, int L, int mode)
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
			for(i = 0; i < L2; ++i) s[i] = -1;
			if(mode > 0){
				int N_down_spins = mode;
				assert(N_down_spins <= L2);

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

					s[swap_var] = 1;
				}
				free(indices_to_flip);
			}
		} else if(mode == -1){   // random
			for(i = 0; i < L2; ++i) s[i] = gsl_rng_uniform_int(rng, 2) * 2 - 1;
		}

		return 0;
	}

	double get_dE(int *s, int L, double h, int ix, int iy)
	/**
	 * Computes the energy difference '(E_future_after_the_flip - E_current)'.
	 * 1. "Now" the s[ix, iy] = s, it will be "= -s", so \sum_{neib} difference d_sum = {(-s) - s = -2s} * {all the neighbours}.
	 * 		But E = -\sum_{neib}, so d_E = -d_sum = 2s * {all the neighbours}
	 * 2. d_\sum(s_i) = (-s) - s = -2s; E = -h * \sum(s_i) => d_E = -h * d_sum(s_i) = 2s * h
	 * 3. So, it total, d_E = 2s(h + s+neib)
	 * @param s - the current state (before the flip)
	 * @param L - linear size of the state
	 * @param h - magnetic field
	 * @param ix - the X index os the spin considered for a flip
	 * @param iy - the Y index os the spin considered for a flip
	 * @return - the values of a potential Energy change
	 */
// the difference between the current energy and energy with s[ix][iy] flipped
	{
		return 2 * s[ix*L + iy] * (s[md(ix + 1, L)*L + iy]
								   + s[ix*L + md(iy + 1, L)]
								   + s[md(ix - 1, L)*L + iy]
								   + s[ix*L + md(iy - 1, L)] + h);
		// units [E] = J, so spins are just 's', not s*J
	}

	int get_flip_point(int *s, int L, double h, double Temp, int *ix, int *iy, double *dE)
	/**
	 * Get the positions [*ix, *iy] of a spin to flip in a MC process
	 * @param s - the state
	 * @param L - the size
	 * @param h - the magnetic field
	 * @param Temp - the Temperature
	 * @param ix - int*, the X index of the spin to be flipped (already decided)
	 * @param iy - int*, the Y index of the spin to be flipped (already decided)
	 * @param dE - the Energy difference necessary for the flip (E_flipped - E_current)
	 * @return - the Error code
	 */
	{
		do{
			*ix = gsl_rng_uniform_int(rng, L);
			*iy = gsl_rng_uniform_int(rng, L);

			*dE = get_dE(s, L, h, *ix, *iy);
		}while(!(*dE <= 0 ? 1 : (gsl_rng_uniform(rng) < exp(- *dE / Temp) ? 1 : 0)));

		return 0;
	}

    void print_E(const double *E, int Nt, char prefix, char suffix)
    {
        if(prefix > 0) printf("Es: %c\n", prefix);
        for(int i = 0; i < Nt; ++i) printf("%lf ", E[i]);
        if(suffix > 0) printf("%c", suffix);
    }

	void print_M(const int *M, int Nt, char prefix, char suffix)
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

	int md(int i, int L){ return i >= 0 ? (i < L ? i : 0) : (L-1); }   // i mod L for i \in [-1; L]
}

