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

py::tuple run_bruteforce(int L, double Temp, double h, int Nt_max, std::optional<int> _verbose)
/**
 *
 * @param L - the side-size of the lattice
 * @param Temp - temperature of the system; units=J, so it's actually T/J
 * @param h - magnetic-field-induced multiplier; unit=J, so it's h/J
 * @param Nt - for many succesful MC steps I want
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

	double **_E;
	int **_M;
	_E = (double**) malloc(sizeof(double*) * 1);
	*_E = (double*) malloc(sizeof(double) * M_arr_len);
	_M = (int**) malloc(sizeof(int*) * 1);
	*_M = (int*) malloc(sizeof(int) * M_arr_len);

	if(verbose){
		printf("using: L=%d  T=%lf  h=%lf  verbose=%d\n", L, Temp, h, verbose);
	}

	Izing::get_init_states_C(L, Temp, h, 1, state_ptr, _E, _M, &Nt, 1, verbose, 0); // allocate all spins = -1

	Izing::run_state(state_ptr, L, Temp, h, -L2-1, L2+1, _E, _M, &Nt, &M_arr_len, 1, verbose, Nt_max);

	if(verbose >= 2){
		printf("Brute-force core done\n");
	}

	py::array_t<double> E = py::array_t<double>(Nt);
	py::buffer_info E_info = E.request();
	double *E_ptr = static_cast<double *>(E_info.ptr);
	memcpy(E_ptr, *_E, sizeof(double) * Nt);
	free(*_E);
	free(_E);

	py::array_t<int> M = py::array_t<double>(Nt);
	py::buffer_info M_info = M.request();
	int *M_ptr = static_cast<int *>(M_info.ptr);
	memcpy(M_ptr, *_M, sizeof(int) * Nt);
	free(*_M);
	free(_M);

	if(verbose >= 2){
		printf("internal memory for EM freed\n");
		Izing::print_E(E_ptr, Nt < 10 ? Nt : 10, 'P');
		printf("exiting py::run_bruteforce\n");
	}

	return py::make_tuple(E, M);
}

// int run_FFS_C(double *flux0, double *d_flux0, int L, double Temp, double h, int *states, int *N_init_states, int *Nt, int *M_arr_len, int *M_interfaces, int N_M_interfaces, double *probs, double *d_probs, double **E, int **M, int to_remember_EM, int verbose)
py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> M_interfaces, int to_get_EM, std::optional<int> _verbose)
/**
 *
 * @param L - the side-size of the lattice
 * @param Temp - temperature of the system; units=J, so it's actually T/J
 * @param h - magnetic-field-induced multiplier; unit=J, so it's h/J
 * @param N_init_states - array of ints [N_M_interfaces+2], how many states do I want on each interface
 * @param M_interfaces - array of ints [N_M_interfaces+2], contains values of M for interfaces. The map is [-L2; M_0](; M_1](...](; M_n-1](; L2]
 * @param to_get_EM - T/F, determines whether the E and M evolution if stored
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

    double **_E;
    int **_M;
    if(to_get_EM){
        _E = (double**) malloc(sizeof(double*) * 1);
        *_E = (double*) malloc(sizeof(double) * M_arr_len);
        _M = (int**) malloc(sizeof(int*) * 1);
        *_M = (int*) malloc(sizeof(int) * M_arr_len);
    }

	if(verbose){
		printf("using: L=%d  T=%lf  h=%lf  EM=%d  v=%d\n", L, Temp, h, to_get_EM, verbose);
		for(i = 1; i <= N_M_interfaces; ++i){
			printf("%d ", M_interfaces_ptr[i]);
		}
		printf("\n");
	}

	Izing::run_FFS_C(&flux0, &d_flux0, L, Temp, h, states_ptr, N_init_states_ptr,
					 Nt_ptr, &M_arr_len, M_interfaces_ptr, N_M_interfaces,
					 probs_ptr, d_probs_ptr, _E, _M, to_get_EM, verbose);

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
	if(verbose >= 2){
		printf("\n");
	}

	py::array_t<double> E;
    py::array_t<int> M;
    if(to_get_EM){
        E = py::array_t<double>(Nt_total);
        M = py::array_t<double>(Nt_total);
        py::buffer_info E_info = E.request();
        py::buffer_info M_info = M.request();
        double *E_ptr = static_cast<double *>(E_info.ptr);
        int *M_ptr = static_cast<int *>(M_info.ptr);
        memcpy(E_ptr, *_E, sizeof(double) * Nt_total);
        memcpy(M_ptr, *_M, sizeof(int) * Nt_total);

        free(*_E);
        free(_E);
        free(*_M);
        free(_M);

		if(verbose >= 2){
			printf("internal memory for EM freed\n");
			if(to_get_EM){
				Izing::print_E(E_ptr, Nt_total < 10 ? Nt_total : 10, 'P');
			}
		}
    }

	if(verbose >= 2){
		printf("exiting py::run_FFS\n");
	}
    return py::make_tuple(states, probs, d_probs, Nt, flux0, d_flux0, E, M);
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

namespace Izing
{
    gsl_rng *rng;
    int seed;
    int verbose_dafault;

	int state_is_valid(const int *s, int L, int k=0, char prefix=0)
	{
		for(int i = 0; i < L*L; ++i) if(abs(s[i]) != 1) {
				printf("%d\n", k);
				print_S(s, L, prefix);
				return 0;
		}
		return 1;
	}

	int init_rand_C(int my_seed)
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
    {
        int i, j;
        int _M = 0;
		int L2 = L*L;
		for(i = 0; i < L2; ++i) _M += s[i];
        return _M;
    }

    double comp_E(const int* s, int L, double h)
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

        return h * _M + _E;
    }

    int generate_state(int *s, int L, gsl_rng *rng, int mode)
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
		} else if(mode == -1){ // random
            for(i = 0; i < L2; ++i) s[i] = gsl_rng_uniform_int(rng, 2) * 2 - 1;
        }

        return 0;
    }

    int md(int i, int L)
    {
        return i >= 0 ? (i < L ? i : 0) : (L-1);   // i mod L for i \in [-1; L]
    }

    double get_dE(int *s, int L, double h, int ix, int iy)
// the difference between the current energy and energy with s[ix][iy] flipped
    {
        return 2 * s[ix*L + iy] * (s[md(ix + 1, L)*L + iy]
                                   + s[ix*L + md(iy + 1, L)]
                                   + s[md(ix - 1, L)*L + iy]
                                   + s[ix*L + md(iy - 1, L)] + h);
        // units [E] = J, so spins are just 's', not s*J
    }

    int get_flip_point(int *s, int L, double h, double Temp, int *ix, int *iy, double *dE, gsl_rng *rng)
    {
        do{
            *ix = gsl_rng_uniform_int(rng, L);
            *iy = gsl_rng_uniform_int(rng, L);

            *dE = get_dE(s, L, h, *ix, *iy);
        }while(!(*dE <= 0 ? 1 : (gsl_rng_uniform(rng) < exp(- *dE / Temp) ? 1 : 0)));

        return 0;
    }

	int run_FFS_C(double *flux0, double *d_flux0, int L, double Temp, double h, int *states, int *N_init_states, int *Nt,
				  int *M_arr_len, int *M_interfaces, int N_M_interfaces, double *probs, double *d_probs, double **E, int **M,
				  int to_remember_EM, int verbose)
	{
		int i, j;
		int Nt_total = 0;
// get the initial states; they should be sampled from the distribution in [-L^2; M_0], but they are all set to have M == -L^2 because then they all will fall into the local optimum and almost forget the initial state, so it's almost equivalent to sampling from the proper distribution if 'F(M_0) - F_min >~ T'
		get_init_states_C(L, Temp, h, N_init_states[0], states, E, M, &Nt_total, to_remember_EM, verbose, 0); // allocate all spins = -1
		if(verbose){
			printf("Init states generated; Nt = %d\n", Nt_total);
		}

		int N_states_analyzed = 0;
		int L2 = L*L;
		int state_size_in_bytes = sizeof(int) * L2;

		for(i = 0; i <= N_M_interfaces; ++i){
			probs[i] = process_step(&(states[L2 * N_states_analyzed]),
									&(states[i == N_M_interfaces ? 0 : L2 * (N_states_analyzed + N_init_states[i])]),
									E, M, &Nt_total, M_arr_len, N_init_states[i],
									N_init_states[i+1], L, Temp, h, M_interfaces[i], M_interfaces[i+1],
									i < N_M_interfaces, to_remember_EM, verbose);
			//d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] / probs[i]));
			d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] * (1 - probs[i])));

			N_states_analyzed += N_init_states[i];

			if(i == 0){
				// we know that 'probs[0] == 1' because M_0 = -L2-1 for run[0]. Thus we can compute the flux
				Nt[0] = Nt_total;
				*flux0 = (double)N_init_states[0] / Nt[0];
				*d_flux0 = *flux0 / sqrt(Nt[0]);   // TODO: use 'Nt/memory_time' instead of 'Nt'
			} else {
				Nt[i] = Nt_total - Nt[i - 1];
			}

			if(verbose){
				if(i == 0){
					printf("flux0 = (%e +- %e) 1/step\n", *flux0, *d_flux0);
				} else {
					printf("-ln(p_%d) = (%lf +- %lf)\n", i, -log(probs[i]), d_probs[i] / probs[i]);   // this assumes p<<1
				}
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
			d_ln_k_AB += Izing::sqr(d_probs[i] / probs[i]);   // this assumes p<<1
		}
		d_ln_k_AB = sqrt(d_ln_k_AB);

		if(verbose){
			printf("-ln(k_AB * [1 step]) = (%lf +- %lf)\n", - ln_k_AB, d_ln_k_AB);
			if(verbose >= 2){

			}
		}

		return 0;
	}

	int get_init_states_C(int L, double Temp, double h, int N_init_states, int *init_states, double **E, int **M, int *Nt, bool to_remember_EM, int verbose, int mode)
	{
		int i;
		int L2 = L*L;
//		int state_size_in_bytes = sizeof(int) * L2;

		// generate N_init_states states in A
		// Here they are identical, but I think it's better to generate them accordingly to equilibrium distribution in A
		for(i = 0; i < N_init_states; ++i){
			generate_state(&(init_states[i * L2]), L, rng, mode);
		}

		*Nt += 1;
		// record initial state (all DOWN) E and M since run_state changes it before recording
		if(to_remember_EM){
			(*M)[*Nt - 1] = comp_M(init_states, L);
			(*E)[*Nt - 1] = comp_E(init_states, L, h);
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
			if(verbose >= 2){
				printf("press any key to continue...\n");
				getchar();
			}
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

		return (double)N_succ / N_runs;   // the probability P(i+i | i) to go from i to i+1
	}

	int run_state(int *s, int L, double Temp, double h, int M_0, int M_next, double **E, int **M, int *Nt, int *M_arr_len, bool to_remember_EM, int verbose, int Nt_max)
	{
		int L2 = L*L;
		int M_current;
		double E_current;
		M_current = comp_M(s, L); // remember the 1st M
		E_current = comp_E(s, L, h); // remember the 1st energy
		if(verbose >= 2){
			printf("E=%lf, M=%d\n", E_current, M_current);
			if(abs(M_current) > L2){
				printf("state: %d\n", state_is_valid(s, L, 0, 'r'));
				getchar();
			}
		}

		int ix, iy;
		double dE;
		while(1){
			if(verbose >= 3) printf("doing Nt=%d\n", *Nt);
			get_flip_point(s, L, h, Temp, &ix, &iy, &dE, rng);
			if(verbose >= 3) printf("flip done\n");
			++(*Nt);
			s[ix*L + iy] *= -1;
			M_current += 2 * s[ix*L + iy];
			E_current += dE;
			if(verbose >= 3) printf("state modified\n");

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
			}
			if(verbose >= 3) printf("done Nt=%d\n", *Nt-1);
			if(Nt_max > 0){
				if(*Nt >= Nt_max){
					if(verbose >= 2) printf("Reached desired Nt >= Nt_max (= %d)\n", Nt_max);
					return 1;
				}
			} else {
				if(M_current == M_0){
					if(verbose >= 2) printf("Fail run\n");
					return 0;   // failed = gone to the initial state A
				} else if(M_current == M_next){
					if(verbose >= 2) printf("Success run\n");
					return 1;   // succeeded = reached the interface 'M == M_next'
				}
			}
		}
	}

    int print_E(const double *E, int Nt, char prefix, char suffix)
    {
        if(prefix > 0){
            printf("%c\n", prefix);
        }

        for(int i = 0; i < Nt; ++i){
            printf("%lf ", E[i]);
        }

        if(suffix > 0){
            printf("%c", suffix);
        }

        return 0;
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
}

