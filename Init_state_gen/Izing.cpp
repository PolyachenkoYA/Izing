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

py::tuple get_init_states(int L, double Temp, double h, int N_init_states, int M_0, int to_get_EM, std::optional<int> _verbose)
{
    auto init_states = py::array_t<int>(N_init_states * L*L);
    py::buffer_info init_states_info = init_states.request();
    int *init_states_ptr = static_cast<int *>(init_states_info.ptr);
    int Nt;
	int M_arr_len = N_init_states;

	int verbose = (_verbose.has_value() ? _verbose.value() : Izing::verbose_dafault);

    double **_E;
    int **_M;
    if(to_get_EM){
        _E = (double**) malloc(sizeof(double*) * 1);
        *_E = (double*) malloc(sizeof(double) * M_arr_len);
        _M = (int**) malloc(sizeof(int*) * 1);
        *_M = (int*) malloc(sizeof(int) * M_arr_len);
    }

	if(verbose){
		printf("using: L=%d  T=%lf  h=%lf  N_init_states=%d  M_0=%d  EM=%d  v=%d\n", L, Temp, h, N_init_states, M_0, to_get_EM, verbose);
	}

    Izing::get_init_states_C(L, Temp, h, N_init_states, M_0, init_states_ptr, _E, _M, &Nt, &M_arr_len, to_get_EM, verbose);

    if(verbose){
		printf("Nt = %d\n", Nt);
	}

    py::array_t<double> E;
    py::array_t<double> M;
    if(to_get_EM){
        E = py::array_t<double>(Nt);
        M = py::array_t<double>(Nt);
        py::buffer_info E_info = E.request();
        py::buffer_info M_info = M.request();
        double *E_ptr = static_cast<double *>(E_info.ptr);
        double *M_ptr = static_cast<double *>(M_info.ptr);
        memcpy(E_ptr, *_E, sizeof(double) * Nt);
        memcpy(M_ptr, *_M, sizeof(double) * Nt);

        free(*_E);
        free(_E);
        free(*_M);
        free(_M);

		if(verbose >= 2){
			if(to_get_EM){
				Izing::print_E(E_ptr, fmin(10, Nt), 'P');
			}
		}
    }


    py::tuple res = py::make_tuple(init_states, E, M);
    return res;
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

    int comp_M(int *s, int L, int *M)
    {
        int i, j;
        int _M = 0;
        for(i = 0; i < L; ++i){
            for(j = 0; j < L; ++j){
                _M += s[i*L + j];
            }
        }

        *M = _M;
        return 0;
    }

    int comp_E(int* s, int L, double h, double *E)
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

        int _M;
        comp_M(s, L, &_M);

        *E = h * (_M) - _E; // J > 0 -> {++ > +-} -> we need to *(-1) because we search for a minimum
        // energy is measured in [E]=J, so J==1, or we store E/J value

        return 0;
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

            dE[0] = get_dE(s, L, h, *ix, *iy);
        }while(!(dE[0] <= 0 ? 1 : (gsl_rng_uniform(rng) < exp(- dE[0] / Temp) ? 1 : 0)));

        return 0;
    }

	int get_init_states_C(int L, double Temp, double h, int N_init_states, int M_0, int *init_states, double **E, int **M, int *Nt, int *M_arr_len, bool to_remember_EM, int verbose)
	{
		int i;
		int L2 = L*L;
		int state_size = sizeof(int) * L2;

		// generate N_init_states states in A
		// Here they are identical, but I think it's better to generate them accordingly to equilibrium distribution in A
		for(i = 0; i < N_init_states; ++i){
			generate_state(&(init_states[i * L2]), L, rng, 0); // allocate all spins = -1
		}

		*Nt += 1;
		// record initial state (all DOWN) E and M since run_state changes it before recording
		if(to_remember_EM){
			comp_M(init_states, L, &((*M)[*Nt - 1]));
			comp_E(init_states, L, h, &((*E)[*Nt - 1]));
		}

		return 0;
	}

//	int get_init_states_C(int L, double Temp, double h, int N_init_states, int M_0, int *init_states, double **E, int **M, int *Nt, int *M_arr_len, bool to_remember_EM, int verbose)
//	{
//		int i;
//		int L2 = L*L;
//		int state_size = sizeof(int) * L2;
//
//		// generate N_init_states states in A
//		// Here they are identical, but I think it's better to generate them accordingly to equilibrium distribution in A
//		int *states_to_run = (int*) malloc(N_init_states * state_size);
//		for(i = 0; i < N_init_states; ++i){
//			generate_state(&(states_to_run[i * L2]), L, rng, 0); // allocate all spins = -1
//		}
//		*Nt += 1;
//
//		// record initial state (all DOWN) E and M since run_state changes it before recording
//		comp_M(states_to_run, L, &((*M)[*Nt - 1]));
//		comp_E(states_to_run, L, h, &((*E)[*Nt - 1]));
//
//		if(verbose >= 2){
//			printf("states generated\n");
//		}
//		process_step(states_to_run, init_states, E, M, Nt, M_arr_len, N_init_states, L, Temp, h, -L2-1, M_0, to_remember_EM, verbose);
//
//        if(verbose) {
//			if (to_remember_EM) {
//				print_E(*E, fmin(10, *Nt), 'E');
//			}
//			printf("Nt_in = %d\n", *Nt);
//		}
//
//		free(states_to_run);
//
//		return 0;
//	}

	double process_step(int *init_states, int *next_states, double **E, int **M, int *Nt, int *M_arr_len, int N_init_states, int N_next_states, int L, double Temp, double h, int M_0, int M_next, int to_save_next_states, bool to_remember_EM, int verbose)
	/**
	 *
	 * @param init_states - are assumed to contain 'N_init_states * state_size' ints representing states to start simulations from
	 * @param next_states - are assumed to be allocated to have 'N_init_states * state_size' ints
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
		int L2 = L*L;
		int state_size = sizeof(int) * L2;
		int N_succ = 0;
		int N_runs = 0;
		int *state_under_process = (int*) malloc(state_size);
		if(verbose){
			printf("doing step:(%d; %d]\n", M_0, M_next);
			if(verbose >= 2){
				getchar();
			}
		}
		while(N_succ < N_next_states){
			int init_state_to_process_ID = gsl_rng_uniform_int(rng, N_init_states);
			if(verbose >= 2){
				printf("state %d:\n", N_succ);
			}
			memcpy(state_under_process, &(init_states[init_state_to_process_ID * L2]), state_size);   // get a copy of the chosen init state
			if(run_state(state_under_process, L, Temp, h, M_0, M_next, E, M, Nt, M_arr_len, to_remember_EM, verbose)){   // run it until it reaches M_0 or M_next
				// Nt is not reinitialized to 0 and that's correct because it shows the total number of EM datapoints
				// the run reached M_next

				++N_succ;
				if(to_save_next_states) {
					memcpy(&(next_states[(N_succ - 1) * L2]), state_under_process, state_size);   // save the resulting system state for the next step
				}
				if(verbose) {
					if(verbose >= 2){
						printf("state %d saved for future\n", N_succ - 1);
					}
					if((N_succ % (N_next_states / 1000 + 1) == 0) || (verbose >= 2)){
						printf("%lf %%          \r", (double)N_succ/N_next_states * 100);
						fflush(stdout);
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

	int run_state(int *s, int L, double Temp, double h, int M_0, int M_next, double **E, int **M, int *Nt, int *M_arr_len, bool to_remember_EM, int verbose)
	{
		int M_current;
		double E_current;
		comp_E(s, L, h, &E_current); // remember the 1st energy
		comp_M(s, L, &M_current); // remember the 1st M
		if(verbose >= 2){
			printf("E=%lf, M=%d\n", E_current, M_current);
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
				}
				(*M)[*Nt - 1] = M_current;
				(*E)[*Nt - 1] = E_current;
			}
			if(verbose >= 3) printf("done Nt=%d\n", *Nt-1);
			if(M_current == M_0){
				if(verbose >= 2) printf("Fail run\n");
				return 0;   // failed = gone to the initial state A
			} else if(M_current == M_next){
				if(verbose >= 2) printf("Success run\n");
				return 1;   // succeeded = reached the interface 'M == M_next'
			}
		}
	}

    int print_E(double *E, int Nt, char prefix, char suffix)
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

    int print_S(int *s, int L, char prefix)
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

