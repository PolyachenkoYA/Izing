#include <iostream>

#include "lattice_gas.h"

#include <pybind11/embed.h>

void test_FFS_C();
void test_BF();

int main(int argc, char** argv) {
//	py::scoped_interpreter guard{};
//	py::module_ sys = py::module_::import("sys");
//	py::print(sys.attr("path"));

	test_BF();
//	test_FFS_C();

	printf("DONE\n");

	return 0;
}

void test_BF()
{
	double phi0[] = {0, 0.05, 0.02};
	phi0[0] = 1.0 - phi0[1] - phi0[2];
	int my_seed = 23;
	int L = 16;
	int i, j;
	int L2 = L*L;
	int to_use_smart_swap = 0;

	lattice_gas::init_rand_C(my_seed);

	lattice_gas::set_OP_default(L2);

	long N_saved_states_max = -1;
	int stab_step = -10;
	int Nt_max = 100000;
	int save_states_stride = 1;
	int move_mode = move_mode_long_swap;
	move_mode = move_mode_swap;

	int state_size_in_bytes = L2 * sizeof(int);

	lattice_gas::set_OP_default(L2);

	double e_ptr[] = {0.000000, 0.000000, 0.000000, 0.000000, -2.680103, -1.340051, 0.000000, -1.340051, -1.715266};
	double mu_ptr[] = {0.000000, 0.000000, 0.000000};

// -------------- check input ----------------
	int verbose = 1;
	int to_remember_timeevol = 1;
	int interface_mode = mode_ID_CS;   // 'M' mode
	int OP_A = 14;
	int OP_B = 44;
	int OP_min_save_state = OP_A;
	int OP_max_save_state = OP_B;
	int OP_min = -1;
	int OP_max = L2 + 1;
	int N_spins_up_init = -1;
	if(stab_step < 0) stab_step *= (-L2);
	int to_equilibrate = 0;


	int *_init_state_ptr = (int*)malloc(state_size_in_bytes);
	lattice_gas::generate_state(_init_state_ptr, L, lround(phi0[main_specie_id] * L2), mode_ID_M, verbose);

// ----------------- create return objects --------------
	long Nt = 0;
	long Nt_OP_saved = 0;
	int N_states_saved;
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

//	int *states_ptr = (int*)malloc(std::max((long)1, N_saved_states_max) * state_size_in_bytes);
	int *states_ptr = (int*)malloc(std::max((long)3, N_saved_states_max) * state_size_in_bytes);

	N_states_saved = 0;

	lattice_gas::get_equilibrated_state(move_mode, L, e_ptr, mu_ptr, states_ptr, &N_states_saved,
										interface_mode, OP_A, OP_B, stab_step, _init_state_ptr, to_use_smart_swap,
										to_equilibrate, verbose);
	++N_states_saved;
	// N_states_saved is set to its initial values by default, so the equilibrated state is not saved
	// ++N prevents further processes from overwriting the initial state so it will be returned as states[0]

/*
 * int move_mode, int L, const double *e, const double *mu, long *time_total, int N_states, int *states,
						 long *OP_arr_len, long *Nt, long *Nt_OP_saved, double **E, int **M, int **biggest_cluster_sizes, int **h_A, int **time,
						 int interface_mode, int OP_A, int OP_B, int to_cluster, int to_start_only_state0,
						 int OP_min_stop_state, int OP_max_stop_state, int *N_states_done,
						 int OP_min_save_state, int OP_max_save_state, int save_state_mode,
						 int N_spins_up_init, int verbose, long Nt_max, int *N_tries, int to_save_final_state,
						 int to_regenerate_init_state, long save_states_stride, int to_use_smart_swap
 */
	lattice_gas::run_bruteforce_C(move_mode, L, e_ptr, mu_ptr, &time_total, N_saved_states_max, states_ptr,
								  to_remember_timeevol ? &OP_arr_len : nullptr,
								  &Nt, &Nt_OP_saved, &_E, &_M, &_biggest_cluster_sizes, &_h_A, &_time,
								  interface_mode, OP_A, OP_B, OP_A > 1, 0,
								  OP_min, OP_max, &N_states_saved,
								  OP_min_save_state, OP_max_save_state, save_state_mode_Inside,
								  N_spins_up_init, verbose, Nt_max, &N_launches, 0,
								  (OP_A <= 1) && (!bool(_init_state_ptr)), save_states_stride,
								  to_use_smart_swap);

	if(to_remember_timeevol){
		free(_E);
		free(_M);
		free(_biggest_cluster_sizes);
		free(_h_A);
		free(_time);
	}

	free(states_ptr);
	free(_init_state_ptr);
}

//void test_BF_PY()
//{
//	py::scoped_interpreter guard{};
//	py::module_ sys = py::module_::import("sys");
//	py::print(sys.attr("path"));
//	py::module_ mm = py::module_::import("numpy");
//	py::module_ mmm = py::module_::import("numpy.core.multiarray");
//
//	int move_mode = move_mode_long_swap;
//	int L = 64;
//	py::gil_scoped_acquire acquire; // reacquire GIL before calling py::array_t ctor
//	auto e = py::array_t<double>(N_species * N_species);
//	auto mu = py::array_t<double>(N_species);
//	long Nt_max = 150000;
//	long N_states_to_save = -1;
//	long save_states_stride = 100;
//	long stab_step = -10;
//	int OP_A = 20;
//	int OP_B = 60;
//
//	run_bruteforce(move_mode, L, e, mu, Nt_max, N_states_to_save, save_states_stride, stab_step,
//				   0, 1, OP_A, OP_B, -1, L*L+1,
//				   OP_A, OP_B, mode_ID_CS, py::none(), 1);
//}

void test_FFS_C()
{
	int verbose = 1;
	int my_seed = 2;

	double phi0[] = {0, 0.02, 0.01};
	phi0[0] = 1.0 - phi0[1] - phi0[2];
	int L = 64;
//	L = 32;
	int N_init_states_default = 1000;
	int N_init_states_A = 1000000;
	int N_OP_interfaces = 5;
	int init_gen_mode = -3;
	int to_remember_timeevol = 0;
	int interface_mode = mode_ID_CS;
	int OP_0;
	int OP_max;
	int move_mode = move_mode_flip;
	int stab_step = -10;
	int to_use_smart_swap = 0;

	move_mode = move_mode_long_swap;
	move_mode = move_mode_swap;

	// for valgrind
	N_init_states_default = 30;
	N_init_states_A = 30;

	int L2 = L*L;

	switch (interface_mode) {
		case mode_ID_M:
			OP_0 = 10;
			OP_max = -OP_0;
			break;
		case mode_ID_CS:
			OP_0 = 24;
			OP_max = 500;
			OP_0 = 6;
			OP_max = 14;
			break;
	}

//	OP_0 = 2;
//	OP_max = L2 - 2;

	int i, j;
	int state_size_in_bytes = L2 * sizeof(int);
	long OP_arr_len = 128;

//	double e = (double*) malloc(sizeof(double) * N_species * N_species);
//	double mu = (double*) malloc(sizeof(double) * N_species);
	double e[] = {0.000000, 0.000000, 0.000000, 0.000000, -2.680103, -1.340051, 0.000000, -1.340051, -1.715266};
	double mu[] = {0.000000, 4.9, 4.92238326};


//	double J_T = 1 / 1.5;
//	double h_T = 0.1 / 1.5;
//	e[0] = e[1] = e[2] = e[1*3 + 0] = e[2*3 + 0] = 0;
//	mu[0] = 0;
//
//	e[1*3 + 1] = 4 * J_T;
////	e[2*3 + 2] = 4 * J_T;
//	e[2*3 + 2] = 0;
//
//	e[1*3 + 2] = e[2*3 + 1] = sqrt(e[1*3 + 1] * e[2*3 + 2]);
//
//	mu[1] = h_T - 4 * J_T;
////	mu[2] = h_T - 4 * J_T;
//	mu[2] = -1e10;

	lattice_gas::set_OP_default(L2);

// [OP_min---OP_0](---OP_1](---...---OP_n-2](---OP_n-1](---OP_max]
//        A       1       2 ...n-1       n-1        B
//        0       1       2 ...n-1       n-1       n
	long *Nt = (long*) malloc(sizeof(long) * (N_OP_interfaces));
	long *Nt_OP_saved = (long*) malloc(sizeof(long) * (N_OP_interfaces));
//	int *OP_arr_len = (int*) malloc(sizeof(int) * (N_OP_interfaces + 1));
	int *OP_interfaces = (int*) malloc((sizeof(int) * (N_OP_interfaces)));
	int *N_init_states = (int*) malloc(sizeof(int) * (N_OP_interfaces));
	double *probs = (double*) malloc(sizeof (double) * (N_OP_interfaces - 1));
	double *d_probs = (double*) malloc(sizeof (double) * (N_OP_interfaces - 1));

//	OP_interfaces[0] = -L2-1;   // here I want runs to finish only on exiting from A to OP_0
	N_init_states[0] = N_init_states_A;
	Nt[0] = 0;
	int N_states_total = N_init_states[0];
	for(i = 1; i < N_OP_interfaces; ++i) {
		Nt[i] = 0;
//		OP_arr_len[i] = M_arr_len_default;
//		states[i] = (int*)malloc(state_size_in_bytes * N_init_states[i]);

		//		N_init_states[i+1] = N_init_states_default + gsl_rng_uniform_int(lattice_gas::rng, (N_init_states_default / 10) * 2 + 1) - N_init_states_default / 10;
		N_init_states[i] = N_init_states_default;
		N_states_total += N_init_states[i];
	}
	for(i = 0; i < N_OP_interfaces; ++i) {
		switch (interface_mode) {
			case mode_ID_M:
//				OP_interfaces[i] = (i == N_OP_interfaces + 1 ? L2 : (i == 0 ? (-L2-1) : (OP_0 + lround((OP_max - OP_0) * (double)(i - 1) / (N_OP_interfaces - 1) / 2) * 2)));   // TODO: check if I can put 'L2+1' here
				OP_interfaces[i] = OP_0 + lround((OP_max - OP_0) * (double)(i) / (N_OP_interfaces - 1) / lattice_gas::OP_step[interface_mode]) * lattice_gas::OP_step[interface_mode];   // TODO: check if I can put 'L2+1' here
				if(i > 0){
					assert(OP_interfaces[i] > OP_interfaces[i-1]);
					assert((OP_interfaces[i] - OP_0) % lattice_gas::OP_step[interface_mode] == 0);   // M_step = 2, so there must be integer number of M_steps between all the M-s on interfaces
				}
				break;
			case mode_ID_CS:
//				OP_interfaces[i] = (i == N_OP_interfaces + 1 ? L2 : (i == 0 ? -1 : (OP_0 + lround((OP_max - OP_0) * (double)(i - 1) / (N_OP_interfaces - 1)))));   // TODO: check if I can put 'L2+1' here
				OP_interfaces[i] = OP_0 + lround((OP_max - OP_0) * (double)(i) / (N_OP_interfaces - 1));   // TODO: check if I can put 'L2+1' here
				assert(OP_interfaces[i] > lattice_gas::OP_min_default[interface_mode]);
				if(i > 0){
					assert(OP_interfaces[i] > OP_interfaces[i-1]);
				}
				break;
		}
	}
	int *states = (int*) malloc(state_size_in_bytes * N_states_total);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets

	int *states_parent_inds = (int*) malloc(state_size_in_bytes * (N_states_total - N_init_states[0]));

	double *E;
	int *M;
	int *time;
	int *biggest_cluster_sizes;
	if(to_remember_timeevol){
		E = (double*) malloc(sizeof(double) * OP_arr_len);
		M = (int*) malloc(sizeof(int) * OP_arr_len);
		time = (int*) malloc(sizeof(int) * OP_arr_len);
		biggest_cluster_sizes = (int*) malloc(sizeof(int) * OP_arr_len);
	}

	//    printf("0: %d\n", lattice_gas::get_seed_C());
	lattice_gas::init_rand_C(my_seed);
//    printf("1: %d\n", lattice_gas::get_seed_C());

	double flux0;
	double d_flux0;

	if(verbose){
		printf("OP interfaces :\n");
		for(i = 0; i < N_OP_interfaces; ++i){
			printf("%d ", OP_interfaces[i]);
		}
		printf("\n");
	}

	int *_init_state_ptr = (int*)malloc(state_size_in_bytes);
	lattice_gas::generate_state(_init_state_ptr, L, lround(phi0[main_specie_id] * L2) + 1, mode_ID_M, verbose);

	/*
	 * 	int run_FFS_C(int move_mode, double *flux0, double *d_flux0, int L, const double *e, const double *mu, int *states,
				  int *states_parent_inds, int *N_init_states, long *Nt, long *Nt_OP_saved, long stab_step,
				  long *OP_arr_len, int *OP_interfaces, int N_OP_interfaces, double *probs, double *d_probs, double **E, int **M,
				  int **biggest_cluster_sizes, int **time, int verbose, int init_gen_mode, int interface_mode,
				  const int *init_state, int to_use_smart_swap);
	 */
	lattice_gas::run_FFS_C(move_mode, &flux0, &d_flux0, L, e, mu, states, states_parent_inds, N_init_states,
						   Nt, Nt_OP_saved, stab_step, to_remember_timeevol ? &OP_arr_len : nullptr, OP_interfaces, N_OP_interfaces,
						   probs, d_probs, &E, &M, &biggest_cluster_sizes, &time,
						   verbose, init_gen_mode, interface_mode, _init_state_ptr, to_use_smart_swap);

	if(to_remember_timeevol){
		free(E);   // the pointer to the array
		free(M);
		free(time);
		free(biggest_cluster_sizes);
	}

	free(states);
	free(probs);
	free(d_probs);
	free(Nt);
	free(Nt_OP_saved);
	free(OP_interfaces);
	free(N_init_states);
	free(_init_state_ptr);
}
