#include <iostream>

#include "Izing.h"

void test_FFC()
{
	int L;
	double Temp;
	double h;
	int N_init_states_default;
	int N_init_states_A;
	int OP_0;
	int OP_max;
	int N_OP_interfaces;
	int init_gen_mode;
	int to_remember_timeevol;
	int interface_mode;
	int def_spin_state;
	int verbose;
	int my_seed;

	def_spin_state = -1;
	verbose = 1;
	my_seed = 2;

	L = 11;
	Temp = 2.1;
	h = -0.01;
	N_init_states_default = 1000;
	N_init_states_A = 100000;
	N_OP_interfaces = 30;
	init_gen_mode = -3;
	to_remember_timeevol = 1;
	interface_mode = mode_ID_M;

	// for valgrind
	N_init_states_default = 50;
	N_init_states_A = 100;

	int L2 = L*L;

	switch (interface_mode) {
		case mode_ID_M:
			OP_0 = -L2 + 4;
			OP_max = -OP_0;
			break;
		case mode_ID_CS:
			OP_0 = 3;
			OP_max = L2 - 10;
			break;
	}

	int i, j;
	int state_size_in_bytes = L2 * sizeof(int);
	long OP_arr_len = 128;

	Izing::set_OP_default(L2);

// [OP_min---OP_0](---OP_1](---...---OP_n-2](---OP_n-1](---OP_max]
//        A       1       2 ...n-1       n-1        B
//        0       1       2 ...n-1       n-1       n
	long *Nt = (long*) malloc(sizeof(long) * (N_OP_interfaces));
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

		//		N_init_states[i+1] = N_init_states_default + gsl_rng_uniform_int(Izing::rng, (N_init_states_default / 10) * 2 + 1) - N_init_states_default / 10;
		N_init_states[i] = N_init_states_default;
		N_states_total += N_init_states[i];
	}
	for(i = 0; i < N_OP_interfaces; ++i) {
		switch (interface_mode) {
			case mode_ID_M:
//				OP_interfaces[i] = (i == N_OP_interfaces + 1 ? L2 : (i == 0 ? (-L2-1) : (OP_0 + lround((OP_max - OP_0) * (double)(i - 1) / (N_OP_interfaces - 1) / 2) * 2)));   // TODO: check if I can put 'L2+1' here
				OP_interfaces[i] = OP_0 + lround((OP_max - OP_0) * (double)(i) / (N_OP_interfaces - 1) / Izing::OP_step[interface_mode]) * Izing::OP_step[interface_mode];   // TODO: check if I can put 'L2+1' here
				if(i > 0){
					assert(OP_interfaces[i] > OP_interfaces[i-1]);
					assert((OP_interfaces[i] - OP_0) % Izing::OP_step[interface_mode] == 0);   // M_step = 2, so there must be integer number of M_steps between all the M-s on interfaces
				}
				break;
			case mode_ID_CS:
//				OP_interfaces[i] = (i == N_OP_interfaces + 1 ? L2 : (i == 0 ? -1 : (OP_0 + lround((OP_max - OP_0) * (double)(i - 1) / (N_OP_interfaces - 1)))));   // TODO: check if I can put 'L2+1' here
				OP_interfaces[i] = OP_0 + lround((OP_max - OP_0) * (double)(i - 1) / (N_OP_interfaces - 1));   // TODO: check if I can put 'L2+1' here
				if(i > 0){
					assert(OP_interfaces[i] > OP_interfaces[i-1]);
				}
				break;
		}
	}
	int *states = (int*) malloc(state_size_in_bytes * N_states_total);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets

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

	//    printf("0: %d\n", Izing::get_seed_C());
	Izing::init_rand_C(my_seed);
//    printf("1: %d\n", Izing::get_seed_C());

	double flux0;
	double d_flux0;

	Izing::run_FFS_C(&flux0, &d_flux0, L, Temp, h, states, N_init_states,
					 Nt, to_remember_timeevol ? &OP_arr_len : nullptr, OP_interfaces, N_OP_interfaces,
					 probs, d_probs,  &E,&M,&biggest_cluster_sizes, &time,
					 verbose, init_gen_mode, interface_mode,
					 def_spin_state);

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
	free(OP_interfaces);
	free(N_init_states);

	printf("test FFS DONE\n");
}

void test_BF()
{
//	py::tuple run_bruteforce(int L, double Temp, double h, long Nt_max, long N_saved_states_max, long dump_step,
//							 std::optional<int> _N_spins_up_init, std::optional<int> _to_remember_timeevol,
//							 std::optional<int> _OP_A, std::optional<int> _OP_B,
//							 std::optional<int> _OP_min, std::optional<int> _OP_max,
//							 std::optional<int> _interface_mode, std::optional<int> _default_spin_state,
//							 std::optional<int> _verbose)
//
	int L = 16;
	double Temp = 5;
	double h = 0;
	long Nt_max = 170717;
	int init_gen_mode = 0;
	int OP_0 = 24;
	int OP_max = 1000000;
	int def_spin_state = -1;
	int N_spins_up_init = 0;
	int my_seed=23;

	int dump_step = 56;

	int i, j;
	int L2 = L*L;

	Izing::set_OP_default(L2);

// -------------- check input ----------------
	assert(L > 0);
	assert(Temp > 0);
	int verbose = 1;
	int to_remember_timeevol = 1;
	int interface_mode = 1;
	assert((interface_mode >= 0) && (interface_mode < N_interface_modes));
	int OP_min = -1;
//	int OP_max = 257;
	assert(OP_max > OP_min);
	int OP_A = 24;
	int OP_B = 1000000;
	long N_saved_states_max = 1000;
//	int N_spins_up_init = 0;

// ----------------- create return objects --------------
	long Nt = 0;
	long Nt_saved = 0;
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

	Izing::init_rand_C(my_seed);

	if(OP_A > 0){
		Izing::get_equilibrated_state(L, Temp, h, states, interface_mode, def_spin_state, OP_A, OP_B, verbose);
		N_states_saved = 1;
	}

	Izing::run_bruteforce_C(L, Temp, h, &time_total, INT_MAX, states,
							to_remember_timeevol ? &OP_arr_len : nullptr,
							&Nt, &Nt_saved, dump_step, &_E, &_M, &_biggest_cluster_sizes, &_h_A, &_time,
							interface_mode, def_spin_state, OP_A, OP_B,
							OP_min, OP_max, &N_states_saved,
							OP_min, OP_A, save_state_mode_Inside,
							N_spins_up_init, verbose, Nt_max, &N_launches, 0,
							0, N_saved_states_max);


	int N_last_elements_to_print = std::min(Nt, (long)10);

//	py::array_t<double> E;
//	py::array_t<int> M;
//	py::array_t<int> biggest_cluster_sizes;
//	py::array_t<int> h_A;
//	py::array_t<int> time;
//	if(to_remember_timeevol){
//		if(verbose >= 2){
//			printf("Brute-force core done, Nt_saved = %ld\n", Nt_saved);
//			Izing::print_E(&(_E[Nt_saved - N_last_elements_to_print]), N_last_elements_to_print, 'F');
//			Izing::print_M(&(_M[Nt_saved - N_last_elements_to_print]), N_last_elements_to_print, 'F');
////		Izing::print_biggest_cluster_sizes(&(_M[Nt_saved - N_last_elements_to_print]), N_last_elements_to_print, 'F');
//		}
//
//		E = py::array_t<double>(Nt_saved);
//		py::buffer_info E_info = E.request();
//		double *E_ptr = static_cast<double *>(E_info.ptr);
//		memcpy(E_ptr, _E, sizeof(double) * Nt_saved);
//		free(_E);
//
//		M = py::array_t<int>(Nt_saved);
//		py::buffer_info M_info = M.request();
//		int *M_ptr = static_cast<int *>(M_info.ptr);
//		memcpy(M_ptr, _M, sizeof(int) * Nt_saved);
//		free(_M);
//
//		biggest_cluster_sizes = py::array_t<int>(Nt_saved);
//		py::buffer_info biggest_cluster_sizes_info = biggest_cluster_sizes.request();
//		int *biggest_cluster_sizes_ptr = static_cast<int *>(biggest_cluster_sizes_info.ptr);
//		memcpy(biggest_cluster_sizes_ptr, _biggest_cluster_sizes, sizeof(int) * Nt_saved);
//		free(_biggest_cluster_sizes);
//
//		h_A = py::array_t<int>(Nt_saved);
//		py::buffer_info h_A_info = h_A.request();
//		int *h_A_ptr = static_cast<int *>(h_A_info.ptr);
//		memcpy(h_A_ptr, _h_A, sizeof(int) * Nt_saved);
//		free(_h_A);
//
//		time = py::array_t<int>(Nt_saved);
//		py::buffer_info time_info = time.request();
//		int *time_ptr = static_cast<int *>(time_info.ptr);
//		memcpy(time_ptr, _time, sizeof(int) * Nt_saved);
//		free(_time);
//
//		if(verbose >= 2){
//			printf("internal memory for EM freed\n");
//			Izing::print_E(&(E_ptr[Nt_saved - N_last_elements_to_print]), N_last_elements_to_print, 'P');
//			Izing::print_M(&(M_ptr[Nt_saved - N_last_elements_to_print]), N_last_elements_to_print, 'P');
//			printf("exiting py::run_bruteforce\n");
//		}
//
//	}
	free(_E);
	free(_M);
	free(_biggest_cluster_sizes);
	free(_h_A);
	free(_time);

	free(states);

	printf("BF DONE\n");
}

int main(int argc, char** argv) {
	test_BF();

    return 0;
}
