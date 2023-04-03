#include <iostream>

#include "lattice_gas.h"


int main(int argc, char** argv) {
//	if(argc != 15){
//		printf("usage:\n%s   L   T   h   N_init_states_all   N_init_states_A   OP_0   OP_max   N_M_interfaces   init_gen_mode   to_remember_timeevol   interface_mode   def_spin_state   verbose   seed\n", argv[0]);
//		return 1;
//	}
//
//	int L = atoi(argv[1]);
//	double Temp = atof(argv[2]);
//	double h =  atof(argv[3]);
//	int N_init_states_default = atoi(argv[4]);
//	int N_init_states_A = atoi(argv[5]);
//	int OP_0 = atoi(argv[6]);
//	int OP_max = atoi(argv[7]);
//	int N_OP_interfaces = atoi(argv[8]);
//	int init_gen_mode = atoi(argv[9]);
//	int to_remember_timeevol = atoi(argv[10]);
//	int interface_mode = atoi(argv[11]);
//	int def_spin_state = atoi(argv[12]);
//	int verbose = atoi(argv[13]);
//	int my_seed = atoi(argv[14]);

	int verbose = 2;
	int my_seed = 2;

	int L = 32;
	int N_init_states_default = 1000;
	int N_init_states_A = 100000;
	int N_OP_interfaces = 3;
	int init_gen_mode = -3;
	int to_remember_timeevol = 1;
	int interface_mode = mode_ID_CS;
	int OP_0;
	int OP_max;
	int move_mode = move_mode_flip;

	// for valgrind
	N_init_states_default = 10;
	N_init_states_A = 10;

	int L2 = L*L;

	switch (interface_mode) {
		case mode_ID_M:
			OP_0 = 10;
			OP_max = -OP_0;
			break;
		case mode_ID_CS:
			OP_0 = 24;
			OP_max = 500;
			break;
	}

//	OP_0 = 2;
//	OP_max = L2 - 2;

	int i, j;
	int state_size_in_bytes = L2 * sizeof(int);
	long OP_arr_len = 128;

	double *e = (double*) malloc(sizeof(double) * N_species * N_species);
	double *mu = (double*) malloc(sizeof(double) * N_species);

	double J_T = 1 / 1.5;
	double h_T = 0.1 / 1.5;
	e[0] = e[1] = e[2] = e[1*3 + 0] = e[2*3 + 0] = 0;
	mu[0] = 0;

	e[1*3 + 1] = 4 * J_T;
//	e[2*3 + 2] = 4 * J_T;
	e[2*3 + 2] = 0;

	e[1*3 + 2] = e[2*3 + 1] = sqrt(e[1*3 + 1] * e[2*3 + 2]);

	mu[1] = h_T - 4 * J_T;
//	mu[2] = h_T - 4 * J_T;
	mu[2] = -1e10;

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

	lattice_gas::run_FFS_C(move_mode, &flux0, &d_flux0, L, e, mu, states, N_init_states,
						   Nt, Nt_OP_saved, to_remember_timeevol ? &OP_arr_len : nullptr, OP_interfaces, N_OP_interfaces,
						   probs, d_probs, &E, &M, &biggest_cluster_sizes, &time,
						   verbose, init_gen_mode, interface_mode, nullptr);

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
	free(e);
	free(mu);

	printf("DONE\n");

	return 0;
}
