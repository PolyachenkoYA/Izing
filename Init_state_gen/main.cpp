#include <iostream>

#include "Izing.h"


int main(int argc, char** argv) {
    if(argc != 13){
        printf("usage:\n%s   L   T   h   N_init_states   OP_0   OP_max   N_M_interfaces   init_gen_mode   to_remember_timeevol   interface_mode   verbose   seed\n", argv[0]);
        return 1;
    }

    int L = atoi(argv[1]);
    double Temp = atof(argv[2]);
    double h =  atof(argv[3]);
    int N_init_states_default = atoi(argv[4]);
    int OP_0 = atoi(argv[5]);
	int OP_max = atoi(argv[6]);
	int N_OP_interfaces = atoi(argv[7]);
	int init_gen_mode = atoi(argv[8]);
    int to_remember_timeevol = atoi(argv[9]);
	int interface_mode = atoi(argv[10]);
    int verbose = atoi(argv[11]);
    int my_seed = atoi(argv[12]);

	L = 11;
	Temp = 2.1;
	h = -0.01;
	N_init_states_default = 50;
	N_OP_interfaces = 30;
	init_gen_mode = -2;
	to_remember_timeevol = 1;
	interface_mode = 2;
	verbose = 1;
	my_seed = 2;

	switch (interface_mode) {
		case 1:
			OP_0 = -L*L + 4;
			OP_max = -OP_0;
			break;
		case 2:
			OP_0 = 3;
			OP_max = L * L - 10;
			break;
	}

    int i, j;
	int L2 = L*L;
	int state_size_in_bytes = L2 * sizeof(int);
	int OP_arr_len = 128;

// [OP_min---OP_0](---OP_1](---...---OP_n-2](---OP_n-1](---OP_max]
//        A       1       2 ...n-1       n-1        B
//        0       1       2 ...n-1       n-1       n
	int *Nt = (int*) malloc(sizeof(int) * (N_OP_interfaces + 1));
//	int *OP_arr_len = (int*) malloc(sizeof(int) * (N_OP_interfaces + 1));
	int *OP_interfaces = (int*) malloc((sizeof(int) * (N_OP_interfaces + 2)));
	int *N_init_states = (int*) malloc(sizeof(int) * (N_OP_interfaces + 2));
	double *probs = (double*) malloc(sizeof (double) * (N_OP_interfaces + 1));
	double *d_probs = (double*) malloc(sizeof (double) * (N_OP_interfaces + 1));

//	OP_interfaces[0] = -L2-1;   // here I want runs to finish only on exiting from A to OP_0
	N_init_states[0] = 100;
	int N_states_total = N_init_states[0];
	for(i = 0; i <= N_OP_interfaces; ++i) {
		Nt[i] = 0;
//		OP_arr_len[i] = M_arr_len_default;
//		states[i] = (int*)malloc(state_size_in_bytes * N_init_states[i]);

		//		N_init_states[i+1] = N_init_states_default + gsl_rng_uniform_int(Izing::rng, (N_init_states_default / 10) * 2 + 1) - N_init_states_default / 10;
		N_init_states[i + 1] = N_init_states_default;
		N_states_total += N_init_states[i + 1];
	}
	for(i = 0; i < N_OP_interfaces + 2; ++i) {
		switch (interface_mode) {
			case 1:
				OP_interfaces[i] = (i == N_OP_interfaces + 1 ? L2 : (i == 0 ? (-L2-1) : (OP_0 + lround((OP_max - OP_0) * (double)(i - 1) / (N_OP_interfaces - 1) / 2) * 2)));   // TODO: check if I can put 'L2+1' here
				if(i > 0){
					assert(OP_interfaces[i] > OP_interfaces[i-1]);
					assert((OP_interfaces[i] - OP_0) % 2 == 0);   // M_step = 2, so there must be integer number of M_steps between all the M-s on interfaces
				}
				break;
			case 2:
				OP_interfaces[i] = (i == N_OP_interfaces + 1 ? L2 : (i == 0 ? -1 : (OP_0 + lround((OP_max - OP_0) * (double)(i - 1) / (N_OP_interfaces - 1)))));   // TODO: check if I can put 'L2+1' here
				if(i > 0){
					assert(OP_interfaces[i] > OP_interfaces[i-1]);
				}
				break;
		}
	}
	int *states = (int*) malloc(state_size_in_bytes * N_states_total);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets

	double *E;
    int *M;
	int *biggest_cluster_sizes;
    if(to_remember_timeevol){
		E = (double*) malloc(sizeof(double) * OP_arr_len);
		M = (int*) malloc(sizeof(int) * OP_arr_len);
		biggest_cluster_sizes = (int*) malloc(sizeof(int) * OP_arr_len);
    }

	//    printf("0: %d\n", Izing::get_seed_C());
    Izing::init_rand_C(my_seed);
//    printf("1: %d\n", Izing::get_seed_C());

	double flux0;
	double d_flux0;

	Izing::run_FFS_C(&flux0, &d_flux0, L, Temp, h, states, N_init_states,
			  Nt, &OP_arr_len, OP_interfaces, N_OP_interfaces,
			  probs, d_probs, &E, &M, &biggest_cluster_sizes,
			  to_remember_timeevol, verbose, init_gen_mode, interface_mode);

    if(to_remember_timeevol){
		free(E);   // the pointer to the array
		free(M);
		free(biggest_cluster_sizes);
    }

    free(states);
	free(probs);
	free(d_probs);
	free(Nt);
	free(OP_interfaces);
	free(N_init_states);

	printf("DONE\n");

    return 0;
}
