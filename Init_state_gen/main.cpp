#include <iostream>

#include "Izing.h"


int main(int argc, char** argv) {
    if(argc != 12){
        printf("usage:\n%s   L   T   h   N_init_states   M_0   M_max   N_M_interfaces   init_gen_mode   to_remember_EM   verbose   seed\n", argv[0]);
        return 1;
    }

    int L = atoi(argv[1]);
    double Temp = atof(argv[2]);
    double h =  atof(argv[3]);
    int N_init_states_default = atoi(argv[4]);
    int M_0 = atoi(argv[5]);
	int M_max = atoi(argv[6]);
	int N_M_interfaces = atoi(argv[7]);
	int init_gen_mode = atoi(argv[8]);
    int to_remember_EM = atoi(argv[9]);
    int verbose = atoi(argv[10]);
    int my_seed = atoi(argv[11]);

	L = 11;
	Temp = 2.0;
	h = -0.01;
	N_init_states_default = 10;
	M_0 = -L*L + 20;
	M_max = -M_0;
	N_M_interfaces = 10;
	init_gen_mode = -2;
	to_remember_EM = 1;
	verbose = 1;
	my_seed = 2;

    int i, j;
	int L2 = L*L;
	int state_size_in_bytes = L2 * sizeof(int);
	int M_arr_len = 128;

// [(-L2)---M_0](---M_1](---...---M_n-2](---M_n-1](---L2]
//        A       1       2 ...n-1       n-1        B
//        0       1       2 ...n-1       n-1       n
	int *Nt = (int*) malloc(sizeof(int) * (N_M_interfaces + 1));
//	int *M_arr_len = (int*) malloc(sizeof(int) * (N_M_interfaces + 1));
	int *M_interfaces = (int*) malloc((sizeof(int) * (N_M_interfaces + 2)));
	int *N_init_states = (int*) malloc(sizeof(int) * (N_M_interfaces + 2));
	double *probs = (double*) malloc(sizeof (double) * (N_M_interfaces + 1));
	double *d_probs = (double*) malloc(sizeof (double) * (N_M_interfaces + 1));

	M_interfaces[0] = -L2-1;   // here I want runs to finish only on exiting from A to M_0
	N_init_states[0] = N_init_states_default;
	int N_states_total = N_init_states[0];
	for(i = 0; i <= N_M_interfaces; ++i) {
		Nt[i] = 0;
//		M_arr_len[i] = M_arr_len_default;
//		states[i] = (int*)malloc(state_size_in_bytes * N_init_states[i]);

		//		N_init_states[i+1] = N_init_states_default + gsl_rng_uniform_int(Izing::rng, (N_init_states_default / 10) * 2 + 1) - N_init_states_default / 10;
		N_init_states[i+1] = N_init_states_default;
		N_states_total += N_init_states[i+1];
		M_interfaces[i+1] = (i < N_M_interfaces ? M_0 + (int)((M_max - M_0) * (double)(i) / (N_M_interfaces - 1) / 2) * 2 : L2);   // TODO: check if I can put 'L2+1' here
		assert(M_interfaces[i+1] > M_interfaces[i]);
		assert((M_interfaces[i+1] - M_0) % 2 == 0);   // M_step = 2, so there must be integer number of M_steps between all the M-s on interfaces
	}
	int *states = (int*) malloc(state_size_in_bytes * N_states_total);   // technically there are N+2 states' sets, but we are not interested in the first and the last sets

	double **E;
    int **M;
//    if(to_remember_EM){
//		E = (double**) malloc(sizeof(double*) * 1);
//		*(E) = (double*) malloc(sizeof(double) * M_arr_len);
//		M = (int**) malloc(sizeof(int*) * 1);
//		*(M) = (int*) malloc(sizeof(int) * M_arr_len);
//    }

	//    printf("0: %d\n", Izing::get_seed_C());
    Izing::init_rand_C(my_seed);
//    printf("1: %d\n", Izing::get_seed_C());

	double *flux0 = (double *) malloc(sizeof(double ));
	double *d_flux0 = (double *) malloc(sizeof(double ));;

	Izing::run_FFS_C(flux0, d_flux0, L, Temp, h, states, N_init_states, Nt, &M_arr_len, M_interfaces, N_M_interfaces, probs, d_probs, E, M, to_remember_EM, verbose, init_gen_mode);

    if(to_remember_EM){
		free(*E);   // array data
		free(E);   // the pointer to the array
		free(*M);
		free(M);
    }

    free(states);
	free(probs);
	free(d_probs);
	free(Nt);
	free(M_interfaces);
	free(N_init_states);
	free(flux0);
	free(d_flux0);

	printf("DONE\n");

    return 0;
}
