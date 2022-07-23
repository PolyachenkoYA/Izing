#include <iostream>

#include "Izing.h"

int main(int argc, char** argv) {
    if(argc != 11){
        printf("usage:\n%s   L   T   h   N_init_states   M_0   M_max   N_M_interfaces   to_remember_EM   verbose   seed\n", argv[0]);
        return 1;
    }

    int L = atoi(argv[1]);
    double Temp = atof(argv[2]);
    double h =  atof(argv[3]);
    int N_init_states_default = atoi(argv[4]);
    int M_0 = atoi(argv[5]);
	int M_max = atoi(argv[6]);
	int N_M_interfaces = atoi(argv[7]);
    int to_remember_EM = atoi(argv[8]);
    int verbose = atoi(argv[9]);
    int my_seed = atoi(argv[10]);

//	L = 11;
//	Temp = 2.0;
//	h = -0.01;
//	N_init_states_default = 10;
//	M_0 = -L*L + 20;
//	M_max = -M_0;
//	N_M_interfaces = 10;
//	to_remember_EM = 1;
//	verbose = 1;
//	my_seed = 2;

    int i, j;
	int L2 = L*L;
	int state_size = L2 * sizeof(int);
	int M_arr_len_default = 128;

// [(-L2)---M_0](---M_1](---...---M_n-2](---M_n-1](---L2]
//        A       1       2 ...n-1       n-1        B
//        0       1       2 ...n-1       n-1       n
	int *Nt = (int*) malloc(sizeof(int) * (N_M_interfaces + 1));
	int *M_arr_len = (int*) malloc(sizeof(int) * (N_M_interfaces + 1));
	int *M_interfaces = (int*) malloc((sizeof(int) * (N_M_interfaces + 2)));
	int *N_init_states = (int*) malloc(sizeof(int) * (N_M_interfaces + 2));
	int **states = (int**) malloc(sizeof(int*) * (N_M_interfaces + 1));   // techically there are N+2 states' sets, but we are not interested in the first and the last sets
	double *probs = (double*) malloc(sizeof (double) * (N_M_interfaces + 1));
	double *d_probs = (double*) malloc(sizeof (double) * (N_M_interfaces + 1));

	M_interfaces[0] = -L2-1;   // here I want runs to finish only on exiting from A to M_0
	N_init_states[0] = N_init_states_default;
	for(i = 0; i <= N_M_interfaces; ++i) {
		Nt[i] = 0;
		M_arr_len[i] = M_arr_len_default;
		states[i] = (int*)malloc(state_size * N_init_states[i]);

		//		N_init_states[i+1] = N_init_states_default + gsl_rng_uniform_int(Izing::rng, (N_init_states_default / 10) * 2 + 1) - N_init_states_default / 10;
		N_init_states[i+1] = N_init_states_default;
		M_interfaces[i+1] = (i < N_M_interfaces ? M_0 + (int)((M_max - M_0) * (double)(i) / (N_M_interfaces - 1) / 2) * 2 : L2);   // TODO: check if I can put 'L2+1' here
		assert(M_interfaces[i+1] > M_interfaces[i]);
		assert((M_interfaces[i+1] - M_0) % 2 == 0);   // M_step = 2, so there must be integer number of M_steps between all the M-s on interfaces
	}

	double ***E;
    int ***M;
    if(to_remember_EM){
        E = (double***) malloc(sizeof(double**) * (N_M_interfaces + 1));
		M = (int***) malloc(sizeof(int**) * (N_M_interfaces + 1));
		for(i = 0; i <= N_M_interfaces; ++i){
			E[i] = (double**) malloc(sizeof(double*) * 1);
			*(E[i]) = (double*) malloc(sizeof(double) * M_arr_len[i]);
			M[i] = (int**) malloc(sizeof(int*) * 1);
			*(M[i]) = (int*) malloc(sizeof(int) * M_arr_len[i]);
		}
    }

	//    printf("0: %d\n", Izing::get_seed_C());
    Izing::init_rand_C(my_seed);
//    printf("1: %d\n", Izing::get_seed_C());

// get the initial states; they should be sampled from the distribution in [-L^2; M_0], but they are all set to have M == -L^2 because then they all will fall into the local optimum and almost forget the initial state, so it's almost equivalent to sampling from the proper distribution if 'F(M_0) - F_min >~ T'
    Izing::get_init_states_C(L, Temp, h, N_init_states[0], M_0, states[0], E[0], M[0], &(Nt[0]), &(M_arr_len[0]), to_remember_EM, verbose);
    printf("hi\n");
    printf("Nt = %d\n", Nt[0]);

	double flux0;
	double d_flux0;
	for(i = 0; i <= N_M_interfaces; ++i){
		probs[i] = Izing::process_step(states[i], states[i == N_M_interfaces ? 0 : i+1], E[i], M[i], &(Nt[i]), &(M_arr_len[i]), N_init_states[i], N_init_states[i+1], L, Temp, h, M_interfaces[i], M_interfaces[i+1], i < N_M_interfaces, to_remember_EM, verbose);
		//d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] / probs[i]));
		d_probs[i] = (i == 0 ? 0 : probs[i] / sqrt(N_init_states[i] * (1 - probs[i])));

		if(i == 0){
			// we know that 'probs[0] == 1' because M_0 = -L2-1 for run[0]. Thus we can compute the flux
			flux0 = (double)N_init_states[0] / Nt[0];
			d_flux0 = flux0 / sqrt(Nt[0]);   // TODO: use 'Nt/memory_time' instead of 'Nt'
		}

		if(verbose){
			if(i == 0){
				printf("flux0 = (%e +- %e) 1/step\n", flux0, d_flux0);
			} else {
				printf("-ln(p_%d) = (%lf +- %lf)\n", i, -log(probs[i]), d_probs[i] / probs[i]);   // this assumes p<<1
			}
			if(verbose >= 2){
				printf("\nstate[%d] beginning: ", i);
				for(j = 0; j < (Nt[i] > 10 ? 10 : Nt[i]); ++j)  printf("%d ", states[i][j]);
			}
		}
	}

	double ln_k_AB = log(flux0 * 1);   // flux has units = 1/time; Here, time is in steps, so it's not a problem. But generally speaking it's not clear what time to use here.
	double d_ln_k_AB = Izing::sqr(d_flux0 / flux0);
	for(i = 1; i <= N_M_interfaces; ++i){
		ln_k_AB += log(probs[i]);
		d_ln_k_AB += Izing::sqr(d_probs[i] / probs[i]);   // this assumes p<<1
	}
	d_ln_k_AB = sqrt(d_ln_k_AB);

	printf("-log(k_AB * [1 step]) = (%lf +- %lf)", -ln_k_AB, d_ln_k_AB);

    if(to_remember_EM){
		for(i = 0; i <= N_M_interfaces; ++i){
			free(E[i][0]);   // array data
			free(E[i]);      // the pointer to the array
			free(M[i][0]);
			free(M[i]);
		}
        free(E);   // array of pointer to arrays
        free(M);
    }

	for(i = 0; i <= N_M_interfaces; ++i){
		free(states[i]);
	}
    free(states);
	free(probs);
	free(d_probs);
	free(Nt);
	free(M_arr_len);
	free(M_interfaces);
	free(N_init_states);

    return 0;
}
