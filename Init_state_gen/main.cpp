#include <iostream>

#include "Izing.h"

int main(int argc, char** argv) {
    if(argc != 9){
        printf("usage:\n%s   L   T   h   N0   M0   to_remember_EM   verbose   seed\n", argv[0]);
        return 1;
    }

    int L = atoi(argv[1]);
    double Temp = atof(argv[2]);
    double h =  atof(argv[3]);
    int N0 = atoi(argv[4]);
    int M0 = atoi(argv[5]);
    int to_remember_EM = atoi(argv[6]);
    int verbose = atoi(argv[7]);
    int my_seed = atoi(argv[8]);
    int i;

    int *init_states = (int*) malloc(sizeof(int) * N0 * L*L);
    int Nt;
    double **E;
    double **M;
    if(to_remember_EM){
        E = (double**) malloc(sizeof(double*) * 1);
        *E = (double*) malloc(sizeof(double) * N0);
        M = (double**) malloc(sizeof(double*) * 1);
        *M = (double*) malloc(sizeof(double) * N0);

    }
//    printf("0: %d\n", Izing::get_seed_C());
    Izing::init_rand_C(my_seed);
//    printf("1: %d\n", Izing::get_seed_C());
    Izing::get_init_states_C(L, Temp, h, N0, M0, init_states, E, M, &Nt, to_remember_EM, verbose);
    printf("hi\n");
    printf("Nt = %d\n", Nt);

//    char filename[80];
//    sprintf(filename, "N%d_T%lf_Nt%d_seed%d.dat", N, Temp, Nt, my_seed);
//
//    FILE *output_file;
//    output_file = fopen(filename, "w");
//    for(i = 0; i < Nt; ++i) {
//        fprintf(output_file, "%lf ", E[i]);
//    }
//    fclose(output_file);

    if(to_remember_EM){
        for(i = 0; i < fmin(10, Nt); ++i)  printf("%lf ", (*E)[i]);

        free(*E);
        free(E);
        free(*M);
        free(M);
    }

    printf("\nI:");
    for(i = 0; i < 10; ++i)  printf("%d ", init_states[i]);
    free(init_states);

    return 0;
}
