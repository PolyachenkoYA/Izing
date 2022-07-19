//
// Created by ypolyach on 10/27/21.
//

#ifndef IZING_IZING_H
#define IZING_IZING_H

#include <gsl/gsl_rng.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

int copy_state(int *src, int* dst, int L);
int comp_E(int* s, int N, double h, double *E);
int comp_M(int* s, int N, int *M);
int generate_state(int *s, int L, gsl_rng *rng, int mode=1);
int md(int i, int L);
double get_dE(int *s, int L, double h, int ix, int iy);
int get_init_states_C(int L, double Temp, double h, int N0, int M0, int *init_states, int my_seed, double **E, double **M, int *Nt, bool to_remember_EM=0, bool verbose=0);
pybind11::tuple get_init_states(int L, double Temp, double h, int N0, int M0, int my_seed, int verbose, int to_get_EM);
int print_E(double *E, int Nt, char prefix=0, char suffix='\n');
int print_S(int **s, int L, char prefix=0);

#endif //IZING_IZING_H
