//
// Created by ypolyach on 10/27/21.
//

#ifndef IZING_IZING_H
#define IZING_IZING_H

#include <gsl/gsl_rng.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace Izing
{
	extern gsl_rng *rng;
    extern int seed;
    extern int verbose_dafault;

	int print_E(double *E, int Nt, char prefix=0, char suffix='\n');
	int print_S(int *s, int L, char prefix=0);

    int init_rand_C(int my_seed);
    int comp_E(int* s, int N, double h, double *E);
    int comp_M(int* s, int N, int *M);
    int generate_state(int *s, int L, gsl_rng *rng, int mode=1);
    int md(int i, int L);
    double get_dE(int *s, int L, double h, int ix, int iy);
	int get_init_states_C(int L, double Temp, double h, int N0, int M_0, int *init_states, double **E, double **M, int *Nt, int *M_arr_len, bool to_remember_EM, int verbose);
	int run_state(int *s, int L, double Temp, double h, int M_0, int M_next, double **E, double **M, int *Nt, int *M_arr_len, bool to_remember_EM, int verbose);
	double process_step(int *init_states, int *next_states, double **E, double **M, int *Nt, int *M_arr_len, int N_init_states, int L, double Temp, double h, int M_0, int M_next, bool to_remember_EM, int verbose);
}

py::tuple get_init_states(int L, double Temp, double h, int N0, int M_0, int to_get_EM, std::optional<int> _verbose);
py::int_ init_rand(int my_seed);
py::int_ set_verbose(int new_verbose);
py::int_ get_seed();

#endif //IZING_IZING_H
