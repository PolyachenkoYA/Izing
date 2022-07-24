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

	int md(int i, int L);
	template <typename T> T sqr(T x) { return x * x; }

	int print_E(const double *E, int Nt, char prefix=0, char suffix='\n');
	int print_S(const int *s, int L, char prefix);

    int init_rand_C(int my_seed);
    double comp_E(const int* s, int N, double h);
	int comp_M(const int *s, int L);
    int generate_state(int *s, int L, gsl_rng *rng, int mode=1);
    double get_dE(int *s, int L, double h, int ix, int iy);
	int run_FFS_C(double *flux0, double *d_flux0, int L, double Temp, double h, int *states, int *N_init_states, int *Nt,
			  int *M_arr_len, int *M_interfaces, int N_M_interfaces, double *probs, double *d_probs, double **E, int **M,
			  int to_remember_EM, int verbose,
			  py::array_t<int> *_Nt, py::array_t<double> *_probs, py::array_t<double> *_d_probs);
	int get_init_states_C(int L, double Temp, double h, int N_init_states, int *init_states, double **E, int **M, int *Nt, bool to_remember_EM, int verbose);
	int run_state(int *s, int L, double Temp, double h, int M_0, int M_next, double **E, int **M, int *Nt, int *M_arr_len, bool to_remember_EM, int verbose);
	double process_step(int *init_states, int *next_states, double **E, int **M, int *Nt, int *M_arr_len,
				 int N_init_states, int N_next_states, int L, double Temp, double h, int M_0, int M_next,
				 int to_save_next_states, bool to_remember_EM, int verbose,
				 py::array_t<int> *_Nt, py::array_t<double> *_probs, py::array_t<double> *_d_probs);
}

//py::tuple get_init_states(int L, double Temp, double h, int N0, int M_0, int to_get_EM, std::optional<int> _verbose);
py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> M_interfaces, int to_get_EM, std::optional<int> _verbose);
py::int_ init_rand(int my_seed);
py::int_ set_verbose(int new_verbose);
py::int_ get_seed();

#endif //IZING_IZING_H
