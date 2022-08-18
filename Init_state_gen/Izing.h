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

#define N_interface_modes 2
#define mode_ID_M 1
#define mode_ID_CS 2

namespace Izing
{
	extern gsl_rng *rng;
    extern int seed;
    extern int verbose_dafault;
	extern int OP_min_default[N_interface_modes];
	extern int OP_max_default[N_interface_modes];
	extern int OP_peak_default[N_interface_modes];
	extern int OP_step[N_interface_modes];

	int md(int i, int L);
	template <typename T> T sqr(T x) { return x * x; }
	template <typename T> void zero_array(T* v, int N, T v0=0) { for(int i = 0; i < N; ++i) v[i] = v0; }
//	template <typename T> char sgn(T val) { return (T(0) < val) - (val < T(0));	}
	template <typename T> char sgn(T val) { return T(0) <= val ? 1 : -1;	}
	template <typename T> T max(T *v, unsigned long N) {
		T mx = v[0];
		for(unsigned long i = 0; i < N; ++i) if(mx < v[i]) mx = v[i];
		return  mx;
	}

	void print_M(const int *M, int Nt, char prefix=0, char suffix='\n');
	void print_E(const double *E, int Nt, char prefix=0, char suffix='\n');
	int print_S(const int *s, int L, char prefix);
	int E_is_valid(const double *E, const double E1, const double E2, int N, int k=0, char prefix=0);
	int state_is_valid(const int *s, int L, int k=0, char prefix=0);

	int run_FFS_C(double *flux0, double *d_flux0, int L, double Temp, double h, int *states, int *N_init_states, int *Nt,
				  int *OP_arr_len, int *OP_interfaces, int N_OP_interfaces, double *probs, double *d_probs, double **E, int **M,
				  int **biggest_cluster_sizes, int to_remember_EM, int verbose, int init_gen_mode, int interfaces_mode,
				  char default_spin_state);
	int run_bruteforce_C(int L, double Temp, double h, int N_states, int *states,
						 int *OP_arr_len, int *Nt, double **E, int **M, int **biggest_cluster_sizes,
						 int to_remember_timeevol, int interface_mode, char default_spin_state,
						 int OP_min_stop_state, int OP_max_stop_state, int *N_states_done,
						 int OP_min_save_state, int OP_max_save_state, int verbose, int Nt_max);
	double process_step(int *init_states, int *next_states, double **E, int **M, int **biggest_cluster_sizes, int *Nt, int *OP_arr_len,
						int N_init_states, int N_next_states, int L, double Temp, double h, int OP_0, int OP_next,
						int to_save_next_states, bool to_remember_EM, int interfaces_mode, char default_spin_state,
						int verbose);
	int run_state(int *s, int L, double Temp, double h, int OP_0, int OP_next, double **E, int **M, int **biggest_cluster_sizes,
				  int *cluster_element_inds, int *cluster_sizes, int *is_checked, int *Nt, int *OP_arr_len, bool to_remember_EM,
				  int interfaces_mode, char default_spin_state, int verbose, int Nt_max=-1, int* states_to_save= nullptr,
				  int *N_states_saved= nullptr, int N_states_to_save=-1, int OP_min_save_state=0, int OP_max_save_state=0);
	int get_init_states_C(int L, double Temp, double h, int N_init_states, int *init_states, int mode, int OP_thr_save_state,
						  int interface_mode, char default_spin_state, int verbose);

	int init_rand_C(int my_seed);
	double comp_E(const int* s, int N, double h);
	int comp_M(const int *s, int L);
	int generate_state(int *s, int L, int mode, int interface_mode, int default_spin_state);
	int get_flip_point(int *s, int L, double h, double Temp, int *ix, int *iy, double *dE);
	double get_dE(int *s, int L, double h, int ix, int iy);
	void set_OP_default(int L2);

	void cluster_state(const int *s, int L, int *cluster_element_inds, int *cluster_sizes, int *N_clusters, int *is_checked, int default_state);
	int add_to_cluster(const int* s, int L, int* is_checked, int* cluster, int* cluster_size, int pos, int cluster_label, int default_state);
	int is_infinite_cluster(const int* cluster, const int* cluster_size, int L, char *present_rows, char *present_columns);
	void uncheck_state(int *is_checked, int N);
	void clear_cluster(int* cluster, int *cluster_size);
	void clear_clusters(int *clusters, int *cluster_sizes, int *N_clusters);
}

//py::tuple get_init_states(int L, double Temp, double h, int N0, int M_0, int to_get_EM, std::optional<int> _verbose);
py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states,
				  pybind11::array_t<int> OP_interfaces, int to_remember_timeevol, int init_gen_mode, int interface_mode,
				  int default_spin_state, std::optional<int> _verbose);
py::tuple run_bruteforce(int L, double Temp, double h, int Nt_max,
						 std::optional<int> _OP_A, std::optional<int> _OP_B,
						 std::optional<int> _OP_min, std::optional<int> _OP_max,
						 std::optional<int> _interface_mode, std::optional<int> _default_spin_state,
						 std::optional<int> _verbose);
int compute_hA(py::array_t<int> *h_A, int *OP, int Nt, int OP_A, int OP_B);
py::int_ init_rand(int my_seed);
py::int_ set_verbose(int new_verbose);
py::int_ get_seed();

#endif //IZING_IZING_H
