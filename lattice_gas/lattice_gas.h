//
// Created by ypolyach on 10/27/21.
//

#ifndef IZING_IZING_H
#define IZING_IZING_H

#include <gsl/gsl_rng.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/cast.h>

#include <Python.h>

#include <random>
#include <set>
#include <algorithm>

namespace py = pybind11;
using namespace py::literals;

#define dim 2

#define mode_ID_M 0
#define mode_ID_CS 1
#define N_interface_modes 2

#define save_state_mode_Inside 1
#define save_state_mode_Influx 2

#define N_species 3
#define main_specie_id 1
#define background_specie_id 0

#define gen_init_state_mode_SinglePiece 0
#define gen_init_state_mode_Random -1
#define gen_init_state_mode_Inside -2
#define gen_init_state_mode_Influx -3

#define move_mode_flip 1
#define move_mode_swap 2
#define move_mode_long_swap 3

namespace lattice_gas
{
	extern std::mt19937 *gen_mt19937;
	extern gsl_rng *rng;
    extern int seed;
    extern int verbose_dafault;
	extern int OP_min_default[N_interface_modes];
	extern int OP_max_default[N_interface_modes];
//	extern int OP_peak_default[N_interface_modes];
	extern int OP_step[N_interface_modes];

	int md(int i, int L);
	template <typename T> T sqr(T x) { return x * x; }
	template <typename T> void zero_array(T* v, long N, T v0=0) { for(long i = 0; i < N; ++i) v[i] = v0; }
	template <typename T> T sum_array(T* v, long N) { T s = 0; for(long i = 0; i < N; ++i) s += v[i]; return s; }
	template <typename T> char sgn(T val) { return T(0) <= val ? 1 : -1;	}
	template <typename T> T max(T *v, unsigned long N) {
		T mx = v[0];
		for(unsigned long i = 0; i < N; ++i) if(mx < v[i]) mx = v[i];
		return  mx;
	}

	void print_M(const int *M, long Nt, char prefix=0, char suffix='\n');
	void print_E(const double *E, long Nt, char prefix=0, char suffix='\n');
	int print_S(const int *s, int L, char prefix);
	int E_is_valid(const double *E, const double E1, const double E2, int N, int k=0, char prefix=0);
	void print_e_matrix(const double *e);
	void print_mu_vector(const double *mu);
	int state_is_valid(const int *s, int L, int k=0, char prefix=0);

	int run_FFS_C(int move_mode, double *flux0, double *d_flux0, int L, const double *e, const double *mu, int *states,
				  int *N_init_states, long *Nt, long *Nt_OP_saved, long stab_step,
				  long *OP_arr_len, int *OP_interfaces, int N_OP_interfaces, double *probs, double *d_probs, double **E, int **M,
				  int **biggest_cluster_sizes, int **time, int verbose, int init_gen_mode, int interface_mode,
				  const int *init_state, int to_use_smart_swap);
	int run_bruteforce_C(int move_mode, int L, const double *e, const double *mu, long *time_total, int N_states, int *states,
						 long *OP_arr_len, long *Nt, long *Nt_OP_saved, double **E, int **M, int **biggest_cluster_sizes, int **h_A, int **time,
						 int interface_mode, int OP_A, int OP_B, int to_cluster,
						 int OP_min_stop_state, int OP_max_stop_state, int *N_states_done,
						 int OP_min_save_state, int OP_max_save_state, int save_state_mode,
						 int N_spins_up_init, int verbose, long Nt_max, int *N_tries, int to_save_final_state,
						 int to_regenerate_init_state, long save_states_stride, int to_use_smart_swap);
	double process_step(int move_mode, int *init_states, int *next_states, double **E, int **M, int **biggest_cluster_sizes, int **time,
						long *Nt, long *Nt_OP_saved, long *OP_arr_len, int N_init_states, int N_next_states,
						int L, const double *e, const double *mu, int OP_0, int OP_next,
						int interfaces_mode, int to_use_smart_swap, int verbose);
	int run_state(int move_mode, int *s, int *OP_current, int L, const double *e, const double *mu, long *time_total,
				  int OP_0, int OP_next, double **E, int **M, int **biggest_cluster_sizes, int **h_A, int **time,
				  int *cluster_element_inds, int *cluster_sizes, int *cluster_types, int *is_checked, long *Nt, long *Nt_OP_saved,
				  long *OP_arr_len, int interfaces_mode, int to_use_smart_swap, int verbose, int to_cluster=1, long Nt_max=-1,
				  int* states_to_save=nullptr, int *N_states_saved=nullptr, int N_states_to_save=-1,
				  int OP_min_save_state=0, int OP_max_save_state=0,
				  int save_state_mode=save_state_mode_Inside, int OP_A=0, int OP_B=0, long save_states_stride=1);
	int get_init_states_C(int move_mode, int L, const double *e, const double *mu, long *time_total, int N_init_states,
						  int *init_states, int mode, int OP_thr_save_state, long stab_step,
						  int interface_mode, int OP_A, int OP_B,
						  double **E, int **M, int **biggest_cluster_size, int **h_A, int **time,
						  long *Nt, long *Nt_OP_saved, long *OP_arr_len, const int *init_state, int to_use_smart_swap,
						  int verbose);
	int get_equilibrated_state(int move_mode, int L, const double *e, const double *mu, int *state, int *N_states_done,
							   int interface_mode, int OP_A, int OP_B, long stab_step, const int *init_state,
							   int to_use_smart_swap, int verbose);


	int init_rand_C(int my_seed);
	int get_max_CS(int *state, int L);
	double comp_E(const int* state, int L, const double *e, const double *mu);
	int comp_M(const int *s, int L);
	int generate_state(int *s, int L, int mode, int interface_mode, int verbose);
	double new_spin_energy(int L, const double *e, const double *mu, const int *s_neibs, int s_new);
	void get_spin_with_neibs(const int *state, int L, int ix, int iy, int *s_group);
	int swap_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new,
				  double *dE, const std::set< int > *swap_positions);
//	int long_swap_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new, double *dE);
	int long_swap_move(const int *state, uint L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new, double *dE);
	int flip_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *s_new, double *dE);
	double swap_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int ix_new, int iy_new);
	double flip_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int s_new);
	void set_OP_default(int L2);
	int get_OP_from_spinsup(int N_spins_up, int L2, int interface_mode);

	bool is_potential_swap_position(int *state, int L, int ix, int iy);
	void update_neib_potpos(int *state, int L, int ix, int iy, std::set< int > *positions);
	void find_potential_swaps(int *state, int L, std::set< int > *positions);


	void cluster_state_C(const int *s, int L, int *cluster_element_inds, int *cluster_sizes, int *cluster_types, int *N_clusters, int *is_checked);
	int add_to_cluster(const int* s, int L, int* is_checked, int* cluster, int* cluster_size, int pos, int cluster_label, int cluster_specie);
//	int is_infinite_cluster(const int* cluster, const int* cluster_size, int L, char *present_rows, char *present_columns);
	void uncheck_state(int *is_checked, int N);
	void clear_cluster(int* cluster, int *cluster_size);
	void clear_clusters(int *clusters, int *cluster_sizes, int *N_clusters);
}

//py::tuple get_init_states(int L, double Temp, double h, int N0, int M_0, int to_get_EM, std::optional<int> _verbose);
py::tuple run_FFS(int move_mode, int L, py::array_t<double> e, py::array_t<double> mu,
				  pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces,
				  int to_remember_timeevol, int init_gen_mode, int interface_mode, long stab_step,
				  std::optional< pybind11::array_t<int> > _init_state, int to_use_smart_swap,
				  std::optional<int> _verbose);
py::tuple run_bruteforce(int move_mode, int L, py::array_t<double> e, py::array_t<double> mu, long Nt_max,
						 long N_saved_states_max, long save_states_stride, long stab_step,
						 std::optional<int> _N_spins_up_init, std::optional<int> _to_remember_timeevol,
						 std::optional<int> _OP_A, std::optional<int> _OP_B,
						 std::optional<int> _OP_min_save_state, std::optional<int> _OP_max_save_state,
						 std::optional<int> _OP_min, std::optional<int> _OP_max,
						 std::optional<int> _interface_mode,
						 std::optional< pybind11::array_t<int> > _init_state,
						 int to_use_smart_swap,
						 std::optional<int> _verbose);
int compute_hA(py::array_t<int> *h_A, int *OP, long Nt, int OP_A, int OP_B);
py::tuple cluster_state(py::array_t<int> state, std::optional<int> _verbose);
void print_state(py::array_t<int> state);
py::int_ init_rand(int my_seed);
py::int_ set_verbose(int new_verbose);
py::int_ get_verbose();
py::int_ get_seed();
void print_possible_move_modes();
py::dict get_move_modes();

template <typename T> T cmpfunc_inrc (const void * a, const void * b) {	return ( *(T*)a - *(T*)b ); }
template <typename T> T cmpfunc_decr (const void * a, const void * b) {	return -cmpfunc_inrc<T>(a, b); }

#endif //IZING_IZING_H

