//
// Created by ypolyach on 10/27/21.
//

#include <gsl/gsl_rng.h>
#include <cmath>
#include <optional>

#include <pybind11/pytypes.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "Izing.h"

py::tuple get_init_states(int L, double Temp, double h, int N0, int M0, std::optional<bool> _verbose, int to_get_EM)
{
    auto init_states = py::array_t<int>(N0 * L*L);
    py::buffer_info init_states_info = init_states.request();
    int *init_states_ptr = static_cast<int *>(init_states_info.ptr);
    int Nt;

    bool verbose = (_verbose.has_value() ? _verbose.value() : Izing::verbose_dafault);

    double **_E;
    double **_M;
    if(to_get_EM){
        _E = (double**) malloc(sizeof(double*) * 1);
        *_E = (double*) malloc(sizeof(double) * N0);
        _M = (double**) malloc(sizeof(double*) * 1);
        *_M = (double*) malloc(sizeof(double) * N0);
    }

    Izing::get_init_states_C(L, Temp, h, N0, M0, init_states_ptr, _E, _M, &Nt, to_get_EM, verbose);
    printf("Nt = %d\n", Nt);

    py::array_t<double> E;
    py::array_t<double> M;
    if(to_get_EM){
        E = py::array_t<double>(Nt);
        M = py::array_t<double>(Nt);
        py::buffer_info E_info = E.request();
        py::buffer_info M_info = M.request();
        double *E_ptr = static_cast<double *>(E_info.ptr);
        double *M_ptr = static_cast<double *>(M_info.ptr);
        memcpy(E_ptr, *_E, sizeof(double) * Nt);
        memcpy(M_ptr, *_M, sizeof(double) * Nt);

        free(*_E);
        free(_E);
        free(*_M);
        free(_M);
    }


//    print_E(E_ptr, Nt, 'P');
    py::tuple res = py::make_tuple(init_states, E, M);
    return res;
}

py::int_ init_rand(int my_seed)
{
    Izing::init_rand_C(my_seed);
    return 0;
}

py::int_ set_verbose(int new_verbose)
{
    Izing::verbose_dafault = new_verbose;
    return 0;
}

namespace Izing
{
    gsl_rng *rng;
    int seed;
    int verbose_dafault;

    int get_seed_C()
    {
        return seed;
    }

    int init_rand_C(int my_seed)
    {
        // initialize random generator
        gsl_rng_env_setup();
        const gsl_rng_type* T = gsl_rng_default;
        rng = gsl_rng_alloc(T);
        gsl_rng_set(rng, my_seed);
//    srand(my_seed);

        seed = my_seed;
        return 0;
    }

    int copy_state(int *src, int* dst, int N)
    {
        memcpy(dst, src, sizeof(int) * N);
//    for(int i = 0; i < N; ++i)
//        dst[i] = src[i];
        return 0;
    }

    int comp_M(int *s, int L, int *M)
    {
        int i, j;
        int _M = 0;
        for(i = 0; i < L; ++i){
            for(j = 0; j < L; ++j){
                _M += s[i*L + j];
            }
        }

        *M = _M;
        return 0;
    }

    int comp_E(int* s, int L, double h, double *E)
    {
        int i, j;
        double _E = 0;
        for(i = 0; i < L-1; ++i){
            for(j = 0; j < L-1; ++j){
                _E += s[i*L + j] * (s[(i+1)*L + j] + s[i*L + (j+1)]);
            }
            _E += s[i*L + (L-1)] * (s[(i+1)*L + (L-1)] + s[i*L + 0]);
        }
        for(j = 0; j < L-1; ++j){
            _E += s[(L-1)*L + j] * (s[0*L + j] + s[(L-1)*L + (j+1)]);
        }
        _E += s[(L-1)*L + (L-1)] * (s[0*L + (L-1)] + s[(L-1)*L + 0]);

        int _M;
        comp_M(s, L, &_M);

        *E = h * (_M) - _E; // J > 0 -> {++ > +-} -> we need to *(-1) because we search for a minimum
        // energy is measured in [E]=J, so J==1

        return 0;
    }

    int generate_state(int *s, int L, gsl_rng *rng, int mode)
    {
        int i, j;

        if(mode == 0){ // random
            for(i = 0; i < L; ++i){
                for(j = 0; j < L; ++j) {
                    s[i*L + j] = gsl_rng_uniform_int(rng, 2) * 2 - 1;
                }
            }
        } else if(mode == 1) { // ordered
            for(i = 0; i < L; ++i){
                for(j = 0; j < L; ++j) {
                    s[i*L + j] = -1;
                }
            }
        }

        return 0;
    }

    int md(int i, int L)
    {
        return i >= 0 ? (i < L ? i : 0) : (L-1);   // i mod L for i \in [-1; L]
    }

    double get_dE(int *s, int L, double h, int ix, int iy)
// the difference between the current energy and energy with s[ix][iy] flipped
    {
        return 2 * s[ix*L + iy] * (s[md(ix + 1, L)*L + iy]
                                   + s[ix*L + md(iy + 1, L)]
                                   + s[md(ix - 1, L)*L + iy]
                                   + s[ix*L + md(iy - 1, L)] + h);
        // units [E] = J, so spins are just 's', not s*J
    }

    int get_flip_point(int *s, int L, double h, double Temp, int *ix, int *iy, double *dE, gsl_rng *rng)
    {
        do{
            *ix = gsl_rng_uniform_int(rng, L);
            *iy = gsl_rng_uniform_int(rng, L);

            dE[0] = get_dE(s, L, h, *ix, *iy);
        }while(!(dE[0] <= 0 ? 1 : (gsl_rng_uniform(rng) < exp(- dE[0] / Temp) ? 1 : 0)));

        return 0;
    }

    int get_init_states_C(int L, double Temp, double h, int N0, int M0, int *init_states, double **E, double **M, int *Nt, bool to_remember_EM, bool verbose)
// N0 - number of initial states with M[ti] = M0
// M[ti-1] is not valid since we are doing MC, so there are no velocities and thus no memory of previous states.

// init_states have to be pre-allocated (since it's possible to know its size in advance)
// E and M will be allocated (and possibly extended) inside the function
    {
        int L2 = L*L;
        int i;

        // create the state which will be evolving and sampling the A-region
        int *s;
        int M_current;
        double E_current;
        s = (int*) malloc(sizeof(int) * (L2));
        generate_state(s, L, Izing::rng, 1); // allocate all spins = -1
        comp_E(s, L, h, &E_current); // remember the 1st energy
        comp_M(s, L, &M_current); // remember the 1st M

        int ix, iy;
        double dE;
//        int do_flip;
        int N_init_states = 0;
        int M_arr_len = N0;
        i = 0;
        while(N_init_states < N0){
            if(to_remember_EM){
                if(i >= M_arr_len){ // double the size of the time-index
                    M_arr_len *= 2;
                    *E = (double*) realloc (*E, sizeof(double) * M_arr_len);
                    *M = (double*) realloc (*M, sizeof(double) * M_arr_len);
                }
                (*M)[i] = M_current;
                (*E)[i] = E_current;
            }
//            printf("E=%5.2lf, M=%5.2lf\n", E[i], M[i]);
            if(M_current == M0){
                // We are doing MC (not MD), so there is no 'velocity' of the system,
                // so we don't need to consider M[i-1] to see that the system came from 'M < M0'
                copy_state(s, &(init_states[N_init_states * L2]), L2);
                ++N_init_states;
                if(verbose){
                    printf("N_states = %d\n", N_init_states);
                }
            }

            get_flip_point(s, L, h, Temp, &ix, &iy, &dE, Izing::rng);
            ++i;
            s[ix*L + iy] *= -1;
            M_current += 2 * s[ix*L + iy];
            E_current += dE;
        }
        *Nt = i;

        if(verbose){
            printf("cleaning memory\n");
        }
        free(s);
//        gsl_rng_free(Izing::rng);

        if(verbose){
            if(to_remember_EM){
                print_E(*E, fmin(10, *Nt), 'E');
            }
            printf("exiting from get_init_states()\n");
            printf("Nt_in = %d\n", *Nt);
        }
        return 0;
    }

//int run_state_C(int L, double Temp, int **s_init, int S_next, int S_0, int my_seed, double *E, double *M)
//{
//    int i;
//    //double *E = (double*) malloc(sizeof(double) * Nt);
//
////    E[0] = 123;
////    print_E(E, Nt, 'c');
////    printf("L = %d\n", L);
//
//    gsl_rng_env_setup();
//    const gsl_rng_type* T = gsl_rng_default;
//    gsl_rng* rng = gsl_rng_alloc(T);
//    gsl_rng_set(rng, my_seed);
//
//    int **s = (int**) malloc(sizeof(int*) * L);
//    for(i = 0; i < L; ++i){
//        s[i] = (int*) malloc(sizeof(int) * L);
//    }
//
//    generate_state(s, L, rng, 1);
//    comp_E(s, L, &(E[0])); // &(E[0]) == E
//    comp_M(s, L, &(M[0]));
//
////    print_E(E, Nt, '0');
//
//    int ix, iy;
//    double dE;
//    int do_flip;
//    for(i = 1; i < Nt; ++i){
//        ix = gsl_rng_uniform_int(rng, L);
//        iy = gsl_rng_uniform_int(rng, L);
//
//        dE = get_dE(s, L, ix, iy);
//        do_flip = (dE <= 0 ? 1 : (gsl_rng_uniform(rng) < exp(- dE / Temp) ? 1 : 0));
////        printf("%d, %lf\n", do_flip, dE);
////        print_S(s, L, '0' + (i % 10));
//        if(do_flip){
//            M[i] = M[i-1] - 2 * s[ix][iy];
//            E[i] = E[i-1] + dE;
//            s[ix][iy] *= -1;
//        } else {
//            E[i] = E[i-1];
//            M[i] = M[i-1];
//        }
//
////        print_E(E, Nt, '0' + (i % 10));
//    }
//
//    for(i = 0; i < L; ++i) free(s[i]);
//    free(s);
//    gsl_rng_free (rng);
//
////    print_E(E, Nt, 'C');
//
//    return 0;
//}

    int print_E(double *E, int Nt, char prefix, char suffix)
    {
        if(prefix > 0){
            printf("%c\n", prefix);
        }

        for(int i = 0; i < Nt; ++i){
            printf("%lf ", E[i]);
        }

        if(suffix > 0){
            printf("%c", suffix);
        }

        return 0;
    }

    int print_S(int **s, int L, char prefix)
    {
        int i, j;

        if(prefix > 0){
            printf("%c\n", prefix);
        }

        for(i = 0; i < L; ++i){
            for(j = 0; j < L; ++j){
                printf("%2d", s[i][j]);
            }
            printf("\n");
        }

        return 0;
    }
}

