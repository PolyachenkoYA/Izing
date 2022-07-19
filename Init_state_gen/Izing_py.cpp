//
// Created by ypolyach on 10/27/21.
//

#include <pybind11/pybind11.h>
#include <ctime>
#include "Izing.h"

namespace py = pybind11;

PYBIND11_MODULE(izing, m)
{
//pybind11::tuple get_init_states(int L, double Temp, double h, int N0, int M0, int my_seed, int verbose, int to_get_EM)
    m.def("get_init_states", &get_init_states,
        "get a tuple (init_states[N0 * L*L], E[Nt], M[Nt]); init with my_seed (default=time(NULL))",
        py::arg("grid_size"),
        py::arg("Temp"),
        py::arg("h"),
        py::arg("N_init_states"),
        py::arg("M0"),
        py::arg("my_seed")=time(nullptr),
        py::arg("verbose")=0,
        py::arg("to_get_EM")=0
    );
}

// # double prob_for_p(double p, int box_size, int N_iter, int my_seed=time(NULL), bool verbose=0);
