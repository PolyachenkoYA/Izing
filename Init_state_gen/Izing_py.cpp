//
// Created by ypolyach on 10/27/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctime>
#include "Izing.h"

namespace py = pybind11;

PYBIND11_MODULE(izing, m)
{
// py::tuple get_init_states(int L, double Temp, double h, int N0, int M0, int verbose, int to_get_EM)
    m.def("get_init_states", &get_init_states,
          "get a tuple (init_states[N0 * L*L], E[Nt], M[Nt]); init with my_seed (default=time(NULL))",
          py::arg("grid_size"),
          py::arg("Temp"),
          py::arg("h"),
          py::arg("N_init_states"),
          py::arg("M0"),
          py::arg("verbose")=py::none(),
          py::arg("to_get_EM")=0
    );

// py::int_ init_rand(int my_seed)
    m.def("init_rand", &init_rand,
        "Initialize GSL rand generator",
        py::arg("seed")=time(nullptr)
    );

// py::int_ set_verbose(int new_verbose)
    m.def("set_verbose", &set_verbose,
          "Set default verbose behaviour",
          py::arg("new_verbose")
    );

// py::int_ set_verbose(int new_verbose)
	m.def("get_seed", &get_seed,
		  "Returns the current seed used for the last GSL random initiation"
	);
}
