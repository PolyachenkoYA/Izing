//
// Created by ypolyach on 10/27/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctime>
#include "Izing.h"

namespace py = pybind11;

// input: 11 3.0 0.0 4 -117 1 1 12345
// input: 11 2.0 -0.01 100000 -101 1 1 2
// input: 11 2.0 -0.01 10 121 1 1 2

PYBIND11_MODULE(izing, m)
{
// py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> M_interfaces, int to_get_EM, std::optional<int> _verbose)
    m.def("run_FFS", &run_FFS,
          "run FFS for the 2D Ising model",
          py::arg("grid_size"),
          py::arg("Temp"),
          py::arg("h"),
          py::arg("N_init_states"),
          py::arg("M_interfaces"),
		  py::arg("to_get_EM")=0,
		  py::arg("verbose")=py::none()
    );

// py::tuple run_bruteforce(int L, double Temp, double h, int Nt_max, std::optional<int> _verbose)
	m.def("run_bruteforce", &run_bruteforce,
		  "run Brute-force simulation for the 2D Ising model for Nt_max steps, starting from M=-L^2",
		  py::arg("grid_size"),
		  py::arg("Temp"),
		  py::arg("h"),
		  py::arg("Nt_max"),
		  py::arg("verbose")=py::none()
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
