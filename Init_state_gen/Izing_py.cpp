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
// py::tuple run_FFS(int L, double Temp, double h, pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces,
//				  int to_remember_timeevol, int init_gen_mode, int interface_mode, std::optional<int> _verbose)
    m.def("run_FFS", &run_FFS,
          "run FFS for the 2D Ising model",
          py::arg("grid_size"),
          py::arg("Temp"),
          py::arg("h"),
          py::arg("N_init_states"),
          py::arg("OP_interfaces"),
		  py::arg("to_remember_timeevol")=0,
		  py::arg("init_gen_mode")=-2,
		  py::arg("interface_mode")=1,
		  py::arg("default_spin_state")=-1,
		  py::arg("verbose")=py::none()
    );

// py::tuple run_bruteforce(int L, double Temp, double h, int Nt_max,
//						 std::optional<int> _OP_to_save_min, std::optional<int> _OP_to_save_max,
//						 std::optional<int> _interface_mode, std::optional<int> _verbose)
	m.def("run_bruteforce", &run_bruteforce,
		  "run Brute-force simulation for the 2D Ising model for Nt_max steps, starting from M=-L^2",
		  py::arg("grid_size"),
		  py::arg("Temp"),
		  py::arg("h"),
		  py::arg("Nt_max"),
		  py::arg("N_spins_up_init")=py::none(),
		  py::arg("to_remember_timeevol")=py::none(),
		  py::arg("OP_A")=py::none(),
		  py::arg("OP_B")=py::none(),
		  py::arg("OP_min")=py::none(),
		  py::arg("OP_max")=py::none(),
		  py::arg("interface_mode")=py::none(),
		  py::arg("default_spin_state")=py::none(),
		  py::arg("verbose")=py::none()
	);

// py::tuple cluster_state(py::array_t<int> state, int default_state)
	m.def("cluster_state", &cluster_state,
		  "Get clusters for a given state",
		  py::arg("state"),
		  py::arg("default_state")=-1,
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
