//
// Created by ypolyach on 10/27/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctime>
#include "lattice_gas.h"

namespace py = pybind11;

PYBIND11_MODULE(lattice_gas, m)
{
// py::tuple run_FFS(int L, py::array_t<double> e, py::array_t<double> mu, pybind11::array_t<int> N_init_states, pybind11::array_t<int> OP_interfaces,
//				  int to_remember_timeevol, int init_gen_mode, int interface_mode,
//				  std::optional<int> _verbose)
    m.def("run_FFS", &run_FFS,
          "run FFS for the 2D lattice gas model",
          py::arg("grid_size"),
          py::arg("e"),
          py::arg("mu"),
          py::arg("N_init_states"),
          py::arg("OP_interfaces"),
		  py::arg("to_remember_timeevol")=0,
		  py::arg("init_gen_mode")=gen_init_state_mode_Inside,
		  py::arg("interface_mode")=mode_ID_CS,
		  py::arg("verbose")=py::none()
    );

// py::tuple run_bruteforce(int L, double **e, double *mu, long Nt_max, long N_saved_states_max,
//						 std::optional<int> _N_spins_up_init, std::optional<int> _to_remember_timeevol,
//						 std::optional<int> _OP_A, std::optional<int> _OP_B,
//						 std::optional<int> _OP_min, std::optional<int> _OP_max,
//						 std::optional<int> _interface_mode,
//						 std::optional<int> _verbose)
	m.def("run_bruteforce", &run_bruteforce,
		  "run Brute-force simulation for the 2D lattice gas model for Nt_max steps, starting from ~equilibrated state",
		  py::arg("grid_size"),
		  py::arg("e"),
		  py::arg("mu"),
		  py::arg("Nt_max"),
		  py::arg("N_saved_states_max"),
		  py::arg("save_states_stride")=1,
		  py::arg("N_spins_up_init")=py::none(),
		  py::arg("to_remember_timeevol")=py::none(),
		  py::arg("OP_A")=py::none(),
		  py::arg("OP_B")=py::none(),
		  py::arg("OP_min")=py::none(),
		  py::arg("OP_max")=py::none(),
		  py::arg("interface_mode")=py::none(),
		  py::arg("verbose")=py::none()
	);

// py::tuple cluster_state(py::array_t<int> state, int default_state)
	m.def("cluster_state", &cluster_state,
		  "Get clusters for a given state",
		  py::arg("state"),
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

// void print_state(py::array_t<int> state, char prefix)
	m.def("print_state", &print_state,
		  "Returns the current seed used for the last GSL random initiation",
		  py::arg("state")
	);
}
