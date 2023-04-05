cmake_minimum_required(VERSION 3.20)
set(CMAKE_VERBOSE_MAKEFILE on)

project(lattice_gas)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(MATH_LIBS m gsl gslcblas)
#find_library(python3.8 PATHS "$ENV{HOME}/anaconda3/lib" NO_DEFAULT_PATH)
set(PY_LIBS crypt pthread dl util rt python3.8)
#set(Python3_INTERPRETER_ID Anaconda)
#set(PYTHON_INCLUDE_PATH "$ENV{HOME}/anaconda3/include/python3.8")
#set(PYTHON_LIBRARIES "$ENV{HOME}/anaconda3/lib/libpython3.8.so")

find_package(Python3 REQUIRED)
include_directories($ENV{HOME}/anaconda3/include)
include_directories($ENV{HOME}/anaconda3/include/python3.8)
link_directories($ENV{HOME}/anaconda3/lib)

#set(Python3_LIBRARY "$ENV{HOME}/anaconda3/lib")
#set(Python3_INCLUDE_DIR "$ENV{HOME}/anaconda3/include")
# "$ENV{HOME}/anaconda3/include/python3.8"
#find_package(Python3 REQUIRED)

find_package(pybind11 REQUIRED)
add_executable(lattice_gas main.cpp lattice_gas.h lattice_gas.cpp)
target_link_libraries(lattice_gas ${MATH_LIBS} ${PY_LIBS} pybind11::embed)

find_package(pybind11 REQUIRED)
pybind11_add_module(lattice_gas.so lattice_gas.h lattice_gas.cpp lattice_gas_py.cpp)
target_link_libraries(lattice_gas.so PRIVATE ${MATH_LIBS} ${PY_LIBS})
