# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yp1065/Izing/lattice_gas

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yp1065/Izing/lattice_gas/build

# Include any dependencies generated for this target.
include CMakeFiles/lattice_gas.so.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/lattice_gas.so.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/lattice_gas.so.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lattice_gas.so.dir/flags.make

CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o: CMakeFiles/lattice_gas.so.dir/flags.make
CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o: /home/yp1065/Izing/lattice_gas/lattice_gas.cpp
CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o: CMakeFiles/lattice_gas.so.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yp1065/Izing/lattice_gas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o -MF CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o.d -o CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o -c /home/yp1065/Izing/lattice_gas/lattice_gas.cpp

CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yp1065/Izing/lattice_gas/lattice_gas.cpp > CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.i

CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yp1065/Izing/lattice_gas/lattice_gas.cpp -o CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.s

CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o: CMakeFiles/lattice_gas.so.dir/flags.make
CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o: /home/yp1065/Izing/lattice_gas/lattice_gas_py.cpp
CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o: CMakeFiles/lattice_gas.so.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yp1065/Izing/lattice_gas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o -MF CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o.d -o CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o -c /home/yp1065/Izing/lattice_gas/lattice_gas_py.cpp

CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yp1065/Izing/lattice_gas/lattice_gas_py.cpp > CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.i

CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yp1065/Izing/lattice_gas/lattice_gas_py.cpp -o CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.s

# Object files for target lattice_gas.so
lattice_gas_so_OBJECTS = \
"CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o" \
"CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o"

# External object files for target lattice_gas.so
lattice_gas_so_EXTERNAL_OBJECTS =

lattice_gas.so.cpython-39-x86_64-linux-gnu.so: CMakeFiles/lattice_gas.so.dir/lattice_gas.cpp.o
lattice_gas.so.cpython-39-x86_64-linux-gnu.so: CMakeFiles/lattice_gas.so.dir/lattice_gas_py.cpp.o
lattice_gas.so.cpython-39-x86_64-linux-gnu.so: CMakeFiles/lattice_gas.so.dir/build.make
lattice_gas.so.cpython-39-x86_64-linux-gnu.so: CMakeFiles/lattice_gas.so.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yp1065/Izing/lattice_gas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module lattice_gas.so.cpython-39-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lattice_gas.so.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/yp1065/Izing/lattice_gas/build/lattice_gas.so.cpython-39-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/lattice_gas.so.dir/build: lattice_gas.so.cpython-39-x86_64-linux-gnu.so
.PHONY : CMakeFiles/lattice_gas.so.dir/build

CMakeFiles/lattice_gas.so.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lattice_gas.so.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lattice_gas.so.dir/clean

CMakeFiles/lattice_gas.so.dir/depend:
	cd /home/yp1065/Izing/lattice_gas/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yp1065/Izing/lattice_gas /home/yp1065/Izing/lattice_gas /home/yp1065/Izing/lattice_gas/build /home/yp1065/Izing/lattice_gas/build /home/yp1065/Izing/lattice_gas/build/CMakeFiles/lattice_gas.so.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lattice_gas.so.dir/depend
