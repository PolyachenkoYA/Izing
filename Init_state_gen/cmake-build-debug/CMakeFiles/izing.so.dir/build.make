# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

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
CMAKE_COMMAND = /home/ypolyach/clion-2020.3.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/ypolyach/clion-2020.3.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ypolyach/!Princeton/Izing/Init_state_gen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/izing.so.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/izing.so.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/izing.so.dir/flags.make

CMakeFiles/izing.so.dir/Izing.cpp.o: CMakeFiles/izing.so.dir/flags.make
CMakeFiles/izing.so.dir/Izing.cpp.o: ../Izing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/izing.so.dir/Izing.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/izing.so.dir/Izing.cpp.o -c /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing.cpp

CMakeFiles/izing.so.dir/Izing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/izing.so.dir/Izing.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing.cpp > CMakeFiles/izing.so.dir/Izing.cpp.i

CMakeFiles/izing.so.dir/Izing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/izing.so.dir/Izing.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing.cpp -o CMakeFiles/izing.so.dir/Izing.cpp.s

CMakeFiles/izing.so.dir/Izing_py.cpp.o: CMakeFiles/izing.so.dir/flags.make
CMakeFiles/izing.so.dir/Izing_py.cpp.o: ../Izing_py.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/izing.so.dir/Izing_py.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/izing.so.dir/Izing_py.cpp.o -c /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing_py.cpp

CMakeFiles/izing.so.dir/Izing_py.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/izing.so.dir/Izing_py.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing_py.cpp > CMakeFiles/izing.so.dir/Izing_py.cpp.i

CMakeFiles/izing.so.dir/Izing_py.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/izing.so.dir/Izing_py.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing_py.cpp -o CMakeFiles/izing.so.dir/Izing_py.cpp.s

# Object files for target izing.so
izing_so_OBJECTS = \
"CMakeFiles/izing.so.dir/Izing.cpp.o" \
"CMakeFiles/izing.so.dir/Izing_py.cpp.o"

# External object files for target izing.so
izing_so_EXTERNAL_OBJECTS =

izing.so.cpython-38-x86_64-linux-gnu.so: CMakeFiles/izing.so.dir/Izing.cpp.o
izing.so.cpython-38-x86_64-linux-gnu.so: CMakeFiles/izing.so.dir/Izing_py.cpp.o
izing.so.cpython-38-x86_64-linux-gnu.so: CMakeFiles/izing.so.dir/build.make
izing.so.cpython-38-x86_64-linux-gnu.so: CMakeFiles/izing.so.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module izing.so.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/izing.so.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/izing.so.dir/build: izing.so.cpython-38-x86_64-linux-gnu.so
.PHONY : CMakeFiles/izing.so.dir/build

CMakeFiles/izing.so.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/izing.so.dir/cmake_clean.cmake
.PHONY : CMakeFiles/izing.so.dir/clean

CMakeFiles/izing.so.dir/depend:
	cd /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ypolyach/!Princeton/Izing/Init_state_gen /home/ypolyach/!Princeton/Izing/Init_state_gen /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-debug/CMakeFiles/izing.so.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/izing.so.dir/depend

