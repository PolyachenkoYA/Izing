# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

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
CMAKE_COMMAND = /home/ypolyach/clion-2020.3.2/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /home/ypolyach/clion-2020.3.2/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ypolyach/!Princeton/Izing/Init_state_gen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release

# Include any dependencies generated for this target.
include CMakeFiles/Izing.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Izing.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Izing.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Izing.dir/flags.make

CMakeFiles/Izing.dir/main.cpp.o: CMakeFiles/Izing.dir/flags.make
CMakeFiles/Izing.dir/main.cpp.o: /home/ypolyach/!Princeton/Izing/Init_state_gen/main.cpp
CMakeFiles/Izing.dir/main.cpp.o: CMakeFiles/Izing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Izing.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Izing.dir/main.cpp.o -MF CMakeFiles/Izing.dir/main.cpp.o.d -o CMakeFiles/Izing.dir/main.cpp.o -c /home/ypolyach/!Princeton/Izing/Init_state_gen/main.cpp

CMakeFiles/Izing.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Izing.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ypolyach/!Princeton/Izing/Init_state_gen/main.cpp > CMakeFiles/Izing.dir/main.cpp.i

CMakeFiles/Izing.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Izing.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ypolyach/!Princeton/Izing/Init_state_gen/main.cpp -o CMakeFiles/Izing.dir/main.cpp.s

CMakeFiles/Izing.dir/Izing.cpp.o: CMakeFiles/Izing.dir/flags.make
CMakeFiles/Izing.dir/Izing.cpp.o: /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing.cpp
CMakeFiles/Izing.dir/Izing.cpp.o: CMakeFiles/Izing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Izing.dir/Izing.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Izing.dir/Izing.cpp.o -MF CMakeFiles/Izing.dir/Izing.cpp.o.d -o CMakeFiles/Izing.dir/Izing.cpp.o -c /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing.cpp

CMakeFiles/Izing.dir/Izing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Izing.dir/Izing.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing.cpp > CMakeFiles/Izing.dir/Izing.cpp.i

CMakeFiles/Izing.dir/Izing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Izing.dir/Izing.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ypolyach/!Princeton/Izing/Init_state_gen/Izing.cpp -o CMakeFiles/Izing.dir/Izing.cpp.s

# Object files for target Izing
Izing_OBJECTS = \
"CMakeFiles/Izing.dir/main.cpp.o" \
"CMakeFiles/Izing.dir/Izing.cpp.o"

# External object files for target Izing
Izing_EXTERNAL_OBJECTS =

Izing: CMakeFiles/Izing.dir/main.cpp.o
Izing: CMakeFiles/Izing.dir/Izing.cpp.o
Izing: CMakeFiles/Izing.dir/build.make
Izing: CMakeFiles/Izing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Izing"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Izing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Izing.dir/build: Izing
.PHONY : CMakeFiles/Izing.dir/build

CMakeFiles/Izing.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Izing.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Izing.dir/clean

CMakeFiles/Izing.dir/depend:
	cd /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ypolyach/!Princeton/Izing/Init_state_gen /home/ypolyach/!Princeton/Izing/Init_state_gen /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release /home/ypolyach/!Princeton/Izing/Init_state_gen/cmake-build-release/CMakeFiles/Izing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Izing.dir/depend

