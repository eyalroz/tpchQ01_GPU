# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build

# Include any dependencies generated for this target.
include CMakeFiles/q1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/q1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/q1.dir/flags.make

CMakeFiles/q1.dir/q1.cpp.o: CMakeFiles/q1.dir/flags.make
CMakeFiles/q1.dir/q1.cpp.o: ../q1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/q1.dir/q1.cpp.o"
	/opt/gcc-5.4.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/q1.dir/q1.cpp.o -c /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/q1.cpp

CMakeFiles/q1.dir/q1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/q1.dir/q1.cpp.i"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/q1.cpp > CMakeFiles/q1.dir/q1.cpp.i

CMakeFiles/q1.dir/q1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/q1.dir/q1.cpp.s"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/q1.cpp -o CMakeFiles/q1.dir/q1.cpp.s

CMakeFiles/q1.dir/tpch_kit.cpp.o: CMakeFiles/q1.dir/flags.make
CMakeFiles/q1.dir/tpch_kit.cpp.o: ../tpch_kit.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/q1.dir/tpch_kit.cpp.o"
	/opt/gcc-5.4.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/q1.dir/tpch_kit.cpp.o -c /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/tpch_kit.cpp

CMakeFiles/q1.dir/tpch_kit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/q1.dir/tpch_kit.cpp.i"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/tpch_kit.cpp > CMakeFiles/q1.dir/tpch_kit.cpp.i

CMakeFiles/q1.dir/tpch_kit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/q1.dir/tpch_kit.cpp.s"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/tpch_kit.cpp -o CMakeFiles/q1.dir/tpch_kit.cpp.s

CMakeFiles/q1.dir/common.cpp.o: CMakeFiles/q1.dir/flags.make
CMakeFiles/q1.dir/common.cpp.o: ../common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/q1.dir/common.cpp.o"
	/opt/gcc-5.4.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/q1.dir/common.cpp.o -c /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/common.cpp

CMakeFiles/q1.dir/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/q1.dir/common.cpp.i"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/common.cpp > CMakeFiles/q1.dir/common.cpp.i

CMakeFiles/q1.dir/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/q1.dir/common.cpp.s"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/common.cpp -o CMakeFiles/q1.dir/common.cpp.s

CMakeFiles/q1.dir/monetdb.cpp.o: CMakeFiles/q1.dir/flags.make
CMakeFiles/q1.dir/monetdb.cpp.o: ../monetdb.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/q1.dir/monetdb.cpp.o"
	/opt/gcc-5.4.0/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/q1.dir/monetdb.cpp.o -c /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/monetdb.cpp

CMakeFiles/q1.dir/monetdb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/q1.dir/monetdb.cpp.i"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/monetdb.cpp > CMakeFiles/q1.dir/monetdb.cpp.i

CMakeFiles/q1.dir/monetdb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/q1.dir/monetdb.cpp.s"
	/opt/gcc-5.4.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/monetdb.cpp -o CMakeFiles/q1.dir/monetdb.cpp.s

# Object files for target q1
q1_OBJECTS = \
"CMakeFiles/q1.dir/q1.cpp.o" \
"CMakeFiles/q1.dir/tpch_kit.cpp.o" \
"CMakeFiles/q1.dir/common.cpp.o" \
"CMakeFiles/q1.dir/monetdb.cpp.o"

# External object files for target q1
q1_EXTERNAL_OBJECTS =

q1: CMakeFiles/q1.dir/q1.cpp.o
q1: CMakeFiles/q1.dir/tpch_kit.cpp.o
q1: CMakeFiles/q1.dir/common.cpp.o
q1: CMakeFiles/q1.dir/monetdb.cpp.o
q1: CMakeFiles/q1.dir/build.make
q1: CMakeFiles/q1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable q1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/q1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/q1.dir/build: q1

.PHONY : CMakeFiles/q1.dir/build

CMakeFiles/q1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/q1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/q1.dir/clean

CMakeFiles/q1.dir/depend:
	cd /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build /export/scratch1/home/tome/Volume/mnt_mac2/expl_comp_strat/build/CMakeFiles/q1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/q1.dir/depend
