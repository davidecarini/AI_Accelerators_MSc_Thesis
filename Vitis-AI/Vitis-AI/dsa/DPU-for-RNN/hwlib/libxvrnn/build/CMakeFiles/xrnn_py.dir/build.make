# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build

# Include any dependencies generated for this target.
include CMakeFiles/xrnn_py.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/xrnn_py.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/xrnn_py.dir/flags.make

CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.o: CMakeFiles/xrnn_py.dir/flags.make
CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.o: ../python/xrnn_python.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.o"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.o -c /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/python/xrnn_python.cpp

CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.i"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/python/xrnn_python.cpp > CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.i

CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.s"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/python/xrnn_python.cpp -o CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.s

CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.o: CMakeFiles/xrnn_py.dir/flags.make
CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.o: ../src/lstm_xrnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.o"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.o -c /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/src/lstm_xrnn.cpp

CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.i"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/src/lstm_xrnn.cpp > CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.i

CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.s"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/src/lstm_xrnn.cpp -o CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.s

CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.o: CMakeFiles/xrnn_py.dir/flags.make
CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.o: ../src/xxrnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.o"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.o -c /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/src/xxrnn.cpp

CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.i"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/src/xxrnn.cpp > CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.i

CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.s"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/src/xxrnn.cpp -o CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.s

# Object files for target xrnn_py
xrnn_py_OBJECTS = \
"CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.o" \
"CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.o" \
"CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.o"

# External object files for target xrnn_py
xrnn_py_EXTERNAL_OBJECTS =

xrnn_py.so: CMakeFiles/xrnn_py.dir/python/xrnn_python.cpp.o
xrnn_py.so: CMakeFiles/xrnn_py.dir/src/lstm_xrnn.cpp.o
xrnn_py.so: CMakeFiles/xrnn_py.dir/src/xxrnn.cpp.o
xrnn_py.so: CMakeFiles/xrnn_py.dir/build.make
xrnn_py.so: CMakeFiles/xrnn_py.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library xrnn_py.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/xrnn_py.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/xrnn_py.dir/build: xrnn_py.so

.PHONY : CMakeFiles/xrnn_py.dir/build

CMakeFiles/xrnn_py.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/xrnn_py.dir/cmake_clean.cmake
.PHONY : CMakeFiles/xrnn_py.dir/clean

CMakeFiles/xrnn_py.dir/depend:
	cd /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build /group/xbjlab/dphi_algo/bokangz/project/RNNT1.4_test/hwlib/libxvrnn/build/CMakeFiles/xrnn_py.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/xrnn_py.dir/depend

