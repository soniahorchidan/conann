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
CMAKE_SOURCE_DIR = /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild

# Utility rule file for googlebenchmark-populate.

# Include any custom commands dependencies for this target.
include CMakeFiles/googlebenchmark-populate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/googlebenchmark-populate.dir/progress.make

CMakeFiles/googlebenchmark-populate: CMakeFiles/googlebenchmark-populate-complete

CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-install
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-mkdir
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-download
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-patch
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-configure
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-build
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-install
CMakeFiles/googlebenchmark-populate-complete: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'googlebenchmark-populate'"
	/usr/local/bin/cmake -E make_directory /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles
	/usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles/googlebenchmark-populate-complete
	/usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-done

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update:
.PHONY : googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-build: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No build step for 'googlebenchmark-populate'"
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E echo_append
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-build

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-configure: googlebenchmark-populate-prefix/tmp/googlebenchmark-populate-cfgcmd.txt
googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-configure: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "No configure step for 'googlebenchmark-populate'"
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E echo_append
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-configure

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-download: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-gitinfo.txt
googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-download: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'googlebenchmark-populate'"
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps && /usr/local/bin/cmake -P /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/tmp/googlebenchmark-populate-gitclone.cmake
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps && /usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-download

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-install: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No install step for 'googlebenchmark-populate'"
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E echo_append
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-install

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'googlebenchmark-populate'"
	/usr/local/bin/cmake -Dcfgdir= -P /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/tmp/googlebenchmark-populate-mkdirs.cmake
	/usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-mkdir

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-patch: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'googlebenchmark-populate'"
	/usr/local/bin/cmake -E echo_append
	/usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-patch

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update:
.PHONY : googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-test: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No test step for 'googlebenchmark-populate'"
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E echo_append
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-build && /usr/local/bin/cmake -E touch /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-test

googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Performing update step for 'googlebenchmark-populate'"
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-src && /usr/local/bin/cmake -P /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/tmp/googlebenchmark-populate-gitupdate.cmake

googlebenchmark-populate: CMakeFiles/googlebenchmark-populate
googlebenchmark-populate: CMakeFiles/googlebenchmark-populate-complete
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-build
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-configure
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-download
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-install
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-mkdir
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-patch
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-test
googlebenchmark-populate: googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/googlebenchmark-populate-update
googlebenchmark-populate: CMakeFiles/googlebenchmark-populate.dir/build.make
.PHONY : googlebenchmark-populate

# Rule to build all files generated by this target.
CMakeFiles/googlebenchmark-populate.dir/build: googlebenchmark-populate
.PHONY : CMakeFiles/googlebenchmark-populate.dir/build

CMakeFiles/googlebenchmark-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/googlebenchmark-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/googlebenchmark-populate.dir/clean

CMakeFiles/googlebenchmark-populate.dir/depend:
	cd /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild /home/ubuntu/conann-dev/faiss-1.9.0/_deps/googlebenchmark-subbuild/CMakeFiles/googlebenchmark-populate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/googlebenchmark-populate.dir/depend

