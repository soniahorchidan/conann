#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "benchmark::benchmark" for configuration ""
set_property(TARGET benchmark::benchmark APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(benchmark::benchmark PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libbenchmark.a"
  )

list(APPEND _cmake_import_check_targets benchmark::benchmark )
list(APPEND _cmake_import_check_files_for_benchmark::benchmark "${_IMPORT_PREFIX}/lib/libbenchmark.a" )

# Import target "benchmark::benchmark_main" for configuration ""
set_property(TARGET benchmark::benchmark_main APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(benchmark::benchmark_main PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libbenchmark_main.a"
  )

list(APPEND _cmake_import_check_targets benchmark::benchmark_main )
list(APPEND _cmake_import_check_files_for_benchmark::benchmark_main "${_IMPORT_PREFIX}/lib/libbenchmark_main.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
