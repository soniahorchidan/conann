#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "faiss" for configuration "Release"
set_property(TARGET faiss APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(faiss PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libfaiss.so"
  IMPORTED_SONAME_RELEASE "libfaiss.so"
  )

list(APPEND _cmake_import_check_targets faiss )
list(APPEND _cmake_import_check_files_for_faiss "${_IMPORT_PREFIX}/lib/libfaiss.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
