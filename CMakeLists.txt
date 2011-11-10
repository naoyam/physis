cmake_minimum_required(VERSION 2.8)
project(physis)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
# config.h files are geneted into the binary dir
include_directories(${CMAKE_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wno-deprecated")
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -malign-double")
endif ()
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O0 -g -Wall -Wno-deprecated")
set(PS_DEBUG TRUE)
#set(PS_VERBOSE_DEBUG TRUE)
#set(PS_VERBOSE TRUE)
#set(PS_DEBUG TRUE)

#set(AUTO_DOUBLE_BUFFERING TRUE)
set(AUTO_DOUBLE_BUFFERING FALSE)
  
find_package(CUDA)
if (CUDA_FOUND AND
    (CUDA_VERSION VERSION_EQUAL "4.0") OR
    (CUDA_VERSION VERSION_GREATER "4.0"))
    set(CUDA_ENABLED TRUE)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_20)
  set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
  foreach(i ${CUDA_LIBRARIES})
    if (i MATCHES libcudart)
      set(CUDA_LIBRT ${i})
    endif()
  endforeach()
  message(STATUS "CUDART=${CUDA_LIBRT}")
  get_filename_component(CUDA_RT_DIR ${CUDA_LIBRT} PATH)
  find_path (CUDA_CUT_INCLUDE_DIR cutil.h
    PATHS ${CUDA_SDK_SEARCH_PATH}
    PATH_SUFFIXES "common/inc"
    NO_DEFAULT_PATH
    )
  message(STATUS "CUDA_CUT_INCLUDE_DIR=${CUDA_CUT_INCLUDE_DIR}")
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cuda_cutil_name cutil64)
  else(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cuda_cutil_name cutil32)
  endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

  find_library(CUDA_CUT_LIBRARY
    NAMES cutil cutil_x86_64 ${cuda_cutil_name}
    PATHS ${CUDA_SDK_SEARCH_PATH}
    # The new version of the sdk shows up in common/lib, but the old one is in lib
    PATH_SUFFIXES "common/lib" "lib"
    DOC "Location of cutil library"
    NO_DEFAULT_PATH
    )
  # Now search system paths
  find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
  mark_as_advanced(CUDA_CUT_LIBRARY)
  set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
  message(STATUS "CUDA_CUT_LIBRARIES=${CUDA_CUT_LIBRARIES}")
else()
  message(STATUS "Compatible CUDA toolkit (4.0 or higher) not found.")
  message(STATUS "CUDA toolkit can be explicitly specified with CUDA_TOOLKIT_ROOT_DIR CMake variable.")
  set(CUDA_ENABLED FALSE)
endif()

find_package(MPI)
if (MPI_FOUND)
  set(MPI_ENABLED TRUE)
  set(MPI_INCLUDE_DIR ${MPI_INCLUDE_PATH})
else()
  set(MPI_ENABLED FALSE)
endif()

# http://www.cmake.org/Wiki/CMake_RPATH_handling
# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Doxygen documentation
find_package(Doxygen)
if (DOXYGEN_FOUND)
  add_custom_target(doxygen ${DOXYGEN_EXECUTABLE}
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  # make clean will wipe the doxygen output directory
  set_property(DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES doxygen)
  #install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
  #DESTINATION docs/doxygen)
else ()
  message(STATUS "Doxygen not found. Some documentation is not generated.")
endif()

add_subdirectory(common)
add_subdirectory(include/physis)
add_subdirectory(util)
add_subdirectory(translator)
add_subdirectory(runtime)
add_subdirectory(tests)
