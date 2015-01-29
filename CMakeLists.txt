# Need CUDA_HOST_COMPILER available in 2.8.10 and onward
cmake_minimum_required(VERSION 2.8.10)

# CMAKE_BUILD_TYPE must be set before calling project.
set (CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
  "Choose the type of build, options are: VerboseDebug Debug Release RelWithDebInfo MinSizeRel.")
message (STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

project(physis)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake"
  "${CMAKE_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
# config.h files are geneted into the binary dir
include_directories(${CMAKE_BINARY_DIR})

function (add_flag var flag)
  set (${var} "${${var}} ${flag}" PARENT_SCOPE)
endfunction ()

foreach (i CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  unset (${i} CACHE)
endforeach ()

if (${CMAKE_C_COMPILER} MATCHES icc)
  set (CMAKE_COMPILER_IS_ICC TRUE)
  message (STATUS "Compiler is icc.")
else ()
  set (CMAKE_COMPILER_IS_ICC FALSE)
endif ()

# Add warning flags
if (CMAKE_COMPILER_IS_GNUCXX)
  set (WARN_FLAGS "-Wall -Wno-unused-function")
  # Unused local typedefs are complained with g++-4.8 and Boost 1.53.  
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.7)
    message (STATUS "CXX compiler version is > 4.7: ${CMAKE_CXX_COMPILER_VERSION}")
    add_flag (WARN_FLAGS "-Wno-unused-local-typedefs")
  endif ()
elseif (CMAKE_COMPILER_IS_ICC)
  set (WARN_FLAGS "-Wall -Wcheck -Wno-deprecated -Wno-unused-function")
endif ()

add_flag (CMAKE_CXX_FLAGS "${WARN_FLAGS}")

# CUDA on Linux requires this flag.
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -malign-double")
  add_flag(CMAKE_CXX_FLAGS "-malign-double")
endif ()

# CUDA on Mac uses libstdc++ rather that the default libc++
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  message (STATUS "Using libstdc++")
  add_flag(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
endif ()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Default logging configuration
set(PS_VERBOSE FALSE)
set(PS_DEBUG FALSE)
set(PS_WARNING TRUE)
# Enable debug logging in the debug mode
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set (PS_DEBUG TRUE)
endif ()
# Disable debug logging in the debug mode
if ((CMAKE_BUILD_TYPE STREQUAL "Release") OR
    (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
  set (PS_WARNING FALSE)
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "VerboseDebug")
  set (PS_DEBUG TRUE)
  set (PS_VERBOSE TRUE)
endif ()

set(AUTO_DOUBLE_BUFFERING FALSE)

set (AUTO_TUNING FALSE CACHE BOOL
  "Flag to enable auto tuning.")
# Ensure the actual value is either TRUE or FALSE so that variable
# replacement with configure_file can be assumed to be either "TRUE"
# or "FALSE", rather than "ON" or "OFF".
if (AUTO_TUNING)
  set (AUTO_TUNING TRUE)
else ()
  set (AUTO_TUNING FALSE)  
endif ()
message (STATUS "Auto tuning: ${AUTO_TUNING}")
  
find_package(CUDA)
if (CUDA_FOUND AND
    (CUDA_VERSION VERSION_EQUAL "4.0") OR
    (CUDA_VERSION VERSION_GREATER "4.0"))
  # Suppress warning on assert(expr && "message")
  set (CUDA_NVCC_FLAGS -arch sm_20;-Xcudafe --diag_suppress=boolean_controlling_expr_is_constant)
  set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
  foreach(i ${CUDA_LIBRARIES})
    if (i MATCHES libcudart)
      set(CUDA_LIBRT ${i})
    endif()
  endforeach()
  message(STATUS "CUDART=${CUDA_LIBRT}")
  get_filename_component(CUDA_RT_DIR ${CUDA_LIBRT} PATH)
else()
  message(WARNING "Compatible CUDA toolkit (4.0 or higher) not found.")
  message(WARNING "CUDA toolkit can be explicitly specified with CUDA_TOOLKIT_ROOT_DIR CMake variable.")
endif()

find_package(MPI)

if (MPI_FOUND)
  set(MPI_INCLUDE_DIR ${MPI_INCLUDE_PATH})
  message(STATUS "MPI found")
  message(STATUS "MPIEXEC=${MPIEXEC}")
  message(STATUS "MPI_INCLUDE_PATH=${MPI_INCLUDE_PATH}")
  message(STATUS "MPI_LIBRARIES=${MPI_LIBRARIES}")
endif()

find_library(NUMA_LIBRARY numa)
if (NUMA_LIBRARY)
  set(NUMA_FOUND TRUE)
else()
  set(NUMA_FOUND FALSE)
endif()

find_package (OpenCL)

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
add_subdirectory(tests)
add_subdirectory(include/physis)
add_subdirectory(util)
add_subdirectory(translator)
add_subdirectory(runtime)
add_subdirectory(examples)
