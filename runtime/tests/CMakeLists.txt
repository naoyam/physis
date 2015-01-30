# Uniformly use libstdc++ even for non-CUDA files to simplify compilation
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_flag(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
endif ()

include_directories(${CMAKE_SOURCE_DIR}/tests/gmock)
#link_directories(${CMAKE_BINARY_DIR}/tests/gmock)
add_library(gmock ../../tests/gmock/gmock-gtest-all.cc)

set (test_src test_buffer.cc)

set(RUNTIME_COMMON_SRC
  ../runtime_common.cc ../buffer.cc ../timing.cc
  ../grid_util.cc)

add_custom_target(test-runtime
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_buffer test_buffer.cc
  ${RUNTIME_COMMON_SRC})

# nvcc does not support C++0x, so the option in CMAKE_CXX_FLAGS must not be propagated to nvcc. 
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS -g;-G)

# Tests for CUDA
if (CUDA_FOUND AND CUDA_RUNTIME_ENABLED)
  list(APPEND test_src test_buffer_cuda.cc)
  cuda_add_executable(test_buffer_cuda test_buffer_cuda.cc
    ${RUNTIME_COMMON_SRC}
    ../buffer_cuda.cu)
endif ()

# Tests for MPI
set (MPI_COMMON_SRC
  ${RUNTIME_COMMON_SRC}
  ../grid.cc
  ../grid_mpi.cc
  ../proc.cc
  ../ipc_mpi.cc
  ../mpi_wrapper.cc)
if (MPI_FOUND AND MPI_RUNTIME_ENABLED)
  list(APPEND test_src test_grid_mpi.cc)
  add_executable(test_grid_mpi
    test_grid_mpi.cc ${MPI_COMMON_SRC})
  target_link_libraries(test_grid_mpi
    ${MPI_LIBRARIES})
endif()

# Tests for MPI-CUDA
set (MPI_CUDA_COMMON_SRC
  ${MPI_COMMON_SRC}
  ../grid_mpi_cuda_exp.cc)
if (MPI_FOUND AND CUDA_FOUND AND MPI_CUDA_RUNTIME_ENABLED)
  list(APPEND test_src
    test_grid_mpi_cuda_exp.cc
    test_grid_mpi_cuda_exp_utype.cc)
  cuda_compile(MPI_CUDA_COMMON_CUDA_OBJ
    ../buffer_cuda.cu
    ../reduce_grid_mpi_cuda_exp.cu)  
  add_executable(test_grid_mpi_cuda_exp
    test_grid_mpi_cuda_exp.cc ${MPI_CUDA_COMMON_SRC}
    ${MPI_CUDA_COMMON_CUDA_OBJ})
  add_executable(test_grid_mpi_cuda_exp_utype
    test_grid_mpi_cuda_exp_utype.cc ${MPI_CUDA_COMMON_SRC}
    ${MPI_CUDA_COMMON_CUDA_OBJ})
  target_link_libraries(test_grid_mpi_cuda_exp
    ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
  target_link_libraries(test_grid_mpi_cuda_exp_utype
    ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
endif()

foreach (i ${test_src})
  get_filename_component(exe ${i} NAME_WE)
  #add_executable(${exe} ${i})
  target_link_libraries(${exe}
    gmock)
  #add_dependencies(test-translator ${exe})
  add_custom_target(test-${exe}
    COMMAND ${exe}
    DEPENDS ${exe}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(test-runtime test-${exe})
  unset(input_files)
endforeach ()

