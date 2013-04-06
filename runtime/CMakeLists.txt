# Boost
#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED program_options)
include_directories(${Boost_INCLUDE_DIRS})

set(RUNTIME_COMMON_SRC runtime_common.cc buffer.cc timing.cc)

add_library(physis_rt_ref ${RUNTIME_COMMON_SRC} reference_runtime.cc)
install(TARGETS physis_rt_ref DESTINATION lib)

if (MPI_ENABLED)
  include_directories(${MPI_INCLUDE_PATH})
  # Pthread is used by OpenMPI.   
  add_definitions(-pthread)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
  add_library(physis_rt_mpi ${RUNTIME_COMMON_SRC} mpi_runtime.cc
    grid.cc grid_mpi.cc  grid_util.cc rpc_mpi.cc mpi_wrapper.cc)
  install(TARGETS physis_rt_mpi DESTINATION lib)
  add_executable(test_mpi_runtime_2d test_mpi_runtime_2d.cc)
  target_link_libraries(test_mpi_runtime_2d physis_rt_mpi ${MPI_LIBRARIES})  
  add_executable(test_mpi_runtime_3d test_mpi_runtime_3d.cc)
  target_link_libraries(test_mpi_runtime_3d physis_rt_mpi ${MPI_LIBRARIES})  
  add_executable(test_grid_mpi_2d test_grid_mpi_2d.cc)
  target_link_libraries(test_grid_mpi_2d physis_rt_mpi ${MPI_LIBRARIES})  
  add_executable(test_grid_mpi_3d test_grid_mpi_3d.cc)
  target_link_libraries(test_grid_mpi_3d physis_rt_mpi ${MPI_LIBRARIES})  
  add_executable(test_grid_mpi_2d_3d test_grid_mpi_2d_3d.cc)
  target_link_libraries(test_grid_mpi_2d_3d physis_rt_mpi ${MPI_LIBRARIES})  
endif()

if (CUDA_ENABLED)
  include_directories(${CUDA_INCLUDE_DIR})
  include_directories(${CUDA_CUT_INCLUDE_DIR})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  cuda_add_library(physis_rt_cuda ${RUNTIME_COMMON_SRC}
    cuda_runtime.cc reduce_cuda.cu buffer_cuda.cu)
  install(TARGETS physis_rt_cuda DESTINATION lib)
endif()

if (MPI_ENABLED AND CUDA_ENABLED)
  cuda_add_library(physis_rt_mpi_cuda ${RUNTIME_COMMON_SRC}
    mpi_cuda_runtime.cc grid.cc grid_mpi.cc grid_mpi_cuda.cc
    grid_util.cc rpc_mpi.cc rpc_mpi_cuda.cc mpi_wrapper.cc
    buffer_cuda.cu reduce_mpi_cuda.cu)
  install(TARGETS physis_rt_mpi_cuda DESTINATION lib)
  add_executable(test_mpi_cuda_runtime test_mpi_cuda_runtime.cc)
  target_link_libraries(test_mpi_cuda_runtime physis_rt_mpi_cuda
    ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
  add_executable(test_grid_mpi_cuda_3d test_grid_mpi_cuda_3d.cc)
  target_link_libraries(test_grid_mpi_cuda_3d physis_rt_mpi_cuda
    ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
endif()
