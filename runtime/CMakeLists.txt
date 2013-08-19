set (CUDA_RUNTIME_ENABLED TRUE CACHE BOOL
  "Flag to enable CUDA target")  
set (MPI_RUNTIME_ENABLED TRUE CACHE BOOL
  "Flag to enable MPI target")
set (MPI_OPENMP_RUNTIME_ENABLED FALSE CACHE BOOL
  "Flag to enable MPI-OpenMP target")
set (MPI_CUDA_RUNTIME_ENABLED FALSE CACHE BOOL
  "Flag to enable MPI-CUDA target")  
set (OPENCL_RUNTIME_ENABLED FALSE CACHE BOOL
  "Flag to enable OpenCL target")
set (MPI_OPENCL_RUNTIME_ENABLED FALSE CACHE BOOL
  "Flag to enable MPI-OpenCL target")  
###########################################################  

# Boost
#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED program_options)
include_directories(${Boost_INCLUDE_DIRS})

set(RUNTIME_COMMON_SRC runtime_common.cc runtime.cc runtime_ref.cc buffer.cc timing.cc)

add_library(physis_rt_ref ${RUNTIME_COMMON_SRC} reference_runtime.cc)
install(TARGETS physis_rt_ref DESTINATION lib)

if (MPI_FOUND AND MPI_RUNTIME_ENABLED)
  include_directories(${MPI_INCLUDE_PATH})
  # Pthread is used by OpenMPI.   
  add_definitions(-pthread)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
  add_library(physis_rt_mpi ${RUNTIME_COMMON_SRC}
    mpi_runtime.cc
    runtime.cc runtime_mpi.cc
    grid.cc grid_mpi.cc grid_space_mpi.cc grid_util.cc
    proc.cc rpc.cc 
    ipc_mpi.cc mpi_wrapper.cc)
  install(TARGETS physis_rt_mpi DESTINATION lib)
  #add_executable(test_mpi_runtime_2d test_mpi_runtime_2d.cc)
  #target_link_libraries(test_mpi_runtime_2d physis_rt_mpi ${MPI_LIBRARIES})  
  #add_executable(test_mpi_runtime_3d test_mpi_runtime_3d.cc)
  #target_link_libraries(test_mpi_runtime_3d physis_rt_mpi ${MPI_LIBRARIES})  
  #add_executable(test_grid_mpi_2d test_grid_mpi_2d.cc)
  #target_link_libraries(test_grid_mpi_2d physis_rt_mpi ${MPI_LIBRARIES})  
  #add_executable(test_grid_mpi_3d test_grid_mpi_3d.cc)
  #target_link_libraries(test_grid_mpi_3d physis_rt_mpi ${MPI_LIBRARIES})  
  #add_executable(test_grid_mpi_2d_3d test_grid_mpi_2d_3d.cc)
  #target_link_libraries(test_grid_mpi_2d_3d physis_rt_mpi ${MPI_LIBRARIES})
  # if (FALSE)
  # add_library(physis_rt_mpi2 ${RUNTIME_COMMON_SRC}
  #   mpi_runtime2.cc
  #   runtime.cc runtime_mpi.cc runtime_mpi2.cc
  #   #mpi_runtime_common.cc
  #   grid.cc grid_mpi.cc grid_mpi2.cc grid_util.cc
  #   proc.cc rpc.cc rpc2.cc
  #   ipc_mpi.cc mpi_wrapper.cc)
  # set_target_properties(
  #   physis_rt_mpi2 PROPERTIES COMPILE_FLAGS "-DMPI_RUNTIME_2"
  # )
  # install(TARGETS physis_rt_mpi2 DESTINATION lib)
  # endif()
endif()

if (CUDA_FOUND AND CUDA_RUNTIME_ENABLED)
  include_directories(${CUDA_INCLUDE_DIR})
  if (AUTO_TUNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif ()
  cuda_add_library(physis_rt_cuda ${RUNTIME_COMMON_SRC}
    runtime.cc runtime_cuda.cc
    cuda_runtime.cc reduce_cuda.cu)
  install(TARGETS physis_rt_cuda DESTINATION lib)
endif()

if (MPI_FOUND AND CUDA_FOUND AND MPI_CUDA_RUNTIME_ENABLED)
  cuda_add_library(physis_rt_mpi_cuda ${RUNTIME_COMMON_SRC}
    mpi_cuda_runtime.cc grid.cc grid_mpi.cc grid_mpi_cuda.cc
    grid_util.cc rpc_mpi.cc rpc_mpi_cuda.cc mpi_wrapper.cc
    inter_proc_comm_mpi.cc
    buffer_cuda.cu reduce_mpi_cuda.cu)
  install(TARGETS physis_rt_mpi_cuda DESTINATION lib)
  add_executable(test_mpi_cuda_runtime test_mpi_cuda_runtime.cc)
  target_link_libraries(test_mpi_cuda_runtime physis_rt_mpi_cuda
    ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
  add_executable(test_grid_mpi_cuda_3d test_grid_mpi_cuda_3d.cc)
  target_link_libraries(test_grid_mpi_cuda_3d physis_rt_mpi_cuda
    ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
endif()

set(RUNTIME_OPENCL_COMMON_SRC
  opencl_kernelinit.cc
  opencl_misc.cc
  opencl_psinit.cc
)

if (OPENCL_FOUND AND OPENCL_RUNTIME_ENABLED)  
  include_directories(${OPENCL_INCLUDE_PATH})
  add_library(physis_rt_opencl ${RUNTIME_COMMON_SRC}
      ${RUNTIME_OPENCL_COMMON_SRC}
      opencl_gridcp.cc
      opencl_gridinit.cc
      opencl_runtime.cc
    )
  install(
    TARGETS physis_rt_opencl
    DESTINATION lib
    )
endif()

if (MPI_FOUND AND OPENCL_FOUND AND MPI_OPENCL_RUNTIME_ENABLED)
  include_directories(${OPENCL_INCLUDE_PATH})
  add_library(physis_rt_mpi_opencl ${RUNTIME_COMMON_SRC}
      grid.cc grid_mpi.cc grid_util.cc rpc_mpi.cc mpi_wrapper.cc
      inter_proc_comm_mpi.cc
      ${RUNTIME_OPENCL_COMMON_SRC}
      buffer_opencl.cc
      grid_mpi_opencl.cc
      rpc_opencl_mpi.cc
      mpi_opencl_runtime_extra.cc
      mpi_opencl_runtime.cc
      rpc_mpi_opencl.cc
  )
  install(
    TARGETS physis_rt_mpi_opencl
    DESTINATION lib
  )
endif()

if (MPI_FOUND AND MPI_OPENMP_RUNTIME_ENABLED)
  include_directories(${MPI_INCLUDE_PATH})
  # Pthread is used by OpenMPI.   
  add_definitions(-pthread)
  add_library(physis_rt_mpi_openmp
    ${RUNTIME_COMMON_SRC}
    buffer_mpi_openmp.cc
    buffer_mpi_openmp_numa.cc
    #mpi_runtime.cc
    mpi_openmp_runtime.cc
    mpi_openmp_runtime_numa.cc
    grid.cc
    grid_util.cc
    grid_util_mpi_openmp.cc
    grid_mpi.cc
    grid_mpi_openmp.cc
    grid_mpi_openmp_misc.cc
    rpc_mpi.cc
    rpc_mpi_openmp.cc
    mpi_wrapper.cc
    inter_proc_comm_mpi.cc
  )
  set_target_properties(
    physis_rt_mpi_openmp PROPERTIES COMPILE_FLAGS "-UUSE_OPENMP_NUMA"
  )
  install(TARGETS physis_rt_mpi_openmp DESTINATION lib)
endif()

if (MPI_FOUND AND NUMA_FOUND AND MPI_OPENMP_RUNTIME_ENABLED)
  include_directories(${MPI_INCLUDE_PATH})
  add_library(physis_rt_mpi_openmp_numa
    ${RUNTIME_COMMON_SRC}
    buffer_mpi_openmp.cc
    buffer_mpi_openmp_numa.cc
    mpi_openmp_runtime.cc
    mpi_openmp_runtime_numa.cc
    grid.cc
    grid_util.cc
    grid_util_mpi_openmp.cc
    grid_mpi.cc
    grid_mpi_openmp.cc
    grid_mpi_openmp_misc.cc
    rpc_mpi.cc
    rpc_mpi_openmp.cc
    mpi_wrapper.cc
    inter_proc_comm_mpi.cc
  )
  set_target_properties(
    physis_rt_mpi_openmp_numa PROPERTIES COMPILE_FLAGS "-DUSE_OPENMP_NUMA"
  )
  install(TARGETS physis_rt_mpi_openmp_numa DESTINATION lib)
endif()
