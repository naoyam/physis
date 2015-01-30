set (CUDA_RUNTIME_ENABLED TRUE CACHE BOOL
  "Flag to enable CUDA target")  
set (CUDA_HM_RUNTIME_ENABLED FALSE CACHE BOOL
  "Flag to enable CUDA HM target")  
set (MPI_RUNTIME_ENABLED TRUE CACHE BOOL
  "Flag to enable MPI target")
set (MPI_OPENMP_RUNTIME_ENABLED FALSE CACHE BOOL
  "Flag to enable MPI-OpenMP target")
set (MPI_CUDA_RUNTIME_ENABLED TRUE CACHE BOOL
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

set(RUNTIME_COMMON_SRC runtime_common.cc buffer.cc timing.cc)

add_library(physis_rt_ref ${RUNTIME_COMMON_SRC} libphysis_rt_ref.cc)
install(TARGETS physis_rt_ref DESTINATION lib)

if (MPI_FOUND AND MPI_RUNTIME_ENABLED)
  include_directories(${MPI_INCLUDE_PATH})
  # Pthread is used by OpenMPI.   
  add_definitions(-pthread)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
  add_library(physis_rt_mpi ${RUNTIME_COMMON_SRC}
    libphysis_rt_mpi.cc
    grid.cc grid_mpi.cc grid_util.cc
    proc.cc 
    ipc_mpi.cc mpi_wrapper.cc)
  install(TARGETS physis_rt_mpi DESTINATION lib)
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
    libphysis_rt_cuda.cc reduce_cuda.cu)
  install(TARGETS physis_rt_cuda DESTINATION lib)
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(
      physis_rt_cuda PROPERTIES
      COMPILE_FLAGS "-stdlib=libstdc++"
      LINK_FLAGS "-stdlib=libstdc++"
      )
  endif ()
endif()

if (CUDA_FOUND AND CUDA_HM_RUNTIME_ENABLED)
  include_directories(${CUDA_INCLUDE_DIR})
  cuda_add_library(physis_rt_cuda_hm ${RUNTIME_COMMON_SRC}
    libphysis_rt_cuda_hm.cc reduce_cuda.cu)
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(
      physis_rt_cuda_hm PROPERTIES
      COMPILE_FLAGS "-stdlib=libstdc++"
      LINK_FLAGS "-stdlib=libstdc++"
      )
  endif ()
  install(TARGETS physis_rt_cuda_hm DESTINATION lib)
endif()

if (MPI_FOUND AND CUDA_FOUND AND MPI_CUDA_RUNTIME_ENABLED)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cub-1.3.2)
  cuda_add_library(physis_rt_mpi_cuda ${RUNTIME_COMMON_SRC}
    libphysis_rt_mpi_cuda.cc
    runtime_mpi_cuda.cc
    grid.cc grid_mpi.cc
    grid_mpi_cuda_exp.cc
    grid_util.cc
    proc.cc rpc_cuda.cc 
    ipc_mpi.cc mpi_wrapper.cc
    buffer_cuda.cu reduce_grid_mpi_cuda_exp.cu)
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(
      physis_rt_mpi_cuda PROPERTIES
      COMPILE_FLAGS "-stdlib=libstdc++"
      LINK_FLAGS "-stdlib=libstdc++"
      )
  endif ()
  install(TARGETS physis_rt_mpi_cuda DESTINATION lib)
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
      libphysis_rt_opencl.cc
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
      ipc_mpi.cc
      ${RUNTIME_OPENCL_COMMON_SRC}
      buffer_opencl.cc
      grid_mpi_opencl.cc
      rpc_opencl_mpi.cc
      libphysis_rt_mpi_opencl_extra.cc
      libphysis_rt_mpi_opencl.cc
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
    libphysis_rt_mpi_openmp.cc
    libphysis_rt_mpi_openmp_numa.cc    
    grid.cc
    grid_util.cc
    grid_util_mpi_openmp.cc
    grid_mpi.cc
    grid_mpi_openmp.cc
    grid_mpi_openmp_misc.cc
    rpc_mpi.cc
    rpc_mpi_openmp.cc
    mpi_wrapper.cc
    ipc_mpi.cc
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
    ipc_mpi.cc
  )
  set_target_properties(
    physis_rt_mpi_openmp_numa PROPERTIES COMPILE_FLAGS "-DUSE_OPENMP_NUMA"
  )
  install(TARGETS physis_rt_mpi_openmp_numa DESTINATION lib)
endif()

add_subdirectory(tests)
