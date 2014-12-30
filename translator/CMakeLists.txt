set (CUDA_TRANSLATOR_ENABLED TRUE CACHE BOOL
  "Flag to enable CUDA target")  
set (CUDA_HM_TRANSLATOR_ENABLED FALSE CACHE BOOL
  "Flag to enable CUDA-HM target")  
set (MPI_TRANSLATOR_ENABLED TRUE CACHE BOOL
  "Flag to enable MPI target")
set (MPI_OPENMP_TRANSLATOR_ENABLED FALSE CACHE BOOL
  "Flag to enable MPI-OpenMP target")
set (MPI_CUDA_TRANSLATOR_ENABLED FALSE CACHE BOOL
  "Flag to enable MPI-CUDA target")  
set (OPENCL_TRANSLATOR_ENABLED FALSE CACHE BOOL
  "Flag to enable OpenCL target")
set (MPI_OPENCL_TRANSLATOR_ENABLED FALSE CACHE BOOL
  "Flag to enable MPI-OpenCL target")  
###########################################################
link_directories(${PROJECT_BINARY_DIR}/common)
link_directories(${PROJECT_BINARY_DIR}/util)

# ROSE is required. Skip if not found.
find_package(ROSE)
if (NOT ROSE_FOUND)
  message (WARNING "ROSE not found. Translators will not be built.")
  return()
endif()

include_directories(${ROSE_INCLUDE_DIR})
# Use precompiled header if available
add_definitions(-include rose.h)

# Boost
#set(Boost_USE_STATIC_LIBS ON)
set(Boost_NO_BOOST_CMAKE true)
find_package(Boost REQUIRED program_options iostreams)
include_directories(${Boost_INCLUDE_DIRS})

# JNI
find_package(JNI REQUIRED)

# common sources
set (PHYSISC_SRC configuration.cc
  translation_context.cc grid.cc domain.cc
  def_analysis.cc map.cc run.cc rose_traversal.cc
  rose_util.cc rose_fortran.cc ast_processing.cc
  translator.cc reference_translator.cc kernel.cc
  alias_analysis.cc SageBuilderEx.cc
  stencil_analysis.cc stencil_range.cc translation_util.cc
  reference_runtime_builder.cc runtime_builder.cc
  rose_ast_attribute.cc reduce.cc fortran_output_fix.cc
  optimizer/optimizer.cc
  optimizer/optimization_common.cc
  optimizer/reference_optimizer.cc
  optimizer/optimization_passes.cc
  optimizer/primitive_optimization.cc
  optimizer/unconditional_get.cc
  optimizer/kernel_inlining.cc
  optimizer/loop_peeling.cc
  optimizer/register_blocking.cc
  optimizer/offset_cse.cc
  optimizer/offset_spatial_cse.cc
  optimizer/loop_opt.cc)

if (MPI_TRANSLATOR_ENABLED) 
  set(PHYSISC_SRC ${PHYSISC_SRC}
    mpi_translator.cc mpi_runtime_builder.cc
    optimizer/mpi_optimizer.cc)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi
    DESTINATION bin)
endif ()

# set(PHYSISC_SRC ${PHYSISC_SRC}
#   mpi_translator2.cc)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi2.cmake
#   ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi2)
# install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi2
#   DESTINATION bin)

if (CUDA_TRANSLATOR_ENABLED)
  set(PHYSISC_SRC ${PHYSISC_SRC}
    cuda_translator.cc
    cuda_builder.cc
    cuda_runtime_builder.cc
    optimizer/cuda_optimizer.cc)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-cuda.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda
    DESTINATION bin)
endif ()

if (MPI_CUDA_TRANSLATOR_ENABLED)
  set(PHYSISC_SRC ${PHYSISC_SRC}
    mpi_cuda_translator.cc
    mpi_cuda_runtime_builder.cc
    mpi_cuda_optimizer.cc
    optimizer/mpi_cuda_optimizer.cc)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi-cuda.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-cuda)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-cuda
    DESTINATION bin)
endif ()

if (OPENCL_TRANSLATOR_ENABLED)
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DOPENCL_DEVICE_HEADER_PATH=\\\"${CMAKE_INSTALL_PREFIX}/include\\\""
    )
  set(PHYSISC_SRC ${PHYSISC_SRC}
    opencl_translator.cc
    opencl_translator_arghack.cc
    opencl_translator_consistency.cc
    opencl_translator_create_kernel.cc
    opencl_translator_edit_kernel.cc
    opencl_translator_getemit.cc
    opencl_translator_misc.cc
    opencl_translator_stencilrun.cc
    )
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/physisc-opencl.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-opencl
    )
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-opencl
    DESTINATION
    bin)
endif()

if (MPI_OPENCL_TRANSLATOR_ENABLED)
  set(PHYSISC_SRC ${PHYSISC_SRC}
    mpi_opencl_create_kernel_body.cc
    mpi_opencl_create_kernel_call.cc
    mpi_opencl_create_kernel_misc.cc
    mpi_opencl_create_kernel.cc
    mpi_opencl_create_kernel_multi.cc
    mpi_opencl_stencilmap.cc
    mpi_opencl_stencilrun.cc
    mpi_opencl_edit_kernel.cc
    mpi_opencl_translator.cc
    mpi_opencl_runtime_builder.cc
    mpi_opencl_optimizer.cc
    )
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi-opencl.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-opencl
    )
  install(
    PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-opencl
    DESTINATION bin)
endif ()

if (MPI_OPENMP_TRANSLATOR_ENABLED)
  set(PHYSISC_SRC 
    ${PHYSISC_SRC}
    mpi_openmp_init.cc
    mpi_openmp_create_kernel.cc
    mpi_openmp_translator.cc
    )
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi-openmp.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-openmp
    )
  install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-openmp
    DESTINATION bin
    )
endif ()

if (CUDA_HM_TRANSLATOR_ENABLED)
  set(PHYSISC_SRC
    ${PHYSISC_SRC}
    cuda_hm_translator.cc
    cuda_hm_runtime_builder.cc
    )
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/physisc-cuda-hm.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda-hm
    )
  install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda-hm
    DESTINATION bin
    )
endif ()

add_library(translator STATIC ${PHYSISC_SRC})
add_executable(physisc physisc.cc)
target_link_libraries(physisc translator)
target_link_libraries(physisc ${ROSE_LIBRARIES})
target_link_libraries(physisc ${JAVA_JVM_LIBRARY})
target_link_libraries(physisc ${Boost_LIBRARIES})
target_link_libraries(physisc configuration)

install(TARGETS physisc DESTINATION bin)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/config.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-ref.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-ref)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-ref
  DESTINATION bin)

install(DIRECTORY ../include/ DESTINATION include
  FILES_MATCHING PATTERN "*.h")

add_subdirectory(test)
