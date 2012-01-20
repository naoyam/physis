
link_directories(${PROJECT_BINARY_DIR}/common)
link_directories(${PROJECT_BINARY_DIR}/util)

# ROSE is required. Skip if not found.
find_package(ROSE)
if (NOT ROSE_FOUND)
  return()
endif()

include_directories(${ROSE_INCLUDE_DIR})

# Boost
#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED program_options)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# JNI
find_package(JNI REQUIRED)

# common sources
set (PHYSISC_SRC physisc.cc configuration.cc
  translation_context.cc grid.cc domain.cc
  def_analysis.cc map.cc run.cc rose_traversal.cc rose_util.cc translator.cc
  reference_translator.cc kernel.cc alias_analysis.cc SageBuilderEx.cc
  stencil_analysis.cc stencil_range.cc translation_util.cc
  reference_runtime_builder.cc runtime_builder.cc
  rose_ast_attribute.cc reduce.cc)

#if (MPI_ENABLED)
#  set(MPI_INCLUDE_DIR ${MPI_INCLUDE_PATH})
set(PHYSISC_SRC ${PHYSISC_SRC}
  mpi_translator.cc mpi_runtime_builder.cc)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi DESTINATION bin)
#endif()

#if (CUDA_ENABLED)
#  set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
  set(PHYSISC_SRC ${PHYSISC_SRC}
    cuda_translator.cc
    cuda_builder.cc)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-cuda.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda DESTINATION bin)
#endif()

#if (MPI_ENABLED AND CUDA_ENABLED)
set(PHYSISC_SRC ${PHYSISC_SRC}
  mpi_cuda_translator.cc
  mpi_cuda_runtime_builder.cc
  mpi_cuda_optimizer.cc)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi-cuda.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-cuda)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-cuda DESTINATION bin)
#endif ()

#if (OPENCL_ENABLED)
#  set(OPENCL_INCLUDE_DIR ${OPENCL_INCLUDE_DIRS})
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
#endif()

#if (MPI_ENABLED AND OPENCL_ENABLED)
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
#endif ()

add_executable(physisc ${PHYSISC_SRC})
target_link_libraries(physisc ${ROSE_LIBRARIES})
target_link_libraries(physisc ${JAVA_JVM_LIBRARY})
target_link_libraries(physisc ${Boost_LIBRARIES})
target_link_libraries(physisc configuration)

install(TARGETS physisc DESTINATION bin)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/config.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-ref.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-ref)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-ref DESTINATION bin)


install(DIRECTORY ../include/ DESTINATION include FILES_MATCHING PATTERN "*.h")



