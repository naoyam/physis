
link_directories(${PROJECT_BINARY_DIR}/common)
link_directories(${PROJECT_BINARY_DIR}/util)

# ROSE is required. Skip if not found.
find_package(ROSE)
if (NOT ROSE_FOUND)
  message (WARNING "ROSE not found. Translators will not be built.")
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
  def_analysis.cc map.cc run.cc rose_traversal.cc rose_util.cc
  translator.cc reference_translator.cc kernel.cc
  alias_analysis.cc SageBuilderEx.cc
  stencil_analysis.cc stencil_range.cc translation_util.cc
  reference_runtime_builder.cc runtime_builder.cc
  rose_ast_attribute.cc reduce.cc
  optimizer/optimizer.cc
  optimizer/optimization_common.cc
  optimizer/reference_optimizer.cc
  optimizer/optimization_passes.cc
  optimizer/premitive_optimization.cc
  optimizer/unconditional_get.cc
  optimizer/kernel_inlining.cc
  optimizer/loop_peeling.cc
  optimizer/register_blocking.cc
  optimizer/offset_cse.cc
  optimizer/offset_spatial_cse.cc
  optimizer/loop_opt.cc)

set(PHYSISC_SRC ${PHYSISC_SRC}
  mpi_translator.cc mpi_runtime_builder.cc
  optimizer/mpi_optimizer.cc)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi
  DESTINATION bin)

set(PHYSISC_SRC ${PHYSISC_SRC}
  cuda_translator.cc
  cuda_builder.cc
  cuda_runtime_builder.cc
  optimizer/cuda_optimizer.cc)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-cuda.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda
  DESTINATION bin)

set(PHYSISC_SRC ${PHYSISC_SRC}
  mpi_cuda_translator.cc
  mpi_cuda_runtime_builder.cc
  mpi_cuda_optimizer.cc
  optimizer/mpi_cuda_optimizer.cc)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi-cuda.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-cuda)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi-cuda
  DESTINATION bin)

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
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/physisc-ref
  DESTINATION bin)

install(DIRECTORY ../include/ DESTINATION include
  FILES_MATCHING PATTERN "*.h")
