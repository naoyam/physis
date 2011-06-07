link_directories(${PROJECT_BINARY_DIR}/common)
link_directories(${PROJECT_BINARY_DIR}/util)

# ROSE
find_package(ROSE)
include_directories(${ROSE_INCLUDE_DIR})

# Boost
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.40 REQUIRED program_options)
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
  rose_ast_attribute.cc)

if (MPI_ENABLED)
  set(MPI_INCLUDE_DIR ${MPI_INCLUDE_PATH})
  set(PHYSISC_SRC ${PHYSISC_SRC}
    mpi_translator.cc mpi_runtime_builder.cc)  
endif()

if (CUDA_ENABLED)
  set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
  set(PHYSISC_SRC ${PHYSISC_SRC}
    cuda_translator.cc
    cuda_builder.cc)
endif()

if (MPI_ENABLED AND CUDA_ENABLED)
  set(PHYSISC_SRC ${PHYSISC_SRC}
    mpi_cuda_translator.cc
    mpi_cuda_runtime_builder.cc
    mpi_cuda_optimizer.cc)
endif ()

add_executable(physisc ${PHYSISC_SRC})
target_link_libraries(physisc ${ROSE_LIBRARIES})
target_link_libraries(physisc ${JAVA_JVM_LIBRARY})
target_link_libraries(physisc ${Boost_LIBRARIES})
target_link_libraries(physisc configuration)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/config.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-ref.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-ref)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-cuda.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-cuda)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/physisc-mpi.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/physisc-mpi)