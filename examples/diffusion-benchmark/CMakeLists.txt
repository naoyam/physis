configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.cmake
  @CMAKE_CURRENT_BINARY_DIR@/tmp/Makefile @ONLY)

install(FILES
  autotune.conf
  baseline.cc
  baseline.h
  CMakeLists.txt
  diffusion3d.cc
  diffusion3d_cuda.cu
  diffusion3d_cuda.h
  diffusion3d_cuda_shared.cu
  diffusion3d_cuda_temporal_blocking.cu
  diffusion3d_cuda_temporal_blocking.h
  diffusion3d.h
  diffusion3d.mic.c
  diffusion3d_mic.cc
  diffusion3d_mic.h
  diffusion3d_openmp.cc
  diffusion3d_openmp.h
  diffusion3d_openmp_temporal_blocking.cc
  diffusion3d_openmp_temporal_blocking.h
  diffusion3d_physis.c
  diffusion3d_physis.h
  main.cc
  opt.conf
  README
  stopwatch.h
  @CMAKE_CURRENT_BINARY_DIR@/tmp/Makefile
  DESTINATION examples/diffusion-benchmark)

