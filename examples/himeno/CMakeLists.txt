configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.cmake
  @CMAKE_CURRENT_BINARY_DIR@/tmp/Makefile @ONLY)

install(FILES
  himenobmtxpa_physis.c himenobmtxpa_original.c
  @CMAKE_CURRENT_BINARY_DIR@/tmp/Makefile
  physis.conf
  opt.conf
  DESTINATION examples/himeno)

