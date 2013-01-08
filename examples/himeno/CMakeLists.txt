configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Makefile.cmake
  @CMAKE_CURRENT_BINARY_DIR@/tmp/Makefile @ONLY)

install(FILES himenobmtxpa_physis.c himenobmtxpa_original.c
  DESTINATION examples/himeno)

install(FILES @CMAKE_CURRENT_BINARY_DIR@/tmp/Makefile
  DESTINATION examples/himeno)
