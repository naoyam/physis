
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh)

add_subdirectory(tests)