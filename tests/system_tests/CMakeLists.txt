
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_system_tests.sh.cmake
  ${CMAKE_BINARY_DIR}/run_system_tests.sh)

add_subdirectory(test_cases)
