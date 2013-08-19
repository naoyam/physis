find_package(ROSE)
if (NOT ROSE_FOUND)
  return()
endif()

include_directories(${ROSE_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/tests/gmock)
link_directories(${CMAKE_BINARY_DIR}/tests/gmock)

set (test_src
  test_ast_processing.cc test_grid.cc
  test_ast_traversal.cc)

add_custom_target(test-translator
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

foreach (i ${test_src})
  get_filename_component(exe ${i} NAME_WE)
  add_executable(${exe} ${i} common.cc)
  target_link_libraries(${exe}
    translator
    gmock
    ${ROSE_LIBRARIES}
    ${JAVA_JVM_LIBRARY}
    ${Boost_LIBRARIES})
  file(GLOB input_files 
    "${CMAKE_CURRENT_SOURCE_DIR}/${exe}_input*.c")
  foreach (input ${input_files})
    get_filename_component(fname ${input} NAME)
    add_custom_command(
      OUTPUT ${fname}
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${fname} ${CMAKE_CURRENT_BINARY_DIR}/${fname}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${fname})
    add_custom_target(${fname} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${fname}) 
    add_dependencies(${exe} ${fname})
  endforeach ()
  #add_dependencies(test-translator ${exe})
  add_custom_target(test-${exe}
    COMMAND ${exe}
    DEPENDS ${exe}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(test-translator test-${exe})
  unset(input_files)
endforeach ()

