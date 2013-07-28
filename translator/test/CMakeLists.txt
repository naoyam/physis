find_package(ROSE)
if (NOT ROSE_FOUND)
  return()
endif()

include_directories(${ROSE_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/tests/gmock)
link_directories(${CMAKE_BINARY_DIR}/tests/gmock)

foreach (i test_ast_processing) 
  add_executable(${i} ${i}.cc)
  target_link_libraries(${i}
    translator
    gmock
    ${ROSE_LIBRARIES}
    ${JAVA_JVM_LIBRARY}
    ${Boost_LIBRARIES})
  add_custom_target(test-translator ${i}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  file(GLOB input_files 
    "${CMAKE_CURRENT_SOURCE_DIR}/${i}_input*.c")
  foreach (input ${input_files})
    get_filename_component(fname ${input} NAME)
    add_custom_command(
      OUTPUT ${fname}
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${fname} ${CMAKE_CURRENT_BINARY_DIR}/${fname}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${fname})
    add_custom_target(${fname} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${fname}) 
    add_dependencies(${i} ${fname})
  endforeach ()
  add_dependencies(test-translator ${i})
  unset(input_files)
endforeach ()

