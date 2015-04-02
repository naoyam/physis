
file(GLOB test_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.manual.ref.c")
foreach (src ${test_srcs})
  get_filename_component(fname ${src} NAME_WE)
  add_executable(${fname}.manual.ref.exe ${src})
endforeach ()

# uses the same manual code as test_redblack
add_executable(test_redblack-separated.manual.ref.exe
  test_redblack.manual.ref.c)

file(GLOB cuda_test_srcs
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.manual.cuda.cu")

if (CUDA_FOUND)
  foreach (src ${cuda_test_srcs})
    get_filename_component(fname ${src} NAME_WE)
    cuda_add_executable(${fname}.manual.cuda.exe ${src})
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set_target_properties(
	${fname}.manual.cuda.exe PROPERTIES
	LINK_FLAGS "-stdlib=libstdc++"
	)
    endif ()
  endforeach ()
  # For integer code, use the normal C code for testing the CUDA version
  add_executable(test_7-pt-int-type.manual.cuda.exe
    test_7-pt-int-type.manual.ref.c)
  add_executable(test_9-pt-reduction.manual.cuda.exe
    test_9-pt-reduction.manual.ref.c)
  add_executable(test_9-pt-periodic-reduction.manual.cuda.exe
    test_9-pt-periodic-reduction.manual.ref.c)
endif ()
