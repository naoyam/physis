
add_executable(test_04.manual.ref.exe test_04.manual.ref.c)
add_executable(test_06.manual.ref.exe test_06.manual.ref.c)

if (CUDA_ENABLED)
  cuda_add_executable(test_04.manual.cuda.exe test_04.manual.cuda.cu)  
  cuda_add_executable(test_06.manual.cuda.exe test_06.manual.cuda.cu)
endif ()
