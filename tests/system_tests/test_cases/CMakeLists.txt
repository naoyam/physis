
add_executable(test_4.manual.ref.exe test_4.manual.ref.c)
add_executable(test_6.manual.ref.exe test_6.manual.ref.c)

if (CUDA_ENABLED)
  cuda_add_executable(test_4.manual.cuda.exe test_4.manual.cuda.cu)  
  cuda_add_executable(test_6.manual.cuda.exe test_6.manual.cuda.cu)
endif ()