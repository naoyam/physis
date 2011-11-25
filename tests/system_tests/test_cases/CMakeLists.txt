
add_executable(test_04.manual.ref.exe test_04.manual.ref.c)
add_executable(test_05.manual.ref.exe test_05.manual.ref.c)
add_executable(test_07.manual.ref.exe test_07.manual.ref.c)
add_executable(test_15.manual.ref.exe test_15.manual.ref.c)

if (CUDA_ENABLED)
  cuda_add_executable(test_04.manual.cuda.exe test_04.manual.cuda.cu)  
  cuda_add_executable(test_05.manual.cuda.exe test_05.manual.cuda.cu)  
  cuda_add_executable(test_07.manual.cuda.exe test_07.manual.cuda.cu)
  cuda_add_executable(test_15.manual.cuda.exe test_15.manual.cuda.cu)  
endif ()
