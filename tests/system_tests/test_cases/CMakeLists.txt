
add_executable(test_04.manual.ref.exe test_04.manual.ref.c)
add_executable(test_05.manual.ref.exe test_05.manual.ref.c)
add_executable(test_07.manual.ref.exe test_07.manual.ref.c)
add_executable(test_asymmetric.manual.ref.exe 
  test_asymmetric.manual.ref.c)
add_executable(test_27-pt-periodic.manual.ref.exe 
  test_27-pt-periodic.manual.ref.c)

if (CUDA_ENABLED)
  cuda_add_executable(test_04.manual.cuda.exe test_04.manual.cuda.cu)  
  cuda_add_executable(test_05.manual.cuda.exe test_05.manual.cuda.cu)  
  cuda_add_executable(test_07.manual.cuda.exe test_07.manual.cuda.cu)
  cuda_add_executable(test_asymmetric.manual.cuda.exe 
    test_asymmetric.manual.cuda.cu)
  cuda_add_executable(test_27-pt-periodic.manual.cuda.exe 
    test_27-pt-periodic.manual.cuda.cu)
endif ()
