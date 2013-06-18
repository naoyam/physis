
add_executable(test_04.manual.ref.exe test_04.manual.ref.c)
add_executable(test_05.manual.ref.exe test_05.manual.ref.c)
add_executable(test_07.manual.ref.exe test_07.manual.ref.c)
add_executable(test_15.manual.ref.exe test_15.manual.ref.c)
add_executable(test_16.manual.ref.exe test_16.manual.ref.c)
add_executable(test_asymmetric.manual.ref.exe 
  test_asymmetric.manual.ref.c)
add_executable(test_3-pt-periodic.manual.ref.exe 
  test_3-pt-periodic.manual.ref.c)
add_executable(test_7-pt-periodic.manual.ref.exe 
  test_7-pt-periodic.manual.ref.c)
add_executable(test_27-pt-periodic.manual.ref.exe 
  test_27-pt-periodic.manual.ref.c)
add_executable(test_asymmetric-periodic.manual.ref.exe
  test_asymmetric-periodic.manual.ref.c)
add_executable(test_stencil-hole.manual.ref.exe
  test_stencil-hole.manual.ref.c)
add_executable(test_7-pt-neumann-cond.manual.ref.exe
  test_7-pt-neumann-cond.manual.ref.c)
add_executable(test_user-defined-type1.manual.ref.exe
  test_user-defined-type1.manual.ref.c)
add_executable(test_user-defined-type3.manual.ref.exe
  test_user-defined-type3.manual.ref.c)

if (CUDA_FOUND)
  cuda_add_executable(test_04.manual.cuda.exe test_04.manual.cuda.cu)  
  cuda_add_executable(test_05.manual.cuda.exe test_05.manual.cuda.cu)  
  cuda_add_executable(test_07.manual.cuda.exe test_07.manual.cuda.cu)
  cuda_add_executable(test_15.manual.cuda.exe test_15.manual.cuda.cu)
  cuda_add_executable(test_16.manual.cuda.exe test_16.manual.cuda.cu)    
  cuda_add_executable(test_asymmetric.manual.cuda.exe 
    test_asymmetric.manual.cuda.cu)
  cuda_add_executable(test_3-pt-periodic.manual.cuda.exe 
    test_3-pt-periodic.manual.cuda.cu)
   cuda_add_executable(test_7-pt-periodic.manual.cuda.exe 
    test_7-pt-periodic.manual.cuda.cu)
  cuda_add_executable(test_27-pt-periodic.manual.cuda.exe 
    test_27-pt-periodic.manual.cuda.cu)
  cuda_add_executable(test_asymmetric-periodic.manual.cuda.exe
    test_asymmetric-periodic.manual.cuda.cu)
  cuda_add_executable(test_stencil-hole.manual.cuda.exe
    test_stencil-hole.manual.cuda.cu)
  cuda_add_executable(test_7-pt-neumann-cond.manual.cuda.exe
    test_7-pt-neumann-cond.manual.cuda.cu)
  cuda_add_executable(test_user-defined-type1.manual.cuda.exe
    test_user-defined-type1.manual.cuda.cu)
  cuda_add_executable(test_user-defined-type3.manual.cuda.exe
    test_user-defined-type3.manual.cuda.cu)
endif ()
