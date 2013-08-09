
add_executable(test_7-pt.manual.ref.exe test_7-pt.manual.ref.c)
add_executable(test_27-pt.manual.ref.exe test_27-pt.manual.ref.c)
add_executable(test_7-pt-multi-iterations.manual.ref.exe test_7-pt-multi-iterations.manual.ref.c)
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
add_executable(test_redblack.manual.ref.exe
  test_redblack.manual.ref.c)
add_executable(test_redblack-periodic.manual.ref.exe
  test_redblack-periodic.manual.ref.c)
# uses the same manual code as test_redblack
add_executable(test_redblack-separated.manual.ref.exe
  test_redblack.manual.ref.c)
add_executable(test_7-pt-int-type.manual.ref.exe
  test_7-pt-int-type.manual.ref.c)
# For integer code, use the normal C code for testing the CUDA version
add_executable(test_7-pt-int-type.manual.cuda.exe
  test_7-pt-int-type.manual.ref.c)
add_executable(test_7-pt-double-type.manual.ref.exe
  test_7-pt-double-type.manual.ref.c)
add_executable(test_7-pt-type-mix.manual.ref.exe
  test_7-pt-type-mix.manual.ref.c)
add_executable(test_3-pt-1d.manual.ref.exe
  test_3-pt-1d.manual.ref.c)
add_executable(test_9-pt-2d.manual.ref.exe
  test_9-pt-2d.manual.ref.c)
add_executable(test_mixed-dim.manual.ref.exe
  test_mixed-dim.manual.ref.c)
add_executable(test_mixed-dim2.manual.ref.exe
  test_mixed-dim2.manual.ref.c)
add_executable(test_mixed-dim3.manual.ref.exe
  test_mixed-dim3.manual.ref.c)
add_executable(test_user-defined-type5.manual.ref.exe
  test_user-defined-type5.manual.ref.c)

if (CUDA_FOUND)
  cuda_add_executable(test_7-pt.manual.cuda.exe test_7-pt.manual.cuda.cu)  
  cuda_add_executable(test_27-pt.manual.cuda.exe test_27-pt.manual.cuda.cu)  
  cuda_add_executable(test_7-pt-multi-iterations.manual.cuda.exe test_7-pt-multi-iterations.manual.cuda.cu)
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
  cuda_add_executable(test_redblack.manual.cuda.exe
    test_redblack.manual.cuda.cu)
  cuda_add_executable(test_redblack-periodic.manual.cuda.exe
    test_redblack-periodic.manual.cuda.cu)
  cuda_add_executable(test_redblack-separated.manual.cuda.exe
    test_redblack.manual.cuda.cu)
  cuda_add_executable(test_7-pt-double-type.manual.cuda.exe
    test_7-pt-double-type.manual.cuda.cu)
  cuda_add_executable(test_7-pt-type-mix.manual.cuda.exe
    test_7-pt-type-mix.manual.cuda.cu)
  cuda_add_executable(test_3-pt-1d.manual.cuda.exe
    test_3-pt-1d.manual.cuda.cu)
  cuda_add_executable(test_9-pt-2d.manual.cuda.exe
    test_9-pt-2d.manual.cuda.cu)
  cuda_add_executable(test_mixed-dim.manual.cuda.exe
    test_mixed-dim.manual.cuda.cu)
  cuda_add_executable(test_mixed-dim2.manual.cuda.exe
    test_mixed-dim2.manual.cuda.cu)
  cuda_add_executable(test_mixed-dim3.manual.cuda.exe
    test_mixed-dim3.manual.cuda.cu)
  cuda_add_executable(test_user-defined-type5.manual.cuda.exe
    test_user-defined-type5.manual.cuda.cu)
endif ()
