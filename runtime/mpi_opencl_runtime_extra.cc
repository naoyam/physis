#include "runtime/mpi_opencl_runtime.h"

using namespace physis::runtime;

namespace {


void InitOpenCL(
    int my_rank, int num_local_processes, int *argc, char ***argv
)
{
  // Assumes each local process has successive process rank
  int dev_id = my_rank % num_local_processes;

  clinfo_generic = new CLMPIinfo(dev_id);
  clinfo_inner = new CLMPIinfo(dev_id);
  clinfo_boundary_copy = new CLMPIinfo(dev_id);

  std::string filen, kerneln;
  clinfo_generic->guess_kernelfile(argc, argv, filen, kerneln);

  clinfo_generic->create_program(filen);
  clinfo_inner->create_program(filen);
  clinfo_boundary_copy->create_program(filen);

  num_clinfo_boundary_kernel = NUM_CLINFO_BOUNDARY_KERNEL;
  unsigned int i;
  for (i = 0; i < num_clinfo_boundary_kernel; i++) {
      CLMPIinfo *clinfo_bk_new = new CLMPIinfo(dev_id);
      clinfo_bk_new->create_program(filen);
      clinfo_boundary_kernel.push_back(clinfo_bk_new);
  }

  // Default
  clinfo_nowusing = clinfo_generic;
}

void DestroyOpenCL(void) {
  delete clinfo_generic;
  delete clinfo_inner;
  delete clinfo_boundary_copy;

  FOREACH(it, clinfo_boundary_kernel.begin(), clinfo_boundary_kernel.end())
  {
    CLMPIinfo *cl_deleteNow = *it;
    delete cl_deleteNow;
  }
  clinfo_boundary_kernel.clear();
}

#ifdef __cplusplus
extern "C" {
#endif

void __PSSetKernel(
      const char *kernelname,
      enum CL_STREAM_FLAG strm_flg, unsigned int strm_num)
{
    std::string string_kernelname = kernelname;
    CLMPIinfo *cl_it;
    switch(strm_flg) {
      case USE_GENERIC:
        clinfo_nowusing = clinfo_generic;
        break;
      case USE_INNER:
        clinfo_nowusing = clinfo_inner;
        break;
      case USE_BOUNDARY_COPY:
        clinfo_nowusing = clinfo_boundary_copy;
      case USE_BOUNDARY_KERNEL:
        if (strm_num >= num_clinfo_boundary_kernel) {
          clinfo_nowusing = clinfo_generic;
          break;
        }

        cl_it = clinfo_generic;
        ENUMERATE(j, it, clinfo_boundary_kernel.begin(), clinfo_boundary_kernel.end()) {
          cl_it = *it;
          if ((unsigned int) j >= strm_num) break;
        }
        clinfo_nowusing = cl_it;
        break;

      default:
        clinfo_nowusing = clinfo_generic;
        break;
    }
    clinfo_nowusing->create_kernel(string_kernelname);
} // __PSSetKernel

void __PSSetKernelArg(unsigned int arg_index, size_t arg_size, const void *arg_val)
{
    clinfo_nowusing->SetKernelArg((cl_uint) arg_index, arg_size, arg_val);
} // __PSSetKernelArg

void __PSSetKernelArgCLMem(unsigned int arg_index, const void *arg_val){
    clinfo_nowusing->SetKernelArg((cl_uint) arg_index, sizeof(cl_mem), arg_val);
} // __PSSetKernelArgCLMem

void __PSRunKernel(size_t *globalsize, size_t *localsize){
    clinfo_nowusing->RunKernel(globalsize, localsize);
} // __PSRunKernel


#ifdef __cplusplus
}
#endif


} // namespace
