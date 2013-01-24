#include "runtime/mpi_opencl_runtime.h"
#include "physis/physis_mpi_opencl.h"

namespace phru = physis::runtime;

namespace physis {
namespace runtime {

void InitOpenCL(
    int my_rank, int num_local_processes, int *argc, char ***argv
                )
{
  // Assumes each local process has successive process rank
  int dev_id = my_rank % num_local_processes;

  LOG_DEBUG() << "Creating generic clqueue.\n";
  phru::clinfo_generic = new phru::CLMPIbaseinfo(dev_id, 1, 0);
  phru::clinfo_inner = 0;
  phru::clinfo_boundary_copy = 0;

  std::string filen, kerneln;
  phru::clinfo_generic->guess_kernelfile(argc, argv, filen, kerneln);
  phru::clinfo_generic->set_kernel_filen(filen);


  num_clinfo_boundary_kernel = NUM_CLINFO_BOUNDARY_KERNEL;
  unsigned int i;
  // FIXME
  // FIXME
  // Get this back later
  for (i = 0; i < ::num_clinfo_boundary_kernel; i++) {
  }

  // Default
  phru::clinfo_nowusing = phru::clinfo_generic;

#if 1
  /* As currently header_path is always 0, move this
     to InitOpencl
  */
  if (!phru::clinfo_nowusing->get_prog()) {
    phru::clinfo_nowusing->set_header_include_path(0);
    std::string kernel_filen = phru::clinfo_nowusing->get_kernel_filen();
    phru::clinfo_nowusing->create_program(kernel_filen);
  }
#endif
}

void DestroyOpenCL(void) {
  if (phru::clinfo_generic)
    delete phru::clinfo_generic;
  if (phru::clinfo_inner)
    delete phru::clinfo_inner;
  if (phru::clinfo_boundary_copy)
    delete phru::clinfo_boundary_copy;

  FOREACH(it, phru::clinfo_boundary_kernel.begin(), phru::clinfo_boundary_kernel.end())
  {
    CLMPIbaseinfo *cl_deleteNow = *it;
    delete cl_deleteNow;
  }
  phru::clinfo_boundary_kernel.clear();
}

} // namespace runtime
} // namespace physis

#ifdef __cplusplus
extern "C" {
#endif

  void __PSSetKernel(
      const char *kernelname,
      enum CL_STREAM_FLAG strm_flg, unsigned int strm_num,
      const char *header_path)
  {
    std::string string_kernelname = kernelname;

    PSAssert(phru::clinfo_generic);

    phru::clinfo_nowusing = phru::clinfo_generic;
#if 0
    /* As currently header_path is always 0, move this
       to InitOpencl
    */
    if (!phru::clinfo_nowusing->get_prog()) {
      phru::clinfo_nowusing->set_header_include_path(header_path);
      std::string kernel_filen = phru::clinfo_nowusing->get_kernel_filen();
      phru::clinfo_nowusing->create_program(kernel_filen);
    }
#endif

    switch(strm_flg) {
      case USE_GENERIC:
        phru::clinfo_nowusing = phru::clinfo_generic;
        break;

      case USE_INNER:
        if (!phru::clinfo_inner) {
          LOG_DEBUG() << "Creating inner clqueue.\n";
          phru::clinfo_inner = new phru::CLMPIbaseinfo(*(phru::clinfo_generic));
        }
        phru::clinfo_nowusing = phru::clinfo_inner;
        break;

      case USE_BOUNDARY_COPY:
        if (!phru::clinfo_boundary_copy) {
          LOG_DEBUG() << "Creating boundary_copy clqueue.\n";
          phru::clinfo_boundary_copy = new phru::CLMPIbaseinfo(*(phru::clinfo_generic));
        }
        phru::clinfo_nowusing = phru::clinfo_boundary_copy;
        break;

      case USE_BOUNDARY_KERNEL:
        if (strm_num >= num_clinfo_boundary_kernel) {
          phru::clinfo_nowusing = phru::clinfo_generic;
          break;
        }

        while (phru::clinfo_boundary_kernel.size() < strm_num) {
          LOG_DEBUG() << "Creating boundary_kernel clqueue.\n";
          phru::CLMPIbaseinfo *clinfo_bk_new = new phru::CLMPIbaseinfo(*(phru::clinfo_generic));
          phru::clinfo_boundary_kernel.push_back(clinfo_bk_new);
        }

        {
          phru::CLMPIbaseinfo *cl_it = phru::clinfo_generic;
          ENUMERATE(j, it, phru::clinfo_boundary_kernel.begin(), phru::clinfo_boundary_kernel.end()) {
            cl_it = *it;
            if ((unsigned int) j >= strm_num) break;
          }
          phru::clinfo_nowusing = cl_it;
        }
        break;

      default:
        LOG_DEBUG() << "The flag value " << strm_flg << 
            " does not match any of the registered values.\n";
        PSAbort(1);

        phru::clinfo_nowusing = phru::clinfo_generic;
        break;
    }

    phru::clinfo_nowusing->create_kernel(string_kernelname);
  } // __PSSetKernel

  void __PS_CL_ThreadSynchronize(void){
    if (phru::clinfo_generic)
      phru::clinfo_generic->sync_queue();
    if (phru::clinfo_inner)
      phru::clinfo_inner->sync_queue();
    if (phru::clinfo_boundary_copy)
      phru::clinfo_boundary_copy->sync_queue();

    FOREACH(it, phru::clinfo_boundary_kernel.begin(), phru::clinfo_boundary_kernel.end())
    {
      phru::CLMPIbaseinfo *cl_usenow = *it;
      cl_usenow->sync_queue();
    }
  }

  void __PSSetKernelArg(unsigned int arg_index, size_t arg_size, const void *arg_val)
  {
    phru::clinfo_nowusing->SetKernelArg((cl_uint) arg_index, arg_size, arg_val);
  } // __PSSetKernelArg

  void __PSSetKernelArgCLMem(unsigned int arg_index, const void *arg_val){
    phru::clinfo_nowusing->SetKernelArg((cl_uint) arg_index, sizeof(cl_mem), arg_val);
  } // __PSSetKernelArgCLMem

  // Float case
  void __PSSetKernelArg_Grid3DFloat(unsigned int *p_argc, __PSGrid3DFloatDev *g) {
    unsigned int argc = *p_argc;
    int dim = 0;
    int fwbw = 0;

    { void *buf = g->p0 ; __PSSetKernelArgCLMem(argc, (void *)(&buf)); argc++; }
    for (dim = 0; dim < 3; dim++) {
      cl_long j = g->dim[dim]; __PSSetKernelArg(argc, sizeof(j), &j); argc++;
    }
    for (dim = 0; dim < 3; dim++) {
      cl_long j = g->local_size[dim]; __PSSetKernelArg(argc, sizeof(j), &j); argc++;
    }
    for (dim = 0; dim < 3; dim++) {
      cl_long j = g->local_offset[dim]; __PSSetKernelArg(argc, sizeof(j), &j); argc++;
    }
    { cl_long j = g->pitch; __PSSetKernelArg(argc, sizeof(j), &j); argc++; }

    for (dim = 0; dim < 3; dim++) {
      for (fwbw = 0 ; fwbw < 2; fwbw++) {
        void *buf = 0;
        if (g->halo && g->halo[dim])
          buf = g->halo[dim][fwbw]; 
        if (buf)
          __PSSetKernelArgCLMem(argc, (void *)(&buf));
        else {
          void *buf_tmp = g->p0;
          // if g->p0 is still NULL, given up
          if (buf_tmp)
            __PSSetKernelArgCLMem(argc, (void *)(&buf_tmp));
        }
        argc++;
        {
          cl_long j = 1;
          if (!buf)
            j = 0;
          __PSSetKernelArg(argc, sizeof(j), &j);
          argc++;
        }
      } // for (fwbw = 0 ; fwbw < 2; fwbw++)
    } // for (dim = 0; dim < 3; dim++)
    for (dim = 0; dim < 3; dim++) {
      for (fwbw = 0 ; fwbw < 2; fwbw++) {
        cl_long j = g->halo_width[dim][fwbw];
        __PSSetKernelArg(argc, sizeof(j), &j); argc++;
      }
    }
    { cl_long j = g->diag; __PSSetKernelArg(argc, sizeof(j), &j); argc++; }

    *p_argc = argc;
  } // __PSSetKernelArg_Grid3DFloat

  // Double case
  void __PSSetKernelArg_Grid3DDouble(unsigned int *p_argc, __PSGrid3DDoubleDev *g) {
    unsigned int argc = *p_argc;
    int dim = 0;
    int fwbw = 0;

    { void *buf = g->p0 ; __PSSetKernelArgCLMem(argc, (void *)(&buf)); argc++; }
    for (dim = 0; dim < 3; dim++) {
      cl_long j = g->dim[dim]; __PSSetKernelArg(argc, sizeof(j), &j); argc++;
    }
    for (dim = 0; dim < 3; dim++) {
      cl_long j = g->local_size[dim]; __PSSetKernelArg(argc, sizeof(j), &j); argc++;
    }
    for (dim = 0; dim < 3; dim++) {
      cl_long j = g->local_offset[dim]; __PSSetKernelArg(argc, sizeof(j), &j); argc++;
    }
    { cl_long j = g->pitch; __PSSetKernelArg(argc, sizeof(j), &j); argc++; }

    for (dim = 0; dim < 3; dim++) {
      for (fwbw = 0 ; fwbw < 2; fwbw++) {
        void *buf = 0;
        if (g->halo && g->halo[dim])
          buf = g->halo[dim][fwbw]; 
        if (buf)
          __PSSetKernelArgCLMem(argc, (void *)(&buf));
        else {
          void *buf_tmp = g->p0;
          // if g->p0 is still NULL, given up
          if (buf_tmp)
            __PSSetKernelArgCLMem(argc, (void *)(&buf_tmp));
        }
        argc++;
        {
          cl_long j = 1;
          if (!buf)
            j = 0;
          __PSSetKernelArg(argc, sizeof(j), &j);
          argc++;
        }
      } // for (fwbw = 0 ; fwbw < 2; fwbw++)
    } // for (dim = 0; dim < 3; dim++)
    for (dim = 0; dim < 3; dim++) {
      for (fwbw = 0 ; fwbw < 2; fwbw++) {
        cl_long j = g->halo_width[dim][fwbw];
        __PSSetKernelArg(argc, sizeof(j), &j); argc++;
      }
    }
    { cl_long j = g->diag; __PSSetKernelArg(argc, sizeof(j), &j); argc++; }

    *p_argc = argc;
  } // __PSSetKernelArg_Grid3DDouble

  void __PSSetKernelArg_Dom(unsigned int *p_argc, __PSDomain *p_dom) {
    unsigned int argc = *p_argc;
    int dim = 0;

    for (dim = 0; dim < 3; dim++) {
      cl_long j;

      j= p_dom->local_min[dim];
      __PSSetKernelArg(argc, sizeof(j), &j); argc++;
      j= p_dom->local_max[dim];
      __PSSetKernelArg(argc, sizeof(j), &j); argc++;
    }

    *p_argc = argc;
  } // __PSSetKernelArg_Dom

  void __PSRunKernel(size_t *globalsize, size_t *localsize){
    phru::clinfo_nowusing->RunKernel(globalsize, localsize);
  } // __PSRunKernel


#ifdef __cplusplus
}
#endif

