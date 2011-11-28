#ifndef PHYSIS_RUNTIME_RPC_OPENCL_H_
#define PHYSIS_RUNTIME_RPC_OPENCL_H_

/* C or C++ standard headers */
#include <cstdarg>
#include <string>

/* OpenCL specific */
#include <CL/cl.h>

/* physis specific headers */
#include "runtime/runtime_common.h"

/* physis-OpenCL specific*/
#include "physis/physis_opencl.h"

namespace physis {
  namespace runtime {

    class CLinfo {
      protected:

      private:
        int err_status;
        cl_device_id cldevid;
        cl_context clcontext;
        cl_command_queue clqueue;
        cl_kernel clkernel;
        cl_program clprog;

        virtual void create_context_queue(void);
        virtual void cleanupcl(void);

        virtual std::string physis_opencl_h_include_path(void);
        virtual std::string create_kernel_contents(std::string kernelfile);

      public:
        CLinfo();
        virtual ~CLinfo();
        virtual int get_status(void) { return err_status; }

        virtual void guess_kernelfile(const int *argc, char ***argv, 
          std::string &filename, std::string &kernelname);

        virtual __PSGrid* GridNew(int elm_size, int num_dims, PSVectorInt dim, int double_buffering);
        virtual void GridFree(__PSGrid *g);
        virtual void GridCopyin(__PSGrid *g, const void *src_buf);
        virtual void GridCopyout(__PSGrid *g, void *dst_buf);
        virtual void GridSet(__PSGrid *g, const void *val_ptr, va_list valst_dim);

        virtual void SetKernelArg(cl_uint arg_index, size_t arg_size, const void *arg_val);
        virtual void create_program(std::string kernelfile);
        virtual void create_kernel(std::string kernelname);
        virtual void RunKernel(size_t *globalsize, size_t *localsize);

    }; // class CLinfo

  } // namespace runtime
} // namespace physis

#endif /* #define PHYSIS_RUNTIME_RPC_OPENCL_H_ */
