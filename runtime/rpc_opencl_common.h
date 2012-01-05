#ifndef PHYSIS_RUNTIME_RPC_OPENCL_COMMON_H_
#define PHYSIS_RUNTIME_RPC_OPENCL_COMMON_H_

/* C or C++ standard headers */
#include <cstdarg>
#include <string>

/* OpenCL specific */
#include <CL/cl.h>

/* physis specific headers */
#include "runtime/runtime_common.h"

/*
  Note:
  CLbaseinfo: OpenCL basic information class
  Members or functions not using __PSGrid belongs to CLbaseinfo.
  Members or functions using __PSGrid should belong to
  CLbase, which should inherit CLbaseinfo.
*/

namespace physis {
  namespace runtime {

    class CLbaseinfo {
      protected:
        int err_status;
        cl_device_id cldevid;
        cl_context clcontext;
        cl_command_queue clqueue;
        cl_kernel clkernel;
        cl_program clprog;

        virtual void create_context_queue(unsigned int id_default = 0);
        virtual void cleanupcl(void);

        virtual std::string physis_opencl_h_include_path(void) const ;
        virtual std::string create_kernel_contents(std::string kernelfile) const;

      public:
        CLbaseinfo(unsigned int id_default = 0);
        virtual ~CLbaseinfo();
        virtual int get_status(void) const { return err_status; }

        virtual void guess_kernelfile(const int *argc, char ***argv, 
          std::string &filename, std::string &kernelname) const;

        virtual void SetKernelArg(cl_uint arg_index, size_t arg_size, const void *arg_val);
        virtual void create_program(std::string kernelfile);
        virtual void create_kernel(std::string kernelname);
        virtual void RunKernel(size_t *globalsize, size_t *localsize);

        virtual cl_context get_context() { return clcontext; }
        virtual cl_command_queue get_queue() { return clqueue; }

    }; // class CLbaseinfo

  } // namespace runtime
} // namespace physis

#endif /* #define PHYSIS_RUNTIME_RPC_OPENCL_COMMON_H_ */
