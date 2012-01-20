/* C or C++ standard headers */
#include <cstdio>
#include <string>

/* physis-OpenCL specific */
#include "runtime/rpc_opencl.h"

#define SOURCE_ARRAY_SIZE  10
#define BUF_SIZE 1024

namespace physis {
  namespace runtime {

    std::string CLbaseinfo::create_kernel_contents(std::string kernelfile) const {
      std::string ret_str = "";
      int num = 0;

      // Write all header files to be included
      // The last element must be NULL


      // FIXME
      // Currently no header files are needed for physis opencl kernel code
      // because all structures ]re expanded into individual elements
      // in kernel code

      const char *header_lists[] = {
#if 0
#endif
        NULL
      };

    // Add needed definitions
    ret_str += "#define PHYSIS_OPENCL\n";
    ret_str += "#define PHYSIS_OPENCL_KERNEL_MODE\n";
    ret_str += "#undef PHYSIS_USER\n";
#if 0
#endif
    // For double usage
    ret_str += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

      // read header files and source file (kernelfile),
      // add the whole contents ret_str
      do {
          for (num = 0; num < SOURCE_ARRAY_SIZE - 1; num++) {
            FILE *fin;
            size_t size = 0;
            const char *file_read = NULL;


            // FIXME
            // Currently no header files are included
#if 0
            if (header_lists[num]) {
                std::string str = physis_opencl_h_include_path();
                str += "/";
                str += header_lists[num];
                file_read = str.c_str();
#else
            if (0) {
#endif
            } else {
                file_read = kernelfile.c_str();
            }

            fin = fopen(file_read, "r");
            if (!fin) {
              fprintf(stderr, "fopen()ing file %s failed\n", file_read);
              break;
              }

            while (1) {
              char cbuf[BUF_SIZE];
              size = fread(cbuf, 1, BUF_SIZE - 1, fin);
              cbuf[size] = 0;
              ret_str += cbuf;
              if (size < BUF_SIZE - 1) break;
              }
            fclose(fin);

            // Insert new line
            ret_str += "\n";

            // exit this loop
            if (!header_lists[num]) {
              num++;
              break;
              }

          }; // for(num = 0; num < SOURCE_ARRAY_SIZE - 1; num++)

      } while(0);

      return ret_str;
    } // create_kernel_contents


    void CLbaseinfo::create_program(std::string kernelfile) {
      std::string str_kern = "";
      const char *cchar_kern = NULL;
      size_t size_kern = 0;

      cl_int status = -1;
      err_status = 1;

      // First read the whole kernel code, save it to str_kern
      str_kern = create_kernel_contents(kernelfile);
      cchar_kern = str_kern.c_str();
      size_kern = str_kern.size();
       

      do {
        // Create program
        LOG_DEBUG() << "Calling clCreateProgram\n";
        clprog = clCreateProgramWithSource(clcontext, 1, &cchar_kern, &size_kern, &status);
        if (status != CL_SUCCESS) {
          fprintf(stderr, "Calling clCreateProgramWithSource failed: status %i.\n", status);
          break;
        }

        // make build options for clBuildProgram
        // Currently can be NULL

        // build program
        LOG_DEBUG() << "Calling clBuildProgram\n";
        status = clBuildProgram(clprog, 0, NULL, NULL, NULL, NULL);
        if (status != CL_SUCCESS) {
          fprintf(stderr, "Calling clBuildProgram failed: status %i.\n", status);
          do {
            size_t len;
            char *log_out = NULL;
            cl_program_build_info infotype = CL_PROGRAM_BUILD_LOG;
            status = clGetProgramBuildInfo(clprog, cldevid, infotype, 0, NULL, &len);
            if (status != CL_SUCCESS) {
              fprintf(stderr, "Calling clGetProgramBuildInfo failed: status %i\n", status);
              break;
            }
            if (!len) break;
            log_out = (char *) malloc(len);
            if (!log_out) {
              fprintf(stderr, "Allocating memory failed for log_out.\n");
              break;
            }
            status = clGetProgramBuildInfo(clprog, cldevid, infotype, len, log_out, NULL);
            if (status != CL_SUCCESS) {
              fprintf(stderr, "Calling clGetProgramBuildInfo failed: status %i.\n", status);
              break;
            }
            fprintf(stderr, "Build log:\n%s\n", log_out);
            free(log_out);
          } while(0);
          break;
        }

      } while (0);

      if (status == CL_SUCCESS) err_status = 0;

    } // create_program


    void CLbaseinfo::create_kernel(std::string kernelname) {
      cl_int status = -1;
      err_status = 1;

      // PSStencilMap may be called with different kernel
      // So first release the previous kernel (if any)
      if (clkernel) clReleaseKernel(clkernel);

      // Now create kernel
      do {
        LOG_DEBUG() << "Calling clCreateKernel\n";
        clkernel = clCreateKernel(clprog, kernelname.c_str(), &status);
        if (status != CL_SUCCESS) {
          fprintf(stderr, "Calling clCreateKernel failed: status %i.\n", status);
          break;
        }
      } while (0);

      if (status == CL_SUCCESS) err_status = 0;

    } // create_kernel

    void CLbaseinfo::SetKernelArg(cl_uint arg_index, size_t arg_size, const void *arg_val){
      cl_int status = -1;
      err_status = 1;

#if 0
      LOG_DEBUG() << "Calling clSetKernelArg() for " << arg_index
        << " with size " << arg_size << " .\n";
#endif
      status = clSetKernelArg(clkernel, arg_index, arg_size, arg_val);
      if (status == CL_SUCCESS) {
        err_status = 0;
      } else {
        fprintf(
          stderr, 
          "Calling clSetKernelArg failed for index %i with size %zi:"
          " status %i\n", arg_index, arg_size, status);
      }
    } // SetKernelArg

    void CLbaseinfo::RunKernel(size_t *globalsize, size_t *localsize) {
      cl_int status = -1;
      err_status = 1;

      // Now use dimension 2
      LOG_DEBUG() << "Calling clEnqueueNDRangeKernel.\n";
      status = clEnqueueNDRangeKernel(clqueue, clkernel, 2, NULL, globalsize, localsize, 0, NULL, NULL);
      if (status == CL_SUCCESS) {
        err_status = 0;
      } else {
        fprintf(stderr, "Calling clEnqueueNDRangeKernel failed: status %i\n", status);
      }

      // FIXME
      // TODO
      // Block
      if (cl_block_events_p)
        clFinish(clqueue);
      else
        clEnqueueBarrier(clqueue);

    } // RunKernel


  } // namespace physis

} // namespace runtime
