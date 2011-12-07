/* C or C++ standard headers */
#include <cstdio>
#include <string>

/* physis-OpenCL specific */
#include "runtime/rpc_opencl.h"
#define INFOSIZE 1024

namespace physis {
  namespace runtime {

    CLinfo::CLinfo() {
      cldevid = 0;
      clcontext = NULL;
      clqueue = NULL;
      clkernel = NULL;
      clprog = NULL;
      err_status = 0;

      // create context and queue
      create_context_queue();

    } // CLinfo()

    CLinfo::~CLinfo() {
      cleanupcl();
    } // ~CLinfo()

    void CLinfo::create_context_queue(void) {
      cl_context context = (cl_context) NULL;
      cl_device_id *dev_ids = (cl_device_id *) NULL;
      cl_device_type devtype = CL_DEVICE_TYPE_GPU;

      cl_platform_id *platform_ids = (cl_platform_id *) NULL;
      cl_platform_id selected_platform_id = (cl_platform_id) NULL;

      cl_command_queue queue = (cl_command_queue) NULL;

      cl_int status = -1;
      cl_uint num_platforms = 0;
      cl_uint num_devices = 0;

      err_status = 1;


      /*
        1 obtain platform ID
           * the function oclGetPlatformID() is not in OpenCL specification and is NVIDIA's
             original implementation, license being unclear
        2 obtain device ID (and the number of devices available)
        3 create the device list
        4 create context from the device list
        5 create command queue

       99 then set context and command queue
      */

      // obtain platform ID
      do {
        cl_uint id_pos = 0;

        status = clGetPlatformIDs(0, NULL, &num_platforms);
        if (status != CL_SUCCESS) {
          fprintf(stderr, "Calling clGetPlatformIDs() failed: status %i.\n", status);
          break;
        }
        if (num_platforms == 0) {
          fprintf(stderr, "No OpenCL platforms were found.\n");
          break;
        }

        platform_ids = new cl_platform_id[num_platforms];
        status = clGetPlatformIDs(num_platforms, platform_ids, NULL);
        if (status != CL_SUCCESS) {
          fprintf(stderr, "Calling clGetPlatformIDs() failed: status %i\n", status);
          break;
        }

        for (id_pos = 0; id_pos < num_platforms; id_pos++) {
          int found = 0;
          selected_platform_id = platform_ids[id_pos];

          do {
            // obtain device ID (and the number of devices available)
            status = clGetDeviceIDs(selected_platform_id, devtype, 0, NULL, &num_devices);
            if (status != CL_SUCCESS){
              fprintf(stderr, "Calling clGetDeviceIDs() failed: status  %i\n", status);
              break;
            }
            if (num_devices == 0) {
              fprintf(stderr, "No devices were found\n");
              break;
            }

            // Create the device list
            dev_ids = new cl_device_id[num_devices];
            status = clGetDeviceIDs(selected_platform_id, devtype, num_devices, dev_ids, NULL);
            if (status != CL_SUCCESS){
              fprintf(stderr, "Calling clGetDeviceIDs() failed: status  %i\n", status);
              break;
            }

            // Create context
            context = clCreateContext(NULL, num_devices, dev_ids, NULL, NULL, &status);
            if (status != CL_SUCCESS){
              fprintf(stderr, "Calling clCreteContext() failed: status %i\n", status);
              break;
            }

            // Create command queue
            // For now, use the 0th device from device list dev_ids
            cldevid = dev_ids[0];
            queue = clCreateCommandQueue(context, cldevid, NULL, &status);
            if (status != CL_SUCCESS){
              fprintf(stderr, "Calling clCreateCommandQueue failed: status %i\n", status);
              break;
            }

            // Okay, now GPU device is found and context, command queue successfully created.
            found = 1;

          } while(0);

          // Cleanup
          if (platform_ids) delete [] (platform_ids);
          if (dev_ids) delete [] (dev_ids);
          platform_ids =  (cl_platform_id *) NULL;
          dev_ids = (cl_device_id *) NULL;

          if (found == 1) {
            // Initialization succeeded.
            // FIXME: throwing to stderr for now
            fprintf(stderr, "Initializing OpenCL platform and devices succeeded\n");
            err_status = 0;
            break;
          }

        } // for (id_pos = 0; id_pos < num_platforms; id_pos++)

      } while(0);

      // Cleanup
      if (platform_ids) delete [] (platform_ids);
      if (dev_ids) delete [] (dev_ids);
      if (err_status) {
        fprintf(stderr, "Initializing OpenCL platform or devices failed.\n");
      }

      // set context and queue
      clcontext = context;
      clqueue = queue;

    } // create_context_queue()


    void CLinfo::cleanupcl(void) {

      if (clkernel) clReleaseKernel(clkernel);
      if (clprog) clReleaseProgram(clprog);
      if (clqueue) clReleaseCommandQueue(clqueue);
      if (clcontext) clReleaseContext(clcontext);

      clkernel = NULL;
      clprog = NULL;
      clqueue = NULL;
      clcontext = NULL;

    } // cleanupcl()

  } // namespace physis

} // namespace runtime
