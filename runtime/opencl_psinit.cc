// Licensed under the BSD license. See LICENSE.txt for more details.

/* C or C++ standard headers */
#include <cstdio>
#include <string>

/* physis-OpenCL specific */
#include "runtime/rpc_opencl.h"
#define INFOSIZE 1024

namespace physis {
namespace runtime {

CLbaseinfo::CLbaseinfo():
    err_status(0),
    cldevid(0),
    clcontext(0),
    clqueue(0),
    clkernel(0),
    clprog(0),
    cl_num_platforms(0),
    cl_pl_ids(0),
    cl_num_devices(0),
    cl_dev_ids(0),
    cl_block_events_p(1)
{
  // create context and queue
  create_context_queue(0);
} // CLbaseinfo()

CLbaseinfo::CLbaseinfo(unsigned int id_use, unsigned int create_queue_p, unsigned int block_events_p):
    err_status(0),
    cldevid(0),
    clcontext(0),
    clqueue(0),
    clkernel(0),
    clprog(0),
    cl_num_platforms(0),
    cl_pl_ids(0),
    cl_num_devices(0),
    cl_dev_ids(0),
    cl_block_events_p(block_events_p)
{
  // create context and queue
  if (create_queue_p) 
    create_context_queue(id_use);
} // CLbaseinfo()

CLbaseinfo::~CLbaseinfo() {
  cleanupcl();
} // ~CLbaseinfo()


void CLbaseinfo::create_context_queue(cl_uint id_use) {
  // It seems that get_platforms_ids takes much longer
  // than the rest parts (create_context_queue_from_platform_ids),
  // and as get_platform_ids has to be called only once per
  // platform, split get_platform_ids part out.

  // clcontext must be shared with queues, so this is also split
  // out

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

  get_platform_ids(cl_num_platforms, cl_pl_ids);
  create_context_from_pls(
      cl_num_platforms, cl_pl_ids,
      cl_num_devices, cl_dev_ids, clcontext
                          );
  clqueue = create_queue_from_context(id_use, cl_dev_ids, clcontext);
}

void CLbaseinfo::get_platform_ids(cl_uint &num_platforms, cl_platform_id *&platform_ids){

  cl_int status = -1;

  /*
    1 obtain platform ID
    * the function oclGetPlatformID() is not in OpenCL specification and is NVIDIA's
    original implementation, license being unclear
  */

  do {
    LOG_DEBUG() << "Getting platform ids.\n";
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
  } while(0);

} // get_platform_ids

void CLbaseinfo::create_context_from_pls(
    cl_uint num_platforms, cl_platform_id *cl_pl_ids_in,
    cl_uint &num_devices, cl_device_id *&dev_ids, cl_context &context
                                         )
{
  cl_int status = -1;

  /*
    2 obtain device ID (and the number of devices available)
    3 create the device list
    4 create context from the device list
  */

  cl_platform_id *platform_ids = cl_pl_ids_in;
  cl_platform_id selected_platform_id = (cl_platform_id) NULL;
  cl_device_type devtype = CL_DEVICE_TYPE_GPU;

  num_devices = 0;
  dev_ids = 0;
  context = 0;


  do {
    cl_uint id_pos = 0;

    if (!platform_ids) break; // Immidiate jump

    for (id_pos = 0; id_pos < num_platforms; id_pos++) {
      selected_platform_id = platform_ids[id_pos];

      LOG_DEBUG() << "Getting device ids.\n";
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
        LOG_DEBUG() << "Creating context.\n";
        context = clCreateContext(NULL, num_devices, dev_ids, NULL, NULL, &status);
        if (status != CL_SUCCESS){
          fprintf(stderr, "Calling clCreteContext() failed: status %i\n", status);
          break;
        }

        // succeeded
        status = 0;

      } while(0);

    } // for (id_pos = 0; id_pos < num_platforms; id_pos++)

  } while(0);

} // create_context_from_pls

cl_command_queue CLbaseinfo::create_queue_from_context(
    unsigned int id_dev_use, cl_device_id *dev_ids, cl_context context
                                                       ) {

  cl_int status = -1;
  cl_command_queue ret_queue = 0;

  /*
    5 create command queue
  */

  do {
    if (!dev_ids) break; // Immediate jump
    if (!context) break; // Immediate jump
    // Create command queue
    // By default, use the 0th device from device list dev_ids
    LOG_DEBUG() << "Creating command queue.\n";
    LOG_DEBUG() << "Using device id " << id_dev_use << " .\n";
    cldevid = dev_ids[id_dev_use];
    ret_queue = clCreateCommandQueue(context, cldevid, NULL, &status);
    if (status != CL_SUCCESS){
      fprintf(stderr, "Calling clCreateCommandQueue failed: status %i\n", status);
      break;
    } 
  } while (0);

  if (!ret_queue) {
    fprintf(stderr, "Initializing OpenCL platform or devices failed.\n");
  }

  return ret_queue;
}


void CLbaseinfo::cleanupcl(void) {


  if (clkernel) clReleaseKernel(clkernel);
  if (clprog) clReleaseProgram(clprog);
  if (clqueue) clReleaseCommandQueue(clqueue);


  if (clcontext) clReleaseContext(clcontext);
  if (cl_dev_ids) delete[] (cl_dev_ids);
  if (cl_pl_ids) delete[] (cl_pl_ids);


  clkernel = 0;
  clprog = 0;
  clqueue = 0;

  clcontext = 0;

  cl_pl_ids = 0;
  cl_dev_ids = 0;

} // cleanupcl()

} // namespace physis

} // namespace runtime
