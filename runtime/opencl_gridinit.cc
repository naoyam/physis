/* C or C++ standard headers */
#include <cstdio>

/* physis-OpenCL specific */
#include "runtime/rpc_opencl.h"

namespace physis {
  namespace runtime {

    // PSGrid3DRealNew(NX, NX, NX); is translated into
    // int dims[3] = {((index_t )NX), ((index_t )NX), ((index_t )NX)};
    // g = __PSGridNew(sizeof(float), 3, dims, 0);

    __PSGrid* CLinfo::GridNew(int elm_size, int num_dims, PSVectorInt dim, int grid_attribute) {
      /*
        1. Allocate grid
        2. Initialize grid
        3. Allocate memory on the device
        4. Set the on-device data (g->devptr)

       99. return allocated grid
      */

      int i = 0;
      cl_int status = -1;
      size_t size = 0;
      __PSGrid *g = NULL;
      err_status = 1;

      do {
        // Allocate grid
        g = (__PSGrid *)malloc(sizeof(__PSGrid));
        if (!g) {
          fprintf(stderr, "Allocating memory failed for __PSGrid.\n");
          break;
        }

        // Initialize grid
        g->elm_size = elm_size;
        g->num_dims = num_dims;
        PSVectorIntCopy(g->dim, dim);
        g->devptr = NULL;
        g->num_elms = 1;
        g->gridattr = grid_attribute;
        g->buf = NULL;

        for (i = 0; i < g->num_dims; i++) {
          g->num_elms *= dim[i];
        }

        // Allocate memory on the device
        do {
          cl_mem dev_mem = (cl_mem) NULL;
          g->buf = NULL;
          size = g->num_elms * g->elm_size;

          // 4th argument can be NULL here: we won't share memory on CPU and device
          dev_mem = clCreateBuffer(clcontext, CL_MEM_READ_WRITE, size, NULL, &status);
          if (status != CL_SUCCESS) {
            fprintf(stderr, "Calling clCreateBuffer failed for buffer %i: status %i\n", i, status);
            break;
          }
          g->buf = dev_mem;
        } while(0);

        if (status != CL_SUCCESS) break;

        // Set the on-device data. g->devptr data will be copied when calling
        // OpenCL global functions.
        status = -1;
        switch (g->num_dims) {
          case 1:
            g->devptr = malloc(sizeof(__PSGrid1DDev));
            if (!g->devptr) {
              fprintf(stderr, "Allocating memory failed for __PSGrid1DDev.\n");
              break;
            }
            ((__PSGrid1DDev*)g->devptr)->buf = g->buf;
            for (i = 0; i < g->num_dims; i++)
              ((__PSGrid1DDev*)g->devptr)->dim[i] = g->dim[i];
            status = CL_SUCCESS;
            break;

          case 2:
            g->devptr = malloc(sizeof(__PSGrid2DDev));
            if (!g->devptr) {
              fprintf(stderr, "Allocating memory failed for __PSGrid2DDev.\n");
              break;
            }
            ((__PSGrid2DDev*)g->devptr)->buf = g->buf;
            for (i = 0; i < g->num_dims; i++)
              ((__PSGrid2DDev*)g->devptr)->dim[i] = g->dim[i];
            status = CL_SUCCESS;
            break;

          case 3:
            g->devptr = malloc(sizeof(__PSGrid3DDev));
            if (!g->devptr) {
              fprintf(stderr, "Allocating memory failed for __PSGrid3DDev.\n");
              break;
            }
            ((__PSGrid3DDev*)g->devptr)->buf = g->buf;
            for (i = 0; i < g->num_dims; i++)
              ((__PSGrid3DDev*)g->devptr)->dim[i] = g->dim[i];
            status = CL_SUCCESS;
            break;

          default:
            LOG_ERROR() << "Unsupported dimension: " << g->num_dims << "\n";
            PSAbort(1);
         } // switch (g->num_dims)

          if (status != CL_SUCCESS) break;

          // Initialization finished
          err_status = 0;
        } while (0);


        return g;

    } // GridNew()

    void CLinfo::GridFree(__PSGrid *g) {
      cl_mem dev_mem = (cl_mem) NULL;

      if (!g) return;

      // Release memory objects on devices
      dev_mem = (cl_mem) g->buf;
      if (dev_mem) clReleaseMemObject(dev_mem);

      // Free g itself
      free(g->devptr);
      free(g);

    } // GridFree

  } // namespace physis
} // namespace runtime
