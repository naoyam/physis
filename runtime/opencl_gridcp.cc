/* physis-OpenCL specific */
#include "runtime/rpc_opencl.h"

namespace physis {
  namespace runtime {

    void CLinfo::GridCopyin(__PSGrid *g, const void *src_buf) {

      // Copy in the data src_buf to device memory object g->buf
      cl_mem dev_mem = (cl_mem) g->buf;
      size_t size_write = g->elm_size * g->num_elms;
      cl_int status = -1;
      err_status = 1;

      status = clEnqueueWriteBuffer(clqueue, dev_mem, CL_TRUE, 
        0, size_write, src_buf,
        0, NULL, NULL);
      if (status == CL_SUCCESS) {
        err_status = 0;
      } else {
        fprintf(stderr, "Calling clEnqueueWriteBuffer failed: status %i\n", status);
      }

    } // GridCopyin

    void CLinfo::GridCopyout(__PSGrid *g, void *dst_buf) {

      // Read out the data from device memory object g->buf to dst_buf.
      cl_mem dev_mem = (cl_mem) g->buf;
      size_t size_read = g->elm_size * g->num_elms;
      cl_int status = -1;
      err_status = 1;

      status = clEnqueueReadBuffer(clqueue, dev_mem, CL_TRUE,
        0, size_read, dst_buf,
        0, NULL, NULL);
      if (status == CL_SUCCESS) {
        err_status = 0;
      } else {
        fprintf(stderr, "Calling clEnqueueReadBuffer failed: status %i\n", status);
      }

    } // GridCopyout

    // For example, { float val; PSGridSet(grid_mat, xx, yy, zz, val); } is translated
    // into { float val; __PSGridSet(grid_mat, &val, xx, yy, zz); }

    // The variable arguments express the coordinate where the new value will be
    // set.
    void CLinfo::GridSet(__PSGrid *g, const void *ptr_val, va_list valst_dim) {

      cl_mem dev_mem = (cl_mem) g->buf;
      size_t offset = 0;
      size_t base_offset = 1;
      index_t idx = 0;
      int i = 0;
      int status = -1;
      err_status = 1;

      for (i = 0; i <  g->num_dims; i++) {
        idx = va_arg(valst_dim, index_t);
        offset += idx;
        base_offset *= g->dim[i];
      }
      offset *= g->elm_size;

      status = clEnqueueWriteBuffer(clqueue, dev_mem, CL_TRUE, 
        offset, g->elm_size, ptr_val,
        0, NULL, NULL);
      if (status == CL_SUCCESS) {
        err_status = 0;
      } else {
        fprintf(stderr, "Calling clEnqueueWriteBuffer failed: status %i\n", status);
      }

    } // GridSet

  } // namespace physis
} // namespace runtime
