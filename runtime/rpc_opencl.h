#ifndef PHYSIS_RUNTIME_RPC_OPENCL_H_
#define PHYSIS_RUNTIME_RPC_OPENCL_H_

/* physis-OpenCL specific*/
#include "physis/physis_opencl.h"
#include "runtime/rpc_opencl_common.h"

/*
  Note:
  CLinfo: OpenCL information class
  Members or functions not using __PSGrid belongs to CLbaseinfo.
  Members or functions using __PSGrid should belong to
  CLbase, which should inherit CLbaseinfo.
*/

namespace physis {
  namespace runtime {

    class CLinfo : public CLbaseinfo {
      protected:

      public:
        CLinfo(): CLbaseinfo() {};
        virtual ~CLinfo() {};

        virtual __PSGrid* GridNew(int elm_size, int num_dims, PSVectorInt dim, int double_buffering);
        virtual void GridFree(__PSGrid *g);
        virtual void GridCopyin(__PSGrid *g, const void *src_buf);
        virtual void GridCopyout(__PSGrid *g, void *dst_buf);
        virtual void GridSet(__PSGrid *g, const void *val_ptr, va_list valst_dim);

    }; // class CLinfo

  } // namespace runtime
} // namespace physis

#endif /* #define PHYSIS_RUNTIME_RPC_OPENCL_H_ */
