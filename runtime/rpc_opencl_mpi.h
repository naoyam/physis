#ifndef PHYSIS_RUNTIME_RPC_OPENCL_MPI_H
#define PHYSIS_RUNTIME_RPC_OPENCL_MPI_H

#include "runtime/rpc_opencl_common.h"

namespace physis {
  namespace runtime {

    class CLMPIinfo : public CLbaseinfo {

      protected:

      public:
        CLMPIinfo(unsigned int id_default = 0);
        virtual ~CLMPIinfo() {} ;


    }; // class CLMPIinfo
  } // namespace runtime
} // namespace physis

#endif /* #define PHYSIS_RUNTIME_RPC_OPENCL_MPI_H */
