// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RPC_MPI_OPENCL_H_
#define PHYSIS_RUNTIME_RPC_MPI_OPENCL_H_

#include "runtime/runtime_common.h"
#include "runtime/rpc_mpi.h"
#include "runtime/grid_mpi_opencl.h"
#include "runtime/rpc_opencl_common.h"
#include "runtime/buffer_opencl.h"

#include <vector>

namespace physis {
namespace runtime {

class MasterMPIOpenCL: public Master {
 public:
  MasterMPIOpenCL(
      const ProcInfo &pinfo, GridSpaceMPIOpenCL *gs,
      MPI_Comm comm,
      CLbaseinfo *cl_in);
  virtual ~MasterMPIOpenCL();
  virtual void Finalize();
  virtual void GridCopyinLocal(GridMPI *g, const void *buf);
  virtual void GridCopyoutLocal(GridMPI *g, void *buf);
 protected:
  BufferOpenCLHost *pinned_buf_;
  CLbaseinfo *cl_generic_;
};

class ClientMPIOpenCL: public Client {
 public:
  ClientMPIOpenCL(
      const ProcInfo &pinfo, GridSpaceMPIOpenCL *gs,
      MPI_Comm comm,
      CLbaseinfo *cl_in);
  virtual ~ClientMPIOpenCL();
  virtual void Finalize();
 protected:
  CLbaseinfo *cl_generic_;
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RPC_MPI_OPENCL_H_ */
