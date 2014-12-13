// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RPC_MPI_CUDA_H_
#define PHYSIS_RUNTIME_RPC_MPI_CUDA_H_

#include "runtime/runtime_common.h"
#include "runtime/rpc.h"
#include "runtime/grid_mpi_cuda.h"
#include "runtime/buffer_cuda.h"

namespace physis {
namespace runtime {

class MasterMPICUDA: public Master {
 public:
  MasterMPICUDA(const ProcInfo &pinfo, GridSpaceMPICUDA *gs,
                MPI_Comm comm);
  virtual ~MasterMPICUDA();
  virtual void Finalize();
  virtual void GridCopyinLocal(GridMPI *g, const void *buf);
  virtual void GridCopyoutLocal(GridMPI *g, void *buf);  
 protected:
  BufferCUDAHost *pinned_buf_;
};

class ClientMPICUDA: public Client {
 public:
  ClientMPICUDA(const ProcInfo &pinfo, GridSpaceMPICUDA *gs,
                MPI_Comm comm);
  virtual ~ClientMPICUDA();
  virtual void Finalize();  
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RPC_MPI_CUDA_H_ */
