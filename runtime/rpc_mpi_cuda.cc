// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/rpc_mpi_cuda.h"
#include "runtime/mpi_wrapper.h"
#include "runtime/grid_util.h"
#include "runtime/runtime_common_cuda.h"

#include <cuda_runtime.h>


namespace physis {
namespace runtime {
MasterMPICUDA::MasterMPICUDA(const ProcInfo &pinfo,
                             GridSpaceMPICUDA *gs, MPI_Comm comm):
    Master(pinfo, gs, comm) {
  pinned_buf_ = new BufferCUDAHost(0, 1);
}

MasterMPICUDA::~MasterMPICUDA() {
  delete pinned_buf_;

}

ClientMPICUDA::ClientMPICUDA(const ProcInfo &pinfo,
                             GridSpaceMPICUDA *gs, MPI_Comm comm):
    Client(pinfo, gs, comm) {
}

ClientMPICUDA::~ClientMPICUDA() {}

// Finalize
void MasterMPICUDA::Finalize() {
  CUDA_SAFE_CALL(cudaDeviceReset());
  static_cast<GridSpaceMPICUDA*>(gs_)->PrintLoadNeighborProf(std::cerr);  
  Master::Finalize();
}

void ClientMPICUDA::Finalize() {
  CUDA_SAFE_CALL(cudaDeviceReset());
  static_cast<GridSpaceMPICUDA*>(gs_)->PrintLoadNeighborProf(std::cerr);   
  Client::Finalize();
}

void MasterMPICUDA::GridCopyinLocal(GridMPI *g, const void *buf) {
  // extract the subgrid from buf for this process to the pinned
  // buffer
  size_t size = g->local_size().accumulate(g->num_dims()) *
      g->elm_size();
  pinned_buf_->EnsureCapacity(size);
  PSAssert(pinned_buf_->Get());
  CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                 g->size(), pinned_buf_->Get(),
                 g->local_offset(), g->local_size());
  // send it out to device memory
  static_cast<BufferCUDADev*>(g->buffer())->Copyin(
      *pinned_buf_, IndexArray(), g->local_size());
}

void MasterMPICUDA::GridCopyoutLocal(GridMPI *g, void *buf) {
  BufferCUDADev *dev_buf = static_cast<BufferCUDADev*>(g->buffer());
  size_t size = g->local_size().accumulate(g->num_dims()) *
                g->elm_size();
  PSAssert(dev_buf->GetLinearSize() == size);
  dev_buf->Copyout(*pinned_buf_, IndexArray(), g->local_size());
  CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                g->size(), pinned_buf_->Get(), g->local_offset(),
                g->local_size());
}

} // namespace runtime
} // namespace physis
