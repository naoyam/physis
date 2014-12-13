// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RPC_CUDA_H_
#define PHYSIS_RUNTIME_RPC_CUDA_H_

#include "runtime/runtime_common.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/rpc.h"
#include "runtime/grid_mpi_cuda_exp.h"
#include "runtime/buffer_cuda.h"

namespace physis {
namespace runtime {

template <class GridSpaceType>
class MasterCUDA: public Master<GridSpaceType> {
 public:
  MasterCUDA(InterProcComm *ipc,
             __PSStencilRunClientFunction *stencil_runs,
             GridSpaceType *gs);
  virtual ~MasterCUDA();
  virtual void Finalize();
  virtual void GridCopyinLocal(typename GridSpaceType::GridType *g, const void *buf);
  virtual void GridCopyoutLocal(typename GridSpaceType::GridType *g, void *buf);  
 protected:
  BufferCUDAHost *pinned_buf_;
};

template <class GridSpaceType>
class ClientCUDA: public Client<GridSpaceType> {
 public:
  ClientCUDA(InterProcComm *ipc,
             __PSStencilRunClientFunction *stencil_runs,
             GridSpaceType *gs);
  virtual void Finalize();  
};

template <class GridSpaceType>
MasterCUDA<GridSpaceType>::MasterCUDA(InterProcComm *ipc,
                                      __PSStencilRunClientFunction *stencil_runs,
                                      GridSpaceType *gs):
    Master<GridSpaceType>(ipc, stencil_runs, gs) {
  pinned_buf_ = new BufferCUDAHost();
}

template <class GridSpaceType>
MasterCUDA<GridSpaceType>::~MasterCUDA() {
  delete pinned_buf_;
}

// Finalize
template <class GridSpaceType>
void MasterCUDA<GridSpaceType>::Finalize() {
  CUDA_SAFE_CALL(cudaDeviceReset());
  this->gs_->PrintLoadNeighborProf(std::cerr);
  Master<GridSpaceType>::Finalize();
}

template <class GridSpaceType>
ClientCUDA<GridSpaceType>::ClientCUDA(
    InterProcComm *ipc,
    __PSStencilRunClientFunction *stencil_runs,
    GridSpaceType *gs): Client<GridSpaceType>(ipc, stencil_runs, gs) {
}


template <class GridSpaceType>
void ClientCUDA<GridSpaceType>::Finalize() {
  CUDA_SAFE_CALL(cudaDeviceReset());
  this->gs_->PrintLoadNeighborProf(std::cerr);   
  Client<GridSpaceType>::Finalize();
}

template <class GridSpaceType>
void MasterCUDA<GridSpaceType>::GridCopyinLocal(typename GridSpaceType::GridType *g, const void *buf) {
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
  g->Copyin(pinned_buf_->Get());
}

template <class GridSpaceType>
void MasterCUDA<GridSpaceType>::GridCopyoutLocal(typename GridSpaceType::GridType *g, void *buf) {
  size_t size = g->local_size().accumulate(g->num_dims()) *
      g->elm_size();
  pinned_buf_->EnsureCapacity(size);
  g->Copyout(pinned_buf_->Get());
  CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                g->size(), pinned_buf_->Get(), g->local_offset(),
                g->local_size());
}

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RPC_CUDA_H_ */
