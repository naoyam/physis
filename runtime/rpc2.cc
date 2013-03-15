// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/rpc2.h"
#include "runtime/grid_mpi2.h"
#include "runtime/grid_util.h"
#include "runtime/mpi_wrapper.h"

namespace physis {
namespace runtime {

// Create
GridMPI *Master2::GridNew(PSType type, int elm_size,
                          int num_dims, const IndexArray &size,
                          const IndexArray &global_offset,
                          const IndexArray &stencil_offset_min,
                          const IndexArray &stencil_offset_max,                          
                          int attr) {
  LOG_DEBUG() << "[" << rank() << "] New\n";
  NotifyCall(FUNC_NEW);
  RequestNEW req = {type, elm_size, num_dims, size,
                    false, global_offset,
                    stencil_offset_min, stencil_offset_max,
                    attr};
  ipc_->Bcast(&req, sizeof(RequestNEW), rank());  
  GridMPI *g = ((GridSpaceMPI2*)gs_)->CreateGrid
      (type, elm_size, num_dims, size,
       global_offset, stencil_offset_min, stencil_offset_max, attr);
  return g;
}

void Client2::GridNew() {
  LOG_DEBUG() << "[" << rank() << "] Create\n";
  RequestNEW req;
  ipc_->Bcast(&req, sizeof(RequestNEW), GetMasterRank());
  static_cast<GridSpaceMPI2*>(gs_)->CreateGrid(
      req.type, req.elm_size, req.num_dims, req.size,
      req.global_offset,
      req.stencil_offset_min, req.stencil_offset_max,
      req.attr);
  LOG_DEBUG() << "[" << rank() << "] Create done\n";
  return;
}


void Master2::GridCopyoutLocal(GridMPI *g, void *buf) {
  if (g->empty()) return;

  GridMPI2 *g2 = (GridMPI2*)g;
  const void *grid_src = g2->buffer()->Get();
  void *tmp_buf = NULL;
  
  if (g2->HasHalo()) {
    tmp_buf = malloc(g->GetLocalBufferSize());
    assert(tmp_buf);
    g2->Copyout(tmp_buf);
    grid_src = tmp_buf;
  }
  
  CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                g->size(), grid_src, g->local_offset(),
                g->local_size());
  
  if (g2->HasHalo()) free(tmp_buf);
  
  return;
}

void Client2::GridCopyout(int id) {
  LOG_DEBUG() << "Copyout\n";
  GridMPI2 *g = static_cast<GridMPI2*>(gs_->FindGrid(id));
  // notify the local offset
  IndexArray ia = g->local_offset();
  ipc_->Send(&ia, sizeof(IndexArray), GetMasterRank());  
  // notify the local size
  ia = g->local_size();
  ipc_->Send(&ia, sizeof(IndexArray), GetMasterRank());  
  if (g->empty()) {
    LOG_DEBUG() << "No copy needed because this grid is empty.\n";
    return;
  }
  Buffer *sbuf = g->buffer();
  if (g->HasHalo()) {
    sbuf = new BufferHost();
    sbuf->EnsureCapacity(g->GetLocalBufferSize());
    g->Copyout(sbuf->Get());
  } 
  ipc_->Send(sbuf->Get(),  g->GetLocalBufferSize(),
             GetMasterRank());
  if (g->HasHalo()) {
    free(sbuf);
  }
  return;
}


void Master2::GridCopyinLocal(GridMPI *g, const void *buf) {
  if (g->empty()) return;

  size_t s = ((GridMPI2*)g)->local_real_size().accumulate(g->num_dims()) *
      g->elm_size();
  PSAssert(g->buffer()->size() == s);

  GridMPI2 *g2 = (GridMPI2*)g;
  void *tmp_buf = NULL;
  void *grid_dst = g->buffer()->Get();
  
  if (g2->HasHalo()) {
    tmp_buf = malloc(g->GetLocalBufferSize());
    assert(tmp_buf);
    grid_dst = tmp_buf;
  }
  
  CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                 g->size(), grid_dst,
                 g->local_offset(), g->local_size());
  
  if (g2->HasHalo()) {
    g2->Copyin(grid_dst);
    free(tmp_buf);
  }
}


void Client2::GridCopyin(int id) {
  LOG_DEBUG() << "Copyin\n";

  GridMPI2 *g = static_cast<GridMPI2*>(gs_->FindGrid(id));
  // notify the local offset
  IndexArray ia = g->local_offset();
  ipc_->Send(&ia, sizeof(IndexArray), GetMasterRank());  
  // notify the local size
  ia = g->local_size();
  ipc_->Send(&ia, sizeof(IndexArray), GetMasterRank());  
  if (g->empty()) {
    LOG_DEBUG() << "No copy needed because this grid is empty.\n";
    return;
  }
  // receive the subregion for this process
  Buffer *dst_buf = g->buffer();
  if (g->HasHalo()) {
    dst_buf = new BufferHost();
    dst_buf->EnsureCapacity(g->GetLocalBufferSize());
  }
  ipc_->Recv(dst_buf->Get(), g->GetLocalBufferSize(),
             GetMasterRank());  
  if (g->HasHalo()) {
    g->Copyin(dst_buf->Get());
    delete dst_buf;
  }
  return;
}


} // namespace runtime
} // namespace physis
