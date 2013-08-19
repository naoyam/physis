// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/runtime_common.h"
#include "runtime/rpc.h"
#include "runtime/grid_util.h"
#include "runtime/grid_space_mpi.h"

namespace physis {
namespace runtime {

Client::Client(int rank, int num_procs, InterProcComm *ipc,
               __PSStencilRunClientFunction *stencil_runs,
               GridSpaceMPI *gs):
    Proc(rank, num_procs, ipc, stencil_runs),
    gs_(gs), done_(false) {
}

void Client::Listen() {
  while (!done_) {
    LOG_INFO() << "Client: listening\n";
    Request req;
    ipc_->Bcast(&req, sizeof(Request), GetMasterRank());
    switch (req.kind) {
      case FUNC_FINALIZE:
        LOG_INFO() << "Client: Finalize requested\n";
        Finalize();
        LOG_INFO() << "Client: Finalize done\n";
        break;
      case FUNC_BARRIER:
        LOG_INFO() << "Client: Barrier requested\n";
        Barrier();
        LOG_INFO() << "Client: Barrier done\n";
        break;
      case FUNC_NEW:
        LOG_INFO() << "Client: new requested\n";
        GridNew();
        LOG_INFO() << "Client: new done\n";        
        break;
      case FUNC_DELETE:
        LOG_INFO() << "Client: free requested\n";
        GridDelete(req.opt);
        LOG_INFO() << "Client: free done\n";        
        break;
      case FUNC_COPYIN:
        LOG_INFO() << "Client: copyin requested\n";
        GridCopyin(req.opt);
        LOG_INFO() << "Client: copyin done\n";        
        break;
      case FUNC_COPYOUT:
        LOG_INFO() << "Client: copyout requested\n";
        GridCopyout(req.opt);
        LOG_INFO() << "Client: copyout done\n";        
        break;
      case FUNC_GET:
        LOG_INFO() << "Client: get requested\n";
        GridGet(req.opt);
        LOG_INFO() << "Client: get done\n";        
        break;
      case FUNC_SET:
        LOG_INFO() << "Client: set requested\n";
        GridSet(req.opt);
        LOG_INFO() << "Client: set done\n";        
        break;
      case FUNC_RUN:
        LOG_DEBUG() << "Client: run requested ("
                        << req.opt << ")\n";
        StencilRun(req.opt);
        LOG_DEBUG() << "Client: run done\n";
        break;
      case FUNC_GRID_REDUCE:
        LOG_DEBUG() << "Client: grid reduce requested ("
                    << req.opt << ")\n";
        GridReduce(req.opt);
        LOG_DEBUG() << "Client: grid reduce done\n";
        break;
      case FUNC_INVALID:
        LOG_INFO() << "Client: invaid request\n";
        PSAbort(1);
      default:
        LOG_ERROR() << "Unsupported request: " << req.kind << "\n";
        PSAbort(1);            
    }
  }
  LOG_INFO() << "Client listening terminated.\n";
  MPI_Finalize();
  exit(EXIT_SUCCESS);
  return;
}

Master::Master(int rank, int num_procs, InterProcComm *ipc,
               __PSStencilRunClientFunction *stencil_runs,
               GridSpaceMPI *gs):
    Proc(rank, num_procs, ipc, stencil_runs), gs_(gs) {
  assert(rank == 0);
}

void Master::NotifyCall(enum RT_FUNC_KIND fkind, int opt) {
  Request r(fkind, opt);
  ipc_->Bcast(&r, sizeof(Request), rank());
}

// Finalize
void Master::Finalize() {
  LOG_DEBUG() << "[" << rank() << "] Finalize\n";
  NotifyCall(FUNC_FINALIZE);
  MPI_Finalize();
}

void Client::Finalize() {
  LOG_DEBUG() << "[" << rank() << "] Finalize\n";
  done_ = true;
}

// Barrier
void Master::Barrier() {
  LOG_DEBUG() << "[" << rank() << "] Barrier\n";
  NotifyCall(FUNC_BARRIER);
  ipc_->Barrier();
}

void Client::Barrier() {
  LOG_DEBUG() << "[" << rank() << "] Barrier\n";
  ipc_->Barrier();
}

// Create
GridMPI *Master::GridNew(PSType type, int elm_size,
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
  GridMPI *g = gs_->CreateGrid
      (type, elm_size, num_dims, size,
       global_offset, stencil_offset_min, stencil_offset_max, attr);
  return g;
}

void Client::GridNew() {
  LOG_DEBUG() << "[" << rank() << "] Create\n";
  RequestNEW req;
  ipc_->Bcast(&req, sizeof(RequestNEW), GetMasterRank());
  static_cast<GridSpaceMPI*>(gs_)->CreateGrid(
      req.type, req.elm_size, req.num_dims, req.size,
      req.global_offset,
      req.stencil_offset_min, req.stencil_offset_max,
      req.attr);
  LOG_DEBUG() << "[" << rank() << "] Create done\n";
  return;
}

// Delete
void Master::GridDelete(GridMPI *g) {
  LOG_DEBUG() << "[" << rank() << "] Delete\n";
  NotifyCall(FUNC_DELETE, g->id());
  gs_->DeleteGrid(g);
  return;
}

void Client::GridDelete(int id) {
  LOG_DEBUG() << "[" << rank() << "] Delete\n";
  gs_->DeleteGrid(id);
  return;
}

void Master::GridCopyinLocal(GridMPI *g, const void *buf) {
  if (g->empty()) return;

  size_t s = ((GridMPI*)g)->local_real_size().accumulate(g->num_dims()) *
      g->elm_size();
  PSAssert(g->buffer()->size() == s);

  void *tmp_buf = NULL;
  void *grid_dst = g->buffer()->Get();
  
  if (g->HasHalo()) {
    tmp_buf = malloc(g->GetLocalBufferSize());
    assert(tmp_buf);
    grid_dst = tmp_buf;
  }
  
  CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                 g->size(), grid_dst,
                 g->local_offset(), g->local_size());
  
  if (g->HasHalo()) {
    g->Copyin(grid_dst);
    free(tmp_buf);
  }
}

// Copyin
void Master::GridCopyin(GridMPI *g, const void *buf) {
  LOG_DEBUG() << "[" << rank() << "] Copyin\n";

  // copyin to own buffer
  GridCopyinLocal(g, buf);

  // Copyin to remote subgrids
  NotifyCall(FUNC_COPYIN, g->id());  
  BufferHost send_buf;
  // Assumes the master rank is 0
  for (int i = 1; i < gs_->num_procs(); ++i) {
    IndexArray subgrid_size, subgrid_offset;
    ipc_->Recv(&subgrid_offset, sizeof(IndexArray), i);
    LOG_VERBOSE() << "sg offset: " << subgrid_offset << "\n";
    ipc_->Recv(&subgrid_size, sizeof(IndexArray), i);
    LOG_VERBOSE() << "sg size: " << subgrid_size << "\n";
    size_t gsize = subgrid_size.accumulate(g->num_dims()) *
        g->elm_size();
    if (gsize == 0) continue;
    send_buf.EnsureCapacity(
        subgrid_size.accumulate(g->num_dims()) * g->elm_size());
    CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                   g->size(), send_buf.Get(),
                   subgrid_offset, subgrid_size);
    ipc_->Send(send_buf.Get(), gsize, i);
  }
  return;
}

void Client::GridCopyin(int id) {
  LOG_DEBUG() << "Copyin\n";

  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(id));
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

void Master::GridCopyoutLocal(GridMPI *g, void *buf) {
  if (g->empty()) return;

  const void *grid_src = g->buffer()->Get();
  void *tmp_buf = NULL;
  
  if (g->HasHalo()) {
    tmp_buf = malloc(g->GetLocalBufferSize());
    assert(tmp_buf);
    g->Copyout(tmp_buf);
    grid_src = tmp_buf;
  }
  
  CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                g->size(), grid_src, g->local_offset(),
                g->local_size());
  
  if (g->HasHalo()) free(tmp_buf);
  
  return;
}

// Copyout
void Master::GridCopyout(GridMPI *g, void *buf) {
  LOG_DEBUG() << "Copyout\n";

  // Copyout self
  GridCopyoutLocal(g, buf);
  
  // Copyout from remote grids
  NotifyCall(FUNC_COPYOUT, g->id());
  BufferHost recv_buf;
  // Assumes the master rank is 0
  for (int i = 1; i < gs_->num_procs(); ++i) {
    IndexArray subgrid_size, subgrid_offset;
    ipc_->Recv(&subgrid_offset, sizeof(IndexArray), i);
    LOG_VERBOSE() << "sg offset: " << subgrid_offset << "\n";
    ipc_->Recv(&subgrid_size, sizeof(IndexArray), i);
    LOG_VERBOSE() << "sg size: " << subgrid_size << "\n";
    //size_t gsize = sbuf.GetLinearSize(subgrid_size);
    size_t gsize = subgrid_size.accumulate(g->num_dims()) *
        g->elm_size();
    if (gsize == 0) continue;
    recv_buf.EnsureCapacity(gsize);
    ipc_->Recv(recv_buf.Get(), gsize, i);
    CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                  g->size(), recv_buf.Get(), subgrid_offset,
                  subgrid_size);
  }
}

void Client::GridCopyout(int id) {
  LOG_DEBUG() << "Copyout\n";
  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(id));
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

void Master::StencilRun(int id, int iter, int num_stencils,
                        void **stencils,
                        unsigned *stencil_sizes) {
  LOG_DEBUG() << "Master StencilRun(" << id << ")\n";
  
  NotifyCall(FUNC_RUN, id);
  // Send the number of iterations
  ipc_->Bcast(&iter, sizeof(int), rank());
  ipc_->Bcast(&num_stencils, sizeof(int), rank());
  ipc_->Bcast(stencil_sizes, num_stencils * sizeof(unsigned),
             rank());
  for (int i = 0; i < num_stencils; ++i) {
    ipc_->Bcast(stencils[i], stencil_sizes[i], rank());
  }
  LOG_DEBUG() << "Calling the stencil function\n";
  // call the stencil obj
  stencil_runs_[id](iter, stencils);
  return;
}

void Client::StencilRun(int id) {
  LOG_DEBUG() << "Client StencilRun(" << id << ")\n";

  // Send the number of iterations
  int iter;  
  ipc_->Bcast(&iter, sizeof(int), GetMasterRank());
  int num_stencils;
  ipc_->Bcast(&num_stencils, sizeof(int), GetMasterRank());
  unsigned *stencil_sizes = new unsigned[num_stencils];
  // Send the stencil object size
  ipc_->Bcast(stencil_sizes, num_stencils * sizeof(unsigned),
             GetMasterRank());
  void **stencils = new void*[num_stencils];
  for (int i = 0; i < num_stencils; ++i) {
    void *sbuf = malloc(stencil_sizes[i]);
    ipc_->Bcast(sbuf, stencil_sizes[i], GetMasterRank());
    stencils[i] = sbuf;
  }
  LOG_DEBUG() << "Calling the stencil function\n";
  stencil_runs_[id](iter, stencils);
  delete[] stencil_sizes;
  delete[] stencils;
  return;
}

void Client::GridGet(int id) {
  LOG_DEBUG() << "Client GridGet(" << id << ")\n";
  if (id != rank()) {
    // this is not a request to me
    LOG_DEBUG() << "Client GridSet done\n";
    return;
  }

  int gid;
  ipc_->Recv(&gid, sizeof(int), GetMasterRank());
  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(gid));
  IndexArray index;
  ipc_->Recv(&index, sizeof(IndexArray), GetMasterRank());
  LOG_DEBUG() << "Get index: " << index << "\n";
  void *buf = malloc(g->elm_size());
  g->Get(index, buf);
  ipc_->Send(buf, g->elm_size(), GetMasterRank());
  LOG_DEBUG() << "Client GridGet done\n";
  free(buf);
  return;
}

void Master::GridGet(GridMPI *g, void *buf, const IndexArray &index) {
  LOG_DEBUG() << "Master GridGet\n";

  int peer_rank = gs_->FindOwnerProcess(g, index);
  LOG_DEBUG() << "Owner: " << peer_rank << "\n";

  if (peer_rank != rank()) {
    // NOTE: We don't need to notify all processes but clients are
    // waiting on MPI_Bcast, so P2P methods are not allowed.
    NotifyCall(FUNC_GET, peer_rank);
    int gid = g->id();
    ipc_->Send(&gid, sizeof(int), peer_rank);
    // MPI_Send does not accept const buffer pointer    
    IndexArray t = index;    
    ipc_->Send(&t, sizeof(IndexArray), peer_rank);
    ipc_->Recv((void*)buf, g->elm_size(), peer_rank);
  } else {
    g->Get(index, buf);
  }
  LOG_DEBUG() << "Master GridGet done\n";
  return;
}

void Client::GridSet(int id) {
  LOG_DEBUG() << "Client GridSet(" << id << ")\n";
  if (id != rank()) {
    // this is not a request to me
    LOG_DEBUG() << "Client GridSet done\n";
    return;
  }

  int gid;
  ipc_->Recv(&gid, sizeof(int), GetMasterRank());
  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(gid));
  IndexArray index;
  ipc_->Recv(&index, sizeof(IndexArray), GetMasterRank());
  LOG_DEBUG() << "Set index: " << index << "\n";
  void *buf = malloc(g->elm_size());
  ipc_->Recv(buf, g->elm_size(), GetMasterRank());
  //memcpy(g->GetAddress(index), buf, g->elm_size());
  g->Set(index, buf);
  LOG_DEBUG() << "Client GridSet done\n";
  free(buf);
  return;
}

void Master::GridSet(GridMPI *g, const void *buf, const IndexArray &index) {
  LOG_DEBUG() << "Master GridSet\n";

  int peer_rank = gs_->FindOwnerProcess(g, index);
  LOG_DEBUG() << "Owner: " << peer_rank << "\n";

  if (peer_rank != rank()) {
    // NOTE: We don't need to notify all processes but clients are
    // waiting on MPI_Bcast, so P2P methods are not allowed.
    NotifyCall(FUNC_SET, peer_rank);
    int gid = g->id();
    ipc_->Send(&gid, sizeof(int), peer_rank);
    // MPI_Send does not accept const buffer pointer    
    IndexArray t = index;    
    ipc_->Send(&t, sizeof(IndexArray), peer_rank);
    ipc_->Send((void*)buf, g->elm_size(), peer_rank);
  } else {
    //memcpy(g->GetAddress(index), buf, g->elm_size());
    g->Set(index, buf);
  }
}

void Client::GridReduce(int id) {
  LOG_DEBUG() << "Client GridReduce(" << id << ")\n";
  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(id));
  PSReduceOp op;
  ipc_->Bcast(&op, sizeof(PSReduceOp), GetMasterRank());
  gs_->ReduceGrid(NULL, op, g);
  return;
}

void Master::GridReduce(void *buf, PSReduceOp op, GridMPI *g) {
  LOG_DEBUG() << "Master GridReduce\n";
  NotifyCall(FUNC_GRID_REDUCE, g->id());
  ipc_->Bcast(&op, sizeof(PSReduceOp), rank());
  gs_->ReduceGrid(buf, op, g);
  LOG_DEBUG() << "Master GridReduce done\n";
}

} // namespace runtime
} // namespace physis
