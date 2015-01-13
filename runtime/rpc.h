// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RPC_H_
#define PHYSIS_RUNTIME_RPC_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_space_mpi.h"
#include "runtime/proc.h"
#include "runtime/grid_util.h"

namespace physis {
namespace runtime {

typedef std::vector<std::pair<int, void*> > StencilArgVector;

enum RT_FUNC_KIND {
  FUNC_INVALID, FUNC_NEW, FUNC_DELETE,
  FUNC_COPYIN, FUNC_COPYOUT,
  FUNC_GET, FUNC_SET,
  FUNC_RUN, FUNC_FINALIZE, FUNC_BARRIER,
  FUNC_GRID_REDUCE
};

struct Request {
  RT_FUNC_KIND kind;
  int opt;
  Request(RT_FUNC_KIND k=FUNC_INVALID, int opt=0)
      : kind(k), opt(opt) {}
};

struct RequestNEW {
  __PSGridTypeInfo type_info;
  int num_dims;
  IndexArray size;
  bool double_buffering;
  IndexArray global_offset;
  IndexArray stencil_offset_min;
  IndexArray stencil_offset_max;
  int attr;
};

template <class GridSpaceType>
class Client: public Proc {
 protected:
  GridSpaceType *gs_;
  bool done_;
 public:
  Client(InterProcComm *ipc,
         __PSStencilRunClientFunction *stencil_runs,
         GridSpaceType *gs);
  virtual ~Client() {}
  virtual void Listen();  
  virtual void Finalize();
  virtual void Barrier();
  virtual void GridNew(int num_members);
  virtual void GridDelete(int id);
  virtual void GridCopyin(int id);
  virtual void GridCopyinStage2(
      typename GridSpaceType::GridType *g);
  virtual void GridCopyout(int id);
  virtual void GridCopyoutStage2(
      typename GridSpaceType::GridType *g);
  virtual void GridSet(int id);
  virtual void GridGet(int id);  
  virtual void StencilRun(int id);
  virtual void GridReduce(int id);
  static int GetMasterRank() {
    return Proc::GetRootRank();
  }
};

template <class GridSpaceType>
class Master: public Proc {
 protected:
  Counter gridCounter;
  GridSpaceType *gs_;  
  void NotifyCall(enum RT_FUNC_KIND fkind, int opt=0);
 public:
  Master(InterProcComm *ipc,
         __PSStencilRunClientFunction *stencil_runs,
         GridSpaceType *gs);
  virtual ~Master() {}
  virtual void Finalize();
  virtual void Barrier();
  virtual typename GridSpaceType::GridType *GridNew(
      __PSGridTypeInfo *type_info,
      int num_dims, const IndexArray &size,
      const IndexArray &global_offset, const IndexArray &stencil_offset_min,
      const IndexArray &stencil_offset_max,
      const int *stencil_offset_min_member,
      const int *stencil_offset_max_member,
      int attr);
  virtual void GridDelete(typename GridSpaceType::GridType *g);
  virtual void GridCopyin(typename GridSpaceType::GridType *g, const void *buf);
  virtual void GridCopyinLocal(typename GridSpaceType::GridType *g, const void *buf);  
  virtual void GridCopyout(typename GridSpaceType::GridType *g, void *buf);
  virtual void GridCopyoutLocal(typename GridSpaceType::GridType *g, void *buf);  
  virtual void GridSet(typename GridSpaceType::GridType *g, const void *buf, const IndexArray &index);
  virtual void GridGet(typename GridSpaceType::GridType *g, void *buf, const IndexArray &index);  
  virtual void StencilRun(int id, int iter, int num_stencils,
                          void **stencils, unsigned *stencil_sizes);
  virtual void GridReduce(void *buf, PSReduceOp op, typename GridSpaceType::GridType *g);
  static int GetMasterRank() {
    return Proc::GetRootRank();
  }
  
};

template <class GridSpaceType>
Client<GridSpaceType>::Client(InterProcComm *ipc,
                              __PSStencilRunClientFunction *stencil_runs,
                              GridSpaceType *gs):
    Proc(ipc, stencil_runs), gs_(gs), done_(false) {
}

template <class GridSpaceType>
void Client<GridSpaceType>::Listen() {
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
        GridNew(req.opt);
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

template <class GridSpaceType>
Master<GridSpaceType>::Master(InterProcComm *ipc,
                              __PSStencilRunClientFunction *stencil_runs,
                              GridSpaceType *gs):
    Proc(ipc, stencil_runs), gs_(gs) {
  assert(rank_ == 0);
}

template <class GridSpaceType>
void Master<GridSpaceType>::NotifyCall(enum RT_FUNC_KIND fkind, int opt) {
  Request r(fkind, opt);
  ipc_->Bcast(&r, sizeof(Request), rank());
}

// Finalize
template <class GridSpaceType>
void Master<GridSpaceType>::Finalize() {
  LOG_DEBUG() << "[" << rank() << "] Finalize\n";
  NotifyCall(FUNC_FINALIZE);
  MPI_Finalize();
}

template <class GridSpaceType>
void Client<GridSpaceType>::Finalize() {
  LOG_DEBUG() << "[" << rank() << "] Finalize\n";
  done_ = true;
}

// Barrier
template <class GridSpaceType>
void Master<GridSpaceType>::Barrier() {
  LOG_DEBUG() << "[" << rank() << "] Barrier\n";
  NotifyCall(FUNC_BARRIER);
  ipc_->Barrier();
}

template <class GridSpaceType>
void Client<GridSpaceType>::Barrier() {
  LOG_DEBUG() << "[" << rank() << "] Barrier\n";
  ipc_->Barrier();
}

// Create
template <class GridSpaceType>
typename GridSpaceType::GridType *Master<GridSpaceType>::GridNew(
    __PSGridTypeInfo *type_info,
    int num_dims, const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &stencil_offset_min,
    const IndexArray &stencil_offset_max,
    const int *stencil_offset_min_member,
    const int *stencil_offset_max_member,
    int attr) {
  LOG_DEBUG() << "[" << rank() << "] New\n";
  NotifyCall(FUNC_NEW, type_info->num_members);
  RequestNEW req = {*type_info, num_dims, size,
                    false, global_offset,
                    stencil_offset_min, stencil_offset_max,
                    attr};
  ipc_->Bcast(&req, sizeof(RequestNEW), rank());
  if (type_info->num_members > 0) {
    PSAssert(stencil_offset_min_member);
    PSAssert(stencil_offset_max_member);
    size_t type_info_size = sizeof(__PSGridTypeMemberInfo) *
        type_info->num_members;
    size_t stencil_offset_size =
        sizeof(PSVectorInt) * type_info->num_members;
    // pack all member-related data to buf
    void *buf = malloc(type_info_size + stencil_offset_size * 2);
    memcpy(buf, type_info->members, type_info_size);
    memcpy((void*)((intptr_t)buf + type_info_size),
           stencil_offset_min_member, stencil_offset_size);
    memcpy((void*)((intptr_t)buf + type_info_size + stencil_offset_size),
           stencil_offset_max_member, stencil_offset_size);
    ipc_->Bcast(buf, type_info_size + stencil_offset_size * 2,
                rank());
    free(buf);
  }
  typename GridSpaceType::GridType *g = gs_->CreateGrid(
      type_info, num_dims, size, global_offset,
      stencil_offset_min, stencil_offset_max,
      stencil_offset_min_member, stencil_offset_max_member, 
      attr);
  return g;
}

template <class GridSpaceType>
void Client<GridSpaceType>::GridNew(int num_members) {
  LOG_DEBUG() << "[" << rank() << "] Create\n";
  RequestNEW req;
  ipc_->Bcast(&req, sizeof(RequestNEW), GetMasterRank());
  int *stencil_offset_min_member = NULL;
  int *stencil_offset_max_member = NULL;
  req.type_info.members = NULL;
  void *buf = NULL;
  if (num_members > 0) {
    size_t type_info_size = sizeof(__PSGridTypeMemberInfo) * num_members;
    size_t stencil_offset_size = sizeof(PSVectorInt) * num_members;
    buf = malloc(type_info_size + stencil_offset_size * 2);
    ipc_->Bcast(buf, type_info_size + stencil_offset_size * 2,
                GetMasterRank());
    req.type_info.members = (__PSGridTypeMemberInfo *)buf;
    stencil_offset_min_member =
        (int*)((intptr_t)buf + type_info_size);
    stencil_offset_max_member =
        (int*)((intptr_t)buf + type_info_size + stencil_offset_size);
  } else {
    req.type_info.num_members = 0;
  }
  this->gs_->CreateGrid(
      &req.type_info, req.num_dims, req.size,
      req.global_offset,
      req.stencil_offset_min, req.stencil_offset_max,
      stencil_offset_min_member, stencil_offset_max_member,
      req.attr);
  if (buf) free(buf);
  LOG_DEBUG() << "[" << rank() << "] Create done\n";
  return;
}

// Delete
template <class GridSpaceType>
void Master<GridSpaceType>::GridDelete(typename GridSpaceType::GridType *g) {
  LOG_DEBUG() << "[" << rank() << "] Delete\n";
  NotifyCall(FUNC_DELETE, g->id());
  this->gs_->DeleteGrid(g);
  return;
}

template <class GridSpaceType>
void Client<GridSpaceType>::GridDelete(int id) {
  LOG_DEBUG() << "[" << rank() << "] Delete\n";
  this->gs_->DeleteGrid(id);
  return;
}

template <class GridSpaceType>
void Master<GridSpaceType>::GridCopyinLocal(typename GridSpaceType::GridType *g, const void *buf) {
  if (g->empty()) return;

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
template <class GridSpaceType>
void Master<GridSpaceType>::GridCopyin(typename GridSpaceType::GridType *g, const void *buf) {
  LOG_DEBUG() << "[" << rank() << "] Copyin\n";
  fprintf(stderr, "PID: %d\n", getpid());
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
        g->elm_total_size();
    if (gsize == 0) continue;
    send_buf.EnsureCapacity(gsize);
    const void *buf_p = buf;
    void *send_buf_p = send_buf.Get();
    for (int j = 0; j < g->num_members(); ++j) {
      LOG_DEBUG() << "Copy member " << j << " for process "  << i << "\n";
      CopyoutSubgrid(g->elm_size(j), g->num_dims(), buf,
                     g->size(), send_buf_p,
                     subgrid_offset, subgrid_size);
      buf_p = (void *)((intptr_t)buf_p + g->num_elms() * g->elm_size(i));
      send_buf_p = (void *)((intptr_t)send_buf_p +
                            subgrid_size.accumulate(g->num_dims()) * g->elm_size(i));      
    }
    ipc_->Send(send_buf.Get(), gsize, i);
  }
  return;
}

template <class GridSpaceType>
void Client<GridSpaceType>::GridCopyinStage2(typename GridSpaceType::GridType *g) {
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
}

template <class GridSpaceType>
void Client<GridSpaceType>::GridCopyin(int id) {
  LOG_DEBUG() << "Copyin\n";

  typename GridSpaceType::GridType *g = static_cast<typename GridSpaceType::GridType*>(gs_->FindGrid(id));
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

  GridCopyinStage2(g);
  return;
}

template <class GridSpaceType>
void Master<GridSpaceType>::GridCopyoutLocal(typename GridSpaceType::GridType *g, void *buf) {
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
template <class GridSpaceType>
void Master<GridSpaceType>::GridCopyout(typename GridSpaceType::GridType *g, void *buf) {
  LOG_DEBUG() << "[" << rank() << "] Copyout\n";

  // Copyout self
  GridCopyoutLocal(g, buf);
  
  // Copyout from remote grids
  NotifyCall(FUNC_COPYOUT, g->id());
  BufferHost recv_buf;
  // Assumes the master rank is 0
  for (int i = 1; i < gs_->num_procs(); ++i) {
    IndexArray subgrid_size, subgrid_offset;
    ipc_->Recv(&subgrid_offset, sizeof(IndexArray), i);
    LOG_DEBUG() << "sg offset: " << subgrid_offset << "\n";
    ipc_->Recv(&subgrid_size, sizeof(IndexArray), i);
    LOG_DEBUG() << "sg size: " << subgrid_size << "\n";
    //size_t gsize = sbuf.GetLinearSize(subgrid_size);
    size_t gsize = subgrid_size.accumulate(g->num_dims()) *
        g->elm_total_size();
    if (gsize == 0) continue;
    LOG_DEBUG() << "Elm TOTAL SIZE: " << g->elm_total_size()<< "\n";
    recv_buf.EnsureCapacity(gsize);
    ipc_->Recv(recv_buf.Get(), gsize, i);
    LOG_DEBUG() << "Copyout subgrid received from " << i << "\n";
    void *recv_buf_p = recv_buf.Get();
    void *buf_p = buf;
    for (int j = 0; j < g->num_members(); ++j) {
      CopyinSubgrid(g->elm_size(j), g->num_dims(), buf,
                    g->size(), recv_buf_p, subgrid_offset,
                    subgrid_size);
      buf_p = (void *)((intptr_t)buf_p + g->num_elms() * g->elm_size(i));
      recv_buf_p = (void *)((intptr_t)recv_buf_p +
                            subgrid_size.accumulate(g->num_dims()) * g->elm_size(i));
    }
  }
}

template <class GridSpaceType>
void Client<GridSpaceType>::GridCopyoutStage2(
    typename GridSpaceType::GridType *g) {
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
}

template <class GridSpaceType>
void Client<GridSpaceType>::GridCopyout(int id) {
  LOG_DEBUG() << "[" << rank() << "] Copyout\n";
  typename GridSpaceType::GridType *g = static_cast<typename GridSpaceType::GridType*>(gs_->FindGrid(id));
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
  GridCopyoutStage2(g);
  return;
}

template <class GridSpaceType>
void Master<GridSpaceType>::StencilRun(int id, int iter, int num_stencils,
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

template <class GridSpaceType>
void Client<GridSpaceType>::StencilRun(int id) {
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

template <class GridSpaceType>
void Client<GridSpaceType>::GridGet(int id) {
  LOG_DEBUG() << "Client GridGet(" << id << ")\n";
  if (id != rank()) {
    // this is not a request to me
    LOG_DEBUG() << "Client GridSet done\n";
    return;
  }

  int gid;
  ipc_->Recv(&gid, sizeof(int), GetMasterRank());
  typename GridSpaceType::GridType *g = static_cast<typename GridSpaceType::GridType*>(gs_->FindGrid(gid));
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

template <class GridSpaceType>
void Master<GridSpaceType>::GridGet(typename GridSpaceType::GridType *g, void *buf, const IndexArray &index) {
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

template <class GridSpaceType>
void Client<GridSpaceType>::GridSet(int id) {
  LOG_DEBUG() << "Client GridSet(" << id << ")\n";
  if (id != rank()) {
    // this is not a request to me
    LOG_DEBUG() << "Client GridSet done\n";
    return;
  }

  int gid;
  ipc_->Recv(&gid, sizeof(int), GetMasterRank());
  typename GridSpaceType::GridType *g = static_cast<typename GridSpaceType::GridType*>(gs_->FindGrid(gid));
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

template <class GridSpaceType>
void Master<GridSpaceType>::GridSet(typename GridSpaceType::GridType *g, const void *buf, const IndexArray &index) {
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

template <class GridSpaceType>
void Client<GridSpaceType>::GridReduce(int id) {
  LOG_DEBUG() << "Client GridReduce(" << id << ")\n";
  typename GridSpaceType::GridType *g = static_cast<typename GridSpaceType::GridType*>(gs_->FindGrid(id));
  PSReduceOp op;
  ipc_->Bcast(&op, sizeof(PSReduceOp), GetMasterRank());
  gs_->ReduceGrid(NULL, op, g);
  return;
}

template <class GridSpaceType>
void Master<GridSpaceType>::GridReduce(void *buf, PSReduceOp op, typename GridSpaceType::GridType *g) {
  LOG_DEBUG() << "Master GridReduce\n";
  NotifyCall(FUNC_GRID_REDUCE, g->id());
  ipc_->Bcast(&op, sizeof(PSReduceOp), rank());
  gs_->ReduceGrid(buf, op, g);
  LOG_DEBUG() << "Master GridReduce done\n";
}

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RPC_H_ */
