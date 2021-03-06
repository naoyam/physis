// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/rpc_mpi.h"
#include "runtime/grid_util.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_runtime.h"
#include "runtime/mpi_wrapper.h"

namespace physis {
namespace runtime {


void Client::Listen() {
  while (true) {
    LOG_INFO() << "Client: listening\n";
    Request req;
    MPI_Bcast(&req, sizeof(Request) / sizeof(int),
              MPI_INT, 0, comm_);
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
  return;
}

Client::Client(const ProcInfo &pinfo, GridSpaceMPI *gs, MPI_Comm comm):
    pinfo_(pinfo), gs_(gs), comm_(comm) {
}


Master::Master(const ProcInfo &pinfo, GridSpaceMPI *gs, MPI_Comm comm):
    pinfo_(pinfo), gs_(gs), comm_(comm) {
}


void Master::NotifyCall(enum RT_FUNC_KIND fkind, int opt) {
  Request r(fkind, opt);
  MPI_Bcast(&r, sizeof(Request), MPI_BYTE, 0, comm_);
}

// Finalize
void Master::Finalize() {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Finalize\n";
  NotifyCall(FUNC_FINALIZE);
  MPI_Finalize();
}

void Client::Finalize() {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Finalize\n";
  MPI_Finalize();
  exit(0);
}

// Barrier
void Master::Barrier() {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Barrier\n";
  NotifyCall(FUNC_BARRIER);
  MPI_Barrier(comm_);
}

void Client::Barrier() {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Barrier\n";
  MPI_Barrier(comm_);
}

// Create
GridMPI *Master::GridNew(PSType type, int elm_size,
                         int num_dims, const IndexArray &size,
                         bool double_buffering,
                         const IndexArray &global_offset,
                         int attr) {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] New\n";
  NotifyCall(FUNC_NEW);
  IndexArray dummy; // Unused in this class (used in Master2)
  RequestNEW req = {type, elm_size, num_dims, size,
                    double_buffering, global_offset, dummy, dummy, attr};
  MPI_Bcast(&req, sizeof(RequestNEW), MPI_BYTE, 0, comm_);
  GridMPI *g = gs_->CreateGrid(type, elm_size, num_dims, size,
                               double_buffering, global_offset,
                               attr);
  return g;
}

void Client::GridNew() {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Create\n";
  RequestNEW req;
  MPI_Bcast(&req, sizeof(RequestNEW), MPI_BYTE, 0, comm_);
  gs_->CreateGrid(req.type, req.elm_size, req.num_dims, req.size,
                  req.double_buffering, req.global_offset, req.attr);
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Create done\n";
  return;
}

// Delete
void Master::GridDelete(GridMPI *g) {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Delete\n";
  NotifyCall(FUNC_DELETE, g->id());
  gs_->DeleteGrid(g);
  return;
}

void Client::GridDelete(int id) {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Delete\n";
  gs_->DeleteGrid(id);
  return;
}

void Master::GridCopyinLocal(GridMPI *g, const void *buf) {
  size_t s = g->local_size().accumulate(g->num_dims()) *
      g->elm_size();
  PSAssert(g->buffer()->size() == s);
  CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                 g->size(), g->buffer()->Get(),
                 g->local_offset(), g->local_size());
}

// Copyin
void Master::GridCopyin(GridMPI *g, const void *buf) {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Copyin\n";


  // copyin to own buffer
  GridCopyinLocal(g, buf);

  // Copyin to remote subgrids
  NotifyCall(FUNC_COPYIN, g->id());  
  BufferHost send_buf;
  for (int i = 1; i < gs_->num_procs(); ++i) {
    IndexArray subgrid_size, subgrid_offset;
    PS_MPI_Recv(&subgrid_offset, sizeof(IndexArray), MPI_BYTE, i,
                0, comm_, MPI_STATUS_IGNORE);
    LOG_VERBOSE_MPI() << "sg offset: " << subgrid_offset << "\n";    
    PS_MPI_Recv(&subgrid_size, sizeof(IndexArray), MPI_BYTE, i,
                0, comm_, MPI_STATUS_IGNORE);
    LOG_VERBOSE_MPI() << "sg size: " << subgrid_size << "\n";        
    size_t gsize = subgrid_size.accumulate(g->num_dims()) *
        g->elm_size();
    if (gsize == 0) continue;
    send_buf.EnsureCapacity(
        subgrid_size.accumulate(g->num_dims()) * g->elm_size());
    CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                   g->size(), send_buf.Get(),
                   subgrid_offset, subgrid_size);
    PS_MPI_Send(send_buf.Get(), gsize, MPI_BYTE, i, 0, comm_);
  }
  return;
}

void Client::GridCopyin(int id) {
  LOG_DEBUG() << "Copyin\n";

  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(id));
  // notify the local offset
  IndexArray ia = g->local_offset();
  PS_MPI_Send(&ia, sizeof(IndexArray), MPI_BYTE, 0, 0, comm_);
  // notify the local size
  ia = g->local_size();
  PS_MPI_Send(&ia, sizeof(IndexArray), MPI_BYTE, 0, 0, comm_);
  if (g->empty()) {
    LOG_DEBUG() << "No copy needed because this grid is empty.\n";
    return;
  }
  // receive the subregion for this process
  Buffer *b = g->buffer();
  PS_MPI_Recv(b->Get(), b->size(), MPI_BYTE, 0, 0, comm_,
              MPI_STATUS_IGNORE);
  //print_grid<float>(g, gs_->my_rank(), std::cerr);
  return;
}

void Master::GridCopyoutLocal(GridMPI *g, void *buf) {
  if (g->empty()) return;

  CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                g->size(), g->buffer()->Get(), g->local_offset(),
                g->local_size());
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
  for (int i = 1; i < gs_->num_procs(); ++i) {
    IndexArray subgrid_size, subgrid_offset;
    PS_MPI_Recv(&subgrid_offset, sizeof(IndexArray), MPI_BYTE, i,
                0, comm_, MPI_STATUS_IGNORE);
    LOG_VERBOSE() << "sg offset: " << subgrid_offset << "\n";
    PS_MPI_Recv(&subgrid_size, sizeof(IndexArray), MPI_BYTE, i,
                0, comm_, MPI_STATUS_IGNORE);
    LOG_VERBOSE() << "sg size: " << subgrid_size << "\n";
    //size_t gsize = sbuf.GetLinearSize(subgrid_size);
    size_t gsize = subgrid_size.accumulate(g->num_dims()) *
        g->elm_size();
    if (gsize == 0) continue;
    recv_buf.EnsureCapacity(gsize);
    PS_MPI_Recv(recv_buf.Get(), gsize, MPI_BYTE,
                i, 0, comm_, MPI_STATUS_IGNORE);
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
  PS_MPI_Send(&ia, sizeof(IndexArray), MPI_BYTE, 0, 0, comm_);
  // notify the local size
  ia = g->local_size();
  PS_MPI_Send(&ia, sizeof(IndexArray), MPI_BYTE, 0, 0, comm_);
  if (g->empty()) {
    LOG_DEBUG() << "No copy needed because this grid is empty.\n";
    return;
  }
  PS_MPI_Send(g->buffer()->Get(), g->buffer()->size(),
              MPI_BYTE, 0, 0, comm_);
  return;
}

void Master::StencilRun(int id, int iter, int num_stencils,
                        void **stencils,
                        unsigned *stencil_sizes) {
  LOG_DEBUG() << "Master StencilRun(" << id << ")\n";
  
  NotifyCall(FUNC_RUN, id);
  // Send the number of iterations
  PS_MPI_Bcast(&iter, 1, MPI_INT, 0, comm_);
  PS_MPI_Bcast(&num_stencils, 1, MPI_INT, 0, comm_);
  PS_MPI_Bcast(stencil_sizes, num_stencils, MPI_UNSIGNED, 0, comm_);
  for (int i = 0; i < num_stencils; ++i) {
    PS_MPI_Bcast(stencils[i], stencil_sizes[i], MPI_BYTE, 0, comm_);
  }
  LOG_DEBUG() << "Calling the stencil function\n";
  // call the stencil obj
  __PS_stencils[id](iter, stencils);
  return;
}

void Client::StencilRun(int id) {
  LOG_DEBUG() << "Client StencilRun(" << id << ")\n";

  // Send the number of iterations
  int iter;  
  PS_MPI_Bcast(&iter, 1, MPI_INT, 0, comm_);
  int num_stencils;
  PS_MPI_Bcast(&num_stencils, 1, MPI_INT, 0, comm_);
  unsigned *stencil_sizes = new unsigned[num_stencils];
  // Send the stencil object size
  PS_MPI_Bcast(stencil_sizes, num_stencils, MPI_UNSIGNED, 0, comm_);
  void **stencils = new void*[num_stencils];
  for (int i = 0; i < num_stencils; ++i) {
    void *sbuf = malloc(stencil_sizes[i]);
    PS_MPI_Bcast(sbuf, stencil_sizes[i], MPI_BYTE, 0, comm_);
    stencils[i] = sbuf;
  }
  LOG_DEBUG() << "Calling the stencil function\n";
  __PS_stencils[id](iter, stencils);
  delete[] stencil_sizes;
  delete[] stencils;
  return;
}

void Client::GridGet(int id) {
  LOG_DEBUG() << "Client GridGet(" << id << ")\n";
  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(id));
  IndexArray zero_offset;
  IndexArray zero_size;
  // just to handle a request from the root
  gs_->LoadSubgrid(g, zero_offset, zero_size, false);
  LOG_DEBUG() << "Client GridGet done\n";
  return;
}

void Master::GridGet(GridMPI *g, void *buf, const IndexArray &index) {
  LOG_DEBUG() << "Master GridGet\n";

  // Notify
  NotifyCall(FUNC_GET, g->id());
  IndexArray size;
  size.Set(1);
  gs_->LoadSubgrid(g, index, size, false);
  //memcpy(buf, g->GetAddress(index), g->elm_size());
  g->Get(index, buf);
  LOG_DEBUG() << "GridGet done\n";
  return;
}

void Client::GridSet(int id) {
  LOG_DEBUG() << "Client GridGet(" << id << ")\n";
  if (id != pinfo_.rank()) {
    // this is not a request to me
    LOG_DEBUG() << "Client GridSet done\n";
    return;
  }

  int gid;
  PS_MPI_Recv(&gid, 1, MPI_INT, 0, 0, comm_, MPI_STATUS_IGNORE);
  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(gid));
  IndexArray index;
  PS_MPI_Recv(&index, sizeof(IndexArray), MPI_BYTE, 0, 0,
              comm_, MPI_STATUS_IGNORE);
  LOG_DEBUG() << "Set index: " << index << "\n";
  void *buf = malloc(g->elm_size());
  PS_MPI_Recv(buf, g->elm_size(), MPI_BYTE, 0, 0,
              comm_, MPI_STATUS_IGNORE);
  //memcpy(g->GetAddress(index), buf, g->elm_size());
  g->Set(index, buf);
  LOG_DEBUG() << "Client GridSet done\n";
  free(buf);
  return;
}

void Master::GridSet(GridMPI *g, const void *buf, const IndexArray &index) {
  LOG_DEBUG() << "Master GridGet\n";

  int peer_rank = gs_->FindOwnerProcess(g, index);
  LOG_DEBUG() << "Owner: " << peer_rank << "\n";

  if (peer_rank != pinfo_.rank()) {
    // NOTE: We don't need to notify all processes but clients are
    // waiting on MPI_Bcast, so P2P methods are not allowed.
    NotifyCall(FUNC_SET, peer_rank);
    int gid = g->id();
    PS_MPI_Send(&gid, 1, MPI_INT, peer_rank, 0, comm_);
    // MPI_Send does not accept const buffer pointer    
    IndexArray t = index;    
    PS_MPI_Send(&t, sizeof(IndexArray), MPI_BYTE, peer_rank, 0, comm_);
    PS_MPI_Send((void*)buf, g->elm_size(), MPI_BYTE, peer_rank, 0, comm_);
  } else {
    //memcpy(g->GetAddress(index), buf, g->elm_size());
    g->Set(index, buf);
  }
}

void Client::GridReduce(int id) {
  LOG_DEBUG() << "Client GridReduce(" << id << ")\n";
  GridMPI *g = static_cast<GridMPI*>(gs_->FindGrid(id));
  PSReduceOp op;
  PS_MPI_Bcast(&op, sizeof(PSReduceOp) / sizeof(int),
               MPI_INT, 0, comm_);
  gs->ReduceGrid(NULL, op, g);
  return;
}

void Master::GridReduce(void *buf, PSReduceOp op, GridMPI *g) {
  LOG_DEBUG() << "Master GridReduce\n";
  NotifyCall(FUNC_GRID_REDUCE, g->id());
  PS_MPI_Bcast(&op, sizeof(PSReduceOp) / sizeof(int),
               MPI_INT, 0, comm_);
  gs->ReduceGrid(buf, op, g);
  LOG_DEBUG() << "Master GridReduce done\n";  
}

} // namespace runtime
} // namespace physis
