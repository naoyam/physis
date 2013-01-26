#include "runtime/rpc_mpi_openmp.h"
#include "runtime/grid_util.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_openmp_runtime.h"
#include "runtime/mpi_wrapper.h"

namespace physis {
namespace runtime {

struct RequestMP {
  RT_FUNC_MP_KIND kind;
  int opt;
  RequestMP(RT_FUNC_MP_KIND k=FUNC_MP_INVALID, int opt=0)
      : kind(k), opt(opt) {}
};

struct RequestMPNEW {
  PSType type;
  int elm_size;
  int num_dims;
  IntArray size;
  bool double_buffering;
  IntArray global_offset;
  int attr;
};


void ClientOpenMP::Listen() {
  while (true) {
    LOG_INFO() << "Client: listening\n";
    RequestMP req;
    MPI_Bcast(&req, sizeof(RequestMP) / sizeof(int),
              MPI_INT, 0, comm_);
    switch (req.kind) {
      case FUNC_MP_FINALIZE:
        LOG_INFO() << "Client: Finalize requested\n";
        Finalize();
        LOG_INFO() << "Client: Finalize done\n";
        break;
      case FUNC_MP_BARRIER:
        LOG_INFO() << "Client: Barrier requested\n";
        Barrier();
        LOG_INFO() << "Client: Barrier done\n";
        break;
      case FUNC_MP_NEW:
        LOG_INFO() << "Client: new requested\n";
        GridNew();
        LOG_INFO() << "Client: new done\n";        
        break;
      case FUNC_MP_DELETE:
        LOG_INFO() << "Client: free requested\n";
        GridDelete(req.opt);
        LOG_INFO() << "Client: free done\n";        
        break;
      case FUNC_MP_COPYIN:
        LOG_INFO() << "Client: copyin requested\n";
        GridCopyin(req.opt);
        LOG_INFO() << "Client: copyin done\n";        
        break;
      case FUNC_MP_COPYOUT:
        LOG_INFO() << "Client: copyout requested\n";
        GridCopyout(req.opt);
        LOG_INFO() << "Client: copyout done\n";        
        break;
      case FUNC_MP_GET:
        LOG_INFO() << "Client: get requested\n";
        GridGet(req.opt);
        LOG_INFO() << "Client: get done\n";        
        break;
      case FUNC_MP_SET:
        LOG_INFO() << "Client: set requested\n";
        GridSet(req.opt);
        LOG_INFO() << "Client: set done\n";        
        break;
      case FUNC_MP_RUN:
        LOG_DEBUG() << "Client: run requested ("
                    << req.opt << ")\n";
        StencilRun(req.opt);
        LOG_DEBUG() << "Client: run done\n";
        break;
      case FUNC_MP_GRID_REDUCE:
        LOG_DEBUG() << "Client: grid reduce requested ("
                    << req.opt << ")\n";
        GridReduce(req.opt);
        LOG_DEBUG() << "Client: grid reduce done\n";
        break;
      case FUNC_MP_INIT_NUMA:
        LOG_DEBUG() << "Client: init numa requested ("
                    << req.opt << ")\n";
        GridInitNUMA(req.opt);
        LOG_DEBUG() << "Client: init numa done\n";
        break;
      case FUNC_MP_INVALID:
        LOG_INFO() << "Client: invaid request\n";
        PSAbort(1);
      default:
        LOG_ERROR() << "Unsupported request: " << req.kind << "\n";
        PSAbort(1);            
    }
  }
  return;
}

ClientOpenMP::ClientOpenMP(
    const ProcInfoOpenMP &pinfo_mp, GridSpaceMPIOpenMP *gs_mp, MPI_Comm comm):
    Client(pinfo_mp, gs_mp, comm),
    pinfo_mp_(pinfo_mp), gs_mp_(gs_mp)
{
}


MasterOpenMP::MasterOpenMP(
    const ProcInfoOpenMP &pinfo_mp, GridSpaceMPIOpenMP *gs_mp, MPI_Comm comm):
    Master(pinfo_mp, gs_mp, comm),
    pinfo_mp_(pinfo_mp), gs_mp_(gs_mp)
{
}


void MasterOpenMP::NotifyCall(enum RT_FUNC_MP_KIND fkind, int opt) {
  RequestMP r(fkind, opt);
  MPI_Bcast(&r, sizeof(RequestMP), MPI_BYTE, 0, comm_);
}

void MasterOpenMP::Finalize() {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Finalize\n";
  NotifyCall(FUNC_MP_FINALIZE);
  MPI_Finalize();
}

// Barrier
void MasterOpenMP::Barrier() {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Barrier\n";
  NotifyCall(FUNC_MP_BARRIER);
  MPI_Barrier(comm_);
}

GridMPIOpenMP *MasterOpenMP::GridNew(PSType type, int elm_size,
                                     int num_dims, const IntArray &size,
                                     bool double_buffering,
                                     const IntArray &global_offset,
                                     int attr) {
  LOG_DEBUG() << "[" << pinfo_mp_.rank() << "] New\n";
  NotifyCall(FUNC_MP_NEW);
  RequestMPNEW req = {type, elm_size, num_dims, size,
                      double_buffering, global_offset, attr};
  MPI_Bcast(&req, sizeof(RequestMPNEW), MPI_BYTE, 0, comm_);
  GridMPIOpenMP *g = gs_mp_->CreateGrid(
      type, elm_size, num_dims, size,
      double_buffering, global_offset,
      pinfo_mp_.division_size(),
      attr);
  return g;
}

void ClientOpenMP::GridNew() {
  LOG_DEBUG() << "[" << pinfo_mp_.rank() << "] Create\n";
  RequestMPNEW req;
  MPI_Bcast(&req, sizeof(RequestMPNEW), MPI_BYTE, 0, comm_);
  gs_mp_->CreateGrid(
      req.type, req.elm_size, req.num_dims, req.size,
      req.double_buffering, req.global_offset,
      pinfo_mp_.division_size(),
      req.attr);
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Create done\n";
  return;
}

// Delete
void MasterOpenMP::GridDelete(GridMPIOpenMP *g) {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Delete\n";
  NotifyCall(FUNC_MP_DELETE, g->id());
  gs_->DeleteGrid(g);
  return;
}

void MasterOpenMP::GridCopyinLocal(GridMPIOpenMP *g, void *buf) {
#if 0
  size_t s = g->local_size().accumulate(g->num_dims()) *
      g->elm_size();
  PSAssert(g->buffer()->GetLinearSize() == s);
  CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                 g->size(), g->buffer()->Get(),
                 g->local_offset(), g->local_size());
#else
  BufferHostOpenMP *gbuf_MP =
      dynamic_cast<BufferHostOpenMP *>(g->buffer());
  PSAssert(gbuf_MP);
  g->CopyinoutSubgrid(
      1,
      g->elm_size(), g->num_elms(),
      buf,
      g->size(),
      gbuf_MP,
      g->local_offset(), g->local_size()
                      );
#endif
}


// Copyin
void MasterOpenMP::GridCopyin(GridMPIOpenMP *g, void *buf) {
  LOG_DEBUG() << "[" << pinfo_.rank() << "] Copyin\n";

  // copyin to own buffer
  GridCopyinLocal(g, buf);

  // Copyin to remote subgrids
  NotifyCall(FUNC_MP_COPYIN, g->id()); 
  // Okay with BufferHost!! 
  BufferHost sbuf(g->num_dims(), g->elm_size());
  for (int i = 1; i < gs_->num_procs(); ++i) {
    // Repeat:
    // - receiving offset
    // - receiving grid
    // - copyout the needed area
    // - sending it to child MPI thread
    // ... for each NUMA memory in the grid
    //     in child MPI thread
    //
    //
    IntArray subgrid_size, subgrid_offset;

    unsigned int num_cpu = 0;
    unsigned int cpuid = 0;
    PS_MPI_Recv(
        &num_cpu, sizeof(unsigned int), MPI_BYTE, i,
        0, comm_, MPI_STATUS_IGNORE);
    LOG_VERBOSE_MPI() 
        << "Child thread " << i
        << " has " << num_cpu << "memory blocks\n";
    for (cpuid = 0; cpuid < num_cpu; cpuid++) {
      LOG_VERBOSE_MPI() 
          << "Thread " << i << " Memory id:" << cpuid
          << "\n";
      PS_MPI_Recv(&subgrid_offset, sizeof(IntArray), MPI_BYTE, i,
                  0, comm_, MPI_STATUS_IGNORE);
      LOG_VERBOSE_MPI() << "sg offset: " << subgrid_offset << "\n";    
      PS_MPI_Recv(&subgrid_size, sizeof(IntArray), MPI_BYTE, i,
                  0, comm_, MPI_STATUS_IGNORE);
      LOG_VERBOSE_MPI() << "sg size: " << subgrid_size << "\n";        
      size_t gsize = sbuf.GetLinearSize(subgrid_size);

      if (gsize == 0) continue;

      sbuf.EnsureCapacity(subgrid_size);
      // Okay with CopyoutSubgrid
      CopyoutSubgrid(g->elm_size(), g->num_dims(), buf,
                     g->size(), sbuf.Get(),
                     subgrid_offset, subgrid_size);
      PS_MPI_Send(sbuf.Get(), gsize, MPI_BYTE, i, 0, comm_);
    } // for (cpuid = 0; cpuid < num_cpu, cpuid++)
  } // for (int i = 1; i < gs_->num_procs(); ++i)
  return;
}

void ClientOpenMP::GridCopyin(int id) {
  LOG_DEBUG() << "Copyin\n";
  GridMPIOpenMP *g = 
      dynamic_cast<GridMPIOpenMP*>(gs_mp_->FindGrid(id));
  PSAssert(g);

  // The opposite
  // - send offset
  // - send gridsize
  // - receive the data from parent MPI thread
  // - copyin to memory
  // ... for each NUMA memory in the grid
  //     in child MPI thread

  BufferHostOpenMP *gbuf_MP =
      dynamic_cast<BufferHostOpenMP *>(g->buffer());
  PSAssert(gbuf_MP);
  PSAssert(g->division() >= gbuf_MP->MPdivision());

  unsigned int num_cpu = 
      gbuf_MP->MPdivision().accumulate(g->num_dims());
  unsigned int cpuid = 0;

  // First notify cpu number
  PS_MPI_Send(&num_cpu, sizeof(unsigned int),
              MPI_BYTE, 0, 0, comm_);

  for (cpuid = 0; cpuid < num_cpu; cpuid++) {
    unsigned int cpugrid[PS_MAX_DIM] = {0, 0, 0};
    {
      unsigned int fact_cpu = 1;
      for (unsigned int dim = 0; dim < (unsigned int) g->num_dims(); dim++) {
        cpugrid[dim] = (cpuid / fact_cpu) % gbuf_MP->MPdivision()[dim];
        fact_cpu *= gbuf_MP->MPdivision()[dim];
      }
    }
    // notify the local offset
    IntArray offset(0,0,0);
    offset += g->local_offset();
    for (unsigned int dim = 0; dim < (unsigned int) g->num_dims(); dim++) {
      offset[dim] += (gbuf_MP->MPoffset())[dim][cpugrid[dim]];
    }
    PS_MPI_Send(&offset, sizeof(IntArray), MPI_BYTE, 0, 0, comm_);

    // notify the local size
    IntArray width(0,0,0);
    for (unsigned int dim = 0; dim < (unsigned int) g->num_dims(); dim++) {
      width[dim] = (gbuf_MP->MPwidth())[dim][cpugrid[dim]];
    }
    PS_MPI_Send(&width, sizeof(IntArray), MPI_BYTE, 0, 0, comm_);

    if (!(width.accumulate(g->num_dims()))) {
      LOG_DEBUG() << "No copy needed because this grid is empty.\n";
      continue;
    }
    // receive the subregion for this process
#if 0
    void *cur_buf_ptr = g->buffer()->Get();
    g->buffer()->MPIRecv(0, comm_, IntArray((index_t)0), g->local_size());
#else
    void *cur_buf_ptr = (gbuf_MP->Get_MP())[cpuid];
    PS_MPI_Recv(
        cur_buf_ptr,
        gbuf_MP->GetLinearSize(width), MPI_BYTE, 0,
        0, comm_, MPI_STATUS_IGNORE
                );

#endif

    //print_grid<float>(g, gs_->my_rank(), std::cerr);
  } // for (cpuid = 0; cpuid < num_cpu; cpuid++)

  return;
}

void MasterOpenMP::GridCopyoutLocal(GridMPIOpenMP *g, void *buf) {
  if (g->empty()) return;

#if 0  
  CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                g->size(), g->buffer()->Get(), g->local_offset(),
                g->local_size());
#else
  BufferHostOpenMP *gbuf_MP =
      dynamic_cast<BufferHostOpenMP *>(g->buffer());
  PSAssert(gbuf_MP);
  g->CopyinoutSubgrid(
      0,
      g->elm_size(), g->num_dims(),
      buf,
      g->size(),
      gbuf_MP,
      g->local_offset(), g->local_size()
                      );
#endif
  return;
}


// Copyout
void MasterOpenMP::GridCopyout(GridMPIOpenMP *g, void *buf) {
  LOG_DEBUG() << "Copyout\n";

  // Copyout self
  GridCopyoutLocal(g, buf);
  
  // Copyout from remote grids
  NotifyCall(FUNC_MP_COPYOUT, g->id());
  // Okay with BufferHost!! 
  BufferHost sbuf(g->num_dims(), g->elm_size());
  for (int i = 1; i < gs_->num_procs(); ++i) {
    // Repeat:
    // - receiving offset
    // - receiving grid
    // - receiving it from child MPI thread
    // - copyin the needed area
    // ... for each NUMA memory in the grid
    //     in child MPI thread
    //
    //
    IntArray subgrid_size, subgrid_offset;

    unsigned int num_cpu = 0;
    unsigned int cpuid = 0;
    PS_MPI_Recv(
        &num_cpu, sizeof(unsigned int), MPI_BYTE, i,
        0, comm_, MPI_STATUS_IGNORE);
    LOG_VERBOSE_MPI() 
        << "Child thread " << i
        << " has " << num_cpu << "memory blocks\n";
    for (cpuid = 0; cpuid < num_cpu; cpuid++) {
      LOG_VERBOSE_MPI() 
          << "Thread " << i << " Memory id:" << cpuid
          << "\n";
      PS_MPI_Recv(&subgrid_offset, sizeof(IntArray), MPI_BYTE, i,
                  0, comm_, MPI_STATUS_IGNORE);
      LOG_VERBOSE() << "sg offset: " << subgrid_offset << "\n";
      PS_MPI_Recv(&subgrid_size, sizeof(IntArray), MPI_BYTE, i,
                  0, comm_, MPI_STATUS_IGNORE);
      LOG_VERBOSE() << "sg size: " << subgrid_size << "\n";
      size_t gsize = sbuf.GetLinearSize(subgrid_size);

      if (gsize == 0) continue;

      sbuf.MPIRecv(i, comm_, IndexArray(), subgrid_size);
      // Okay with CopyoutSubgrid
      CopyinSubgrid(g->elm_size(), g->num_dims(), buf,
                    g->size(), sbuf.Get(), subgrid_offset,
                    subgrid_size);
    } // for (cpuid = 0; cpuid < num_cpu, cpuid++)
  } // for (int i = 1; i < gs_->num_procs(); ++i)
}

void ClientOpenMP::GridCopyout(int id) {
  LOG_DEBUG() << "Copyout\n";
  GridMPIOpenMP *g = 
      dynamic_cast<GridMPIOpenMP*>(gs_mp_->FindGrid(id));
  PSAssert(g);

  // The opposite
  // - send offset
  // - send gridsize
  // - send the data from parent MPI thread
  // ... for each NUMA memory in the grid
  //     in child MPI thread

  BufferHostOpenMP *gbuf_MP =
      dynamic_cast<BufferHostOpenMP *>(g->buffer());
  PSAssert(gbuf_MP);
  PSAssert(g->division() >= gbuf_MP->MPdivision());

  unsigned int num_cpu = 
      gbuf_MP->MPdivision().accumulate(g->num_dims());
  unsigned int cpuid = 0;

  // First notify cpu number
  PS_MPI_Send(&num_cpu, sizeof(unsigned int),
              MPI_BYTE, 0, 0, comm_);

  for (cpuid = 0; cpuid < num_cpu; cpuid++) {
    unsigned int cpugrid[PS_MAX_DIM] = {0, 0, 0};
    {
      unsigned int fact_cpu = 1;
      for (unsigned int dim = 0; dim < (unsigned int) g->num_dims(); dim++) {
        cpugrid[dim] = (cpuid / fact_cpu) % gbuf_MP->MPdivision()[dim];
        fact_cpu *= gbuf_MP->MPdivision()[dim];
      }
    }
    // notify the local offset
    IntArray offset(0,0,0);
    offset += g->local_offset();
    for (unsigned int dim = 0; dim < (unsigned int) g->num_dims(); dim++) {
      offset[dim] += (gbuf_MP->MPoffset())[dim][cpugrid[dim]];
    }
    PS_MPI_Send(&offset, sizeof(IntArray), MPI_BYTE, 0, 0, comm_);

    // notify the local size
    IntArray width(0,0,0);
    for (unsigned int dim = 0; dim < (unsigned int) g->num_dims(); dim++) {
      width[dim] = (gbuf_MP->MPwidth())[dim][cpugrid[dim]];
    }
    PS_MPI_Send(&width, sizeof(IntArray), MPI_BYTE, 0, 0, comm_);

    if (!(width.accumulate(g->num_dims()))) {
      LOG_DEBUG() << "No copy needed because this grid is empty.\n";
      continue;
    }

    if (!(width.accumulate(g->num_dims()))) {
      LOG_DEBUG() << "No copy needed because this grid is empty.\n";
      continue;
    }

#if 0
    g->buffer()->MPISend(0, comm_, IntArray((index_t)0), g->local_size());
#else
    void *cur_buf_ptr = (gbuf_MP->Get_MP())[cpuid];
    PS_MPI_Send(
        cur_buf_ptr,
        gbuf_MP->GetLinearSize(width), MPI_BYTE, 0,
        0, comm_
                );
#endif
  } // for (cpuid = 0; cpuid < num_cpu; cpuid++)

  return;
}

void MasterOpenMP::StencilRun(int id, int iter, int num_stencils,
                              void **stencils,
                              int *stencil_sizes) {
  LOG_DEBUG() << "Master StencilRun(" << id << ")\n";
  
  NotifyCall(FUNC_MP_RUN, id);
  // Send the number of iterations
  PS_MPI_Bcast(&iter, 1, MPI_INT, 0, comm_);
  PS_MPI_Bcast(&num_stencils, 1, MPI_INT, 0, comm_);
  PS_MPI_Bcast(stencil_sizes, num_stencils, MPI_INT, 0, comm_);
  for (int i = 0; i < num_stencils; ++i) {
    PS_MPI_Bcast(stencils[i], stencil_sizes[i], MPI_BYTE, 0, comm_);
  }
  LOG_DEBUG() << "Calling the stencil function\n";
  // call the stencil obj
  __PS_stencils[id](iter, stencils);
  return;
}

void MasterOpenMP::GridGet(GridMPIOpenMP *g, void *buf, const IntArray &index) {
  LOG_DEBUG() << "Master GridGet\n";

  // Notify
  NotifyCall(FUNC_MP_GET, g->id());
  IntArray size;
  size.assign(1);
  gs_->LoadSubgrid(g, index, size, false);
  //memcpy(buf, g->GetAddress(index), g->elm_size());
  g->Get(index, buf);
  LOG_DEBUG() << "GridGet done\n";
  return;
}

void MasterOpenMP::GridSet(GridMPIOpenMP *g, const void *buf, const IntArray &index) {
  LOG_DEBUG() << "Master GridGet\n";

  int peer_rank = gs_->FindOwnerProcess(g, index);
  LOG_DEBUG() << "Owner: " << peer_rank << "\n";

  if (peer_rank != pinfo_.rank()) {
    // NOTE: We don't need to notify all processes but clients are
    // waiting on MPI_Bcast, so P2P methods are not allowed.
    NotifyCall(FUNC_MP_SET, peer_rank);
    int gid = g->id();
    PS_MPI_Send(&gid, 1, MPI_INT, peer_rank, 0, comm_);
    // MPI_Send does not accept const buffer pointer    
    IntArray t = index;    
    PS_MPI_Send(&t, sizeof(IntArray), MPI_BYTE, peer_rank, 0, comm_);
    PS_MPI_Send((void*)buf, g->elm_size(), MPI_BYTE, peer_rank, 0, comm_);
  } else {
    //memcpy(g->GetAddress(index), buf, g->elm_size());
    g->Set(index, buf);
  }
}

void MasterOpenMP::GridReduce(void *buf, PSReduceOp op, GridMPIOpenMP *g) {
  LOG_DEBUG() << "Master GridReduce\n";
  NotifyCall(FUNC_MP_GRID_REDUCE, g->id());
  PS_MPI_Bcast(&op, sizeof(PSReduceOp) / sizeof(int),
               MPI_INT, 0, comm_);
  gs->ReduceGrid(buf, op, g);
  LOG_DEBUG() << "Master GridReduce done\n";  
}

void MasterOpenMP::GridInitNUMA(GridMPIOpenMP *g, unsigned int maxMPthread)
{
  LOG_DEBUG() << "Master GridINitNUMA\n";
  NotifyCall(FUNC_MP_INIT_NUMA, g->id());
  for (int i = 1; i < gs_->num_procs(); ++i) {
    // Send the max openmp thread number
    unsigned int sendbuf = maxMPthread;
    PS_MPI_Send(&sendbuf, sizeof(unsigned int), MPI_BYTE, i, 0, comm_);
  } // for (int i = 1; i < gs_->num_procs(); ++i)
  g->InitNUMA(maxMPthread);
}

void ClientOpenMP::GridInitNUMA(int gridid)
{
  LOG_DEBUG() << "Client GridINitNUMA(" << gridid << ")\n";
  GridMPIOpenMP *g = dynamic_cast<GridMPIOpenMP *>(gs_->FindGrid(gridid));
  PSAssert(g);
  // Receive the max openmp therad number
  unsigned int maxMPthread = 0;
  PS_MPI_Recv(
      &maxMPthread, sizeof(unsigned int), MPI_BYTE, 0,
      0, comm_, MPI_STATUS_IGNORE
              );
  g->InitNUMA(maxMPthread);
}

} // namespace runtime
} // namespace physis
