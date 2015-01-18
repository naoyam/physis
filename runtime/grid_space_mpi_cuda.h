// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_GRID_SPACE_MPI_CUDA_H_
#define PHYSIS_RUNTIME_GRID_SPACE_MPI_CUDA_H_

#include "runtime/grid_mpi_cuda_exp.h"

namespace physis {
namespace runtime {

template <class GridType>
class GridSpaceMPICUDA: public GridSpaceMPI<GridType> {
 public:
  //using GridSpaceMPI<GridType>::ExchangeBoundaries;  
  GridSpaceMPICUDA(int num_dims, const IndexArray &global_size,
                   int proc_num_dims, const IntArray &proc_size,
                   InterProcComm &ipc);
  virtual ~GridSpaceMPICUDA();

  //! Send halo boundaries for accesses in a dimension.
  /*!
    The halo width for the stencl function is given by halo_width, but
    this class sends the whole halo region for this subgrid. Thus,
    halo_width is not used.
   */
  virtual bool SendBoundaries(GridType *grid,
                              int member, int dim,
                              const Width2 &halo_width,
                              bool forward, bool diagonal, bool periodic,
                              DataCopyProfile &prof,
                              MPI_Request &req) const;

  //! Receive halo boundaries for accesses in a dimension.
  /*!
    The halo width for the stencl function is given by halo_width, but
    this class sends the whole halo region for this subgrid. Thus,
    halo_width is not used.
   */
  virtual bool RecvBoundaries(GridType *grid,
                              int member, int dim,
                              const Width2 &halo_width,
                              bool forward, bool diagonal, bool periodic,
                              DataCopyProfile &prof) const;
  
  virtual void ExchangeBoundaries(GridType *grid,
                                  int member, int dim,
                                  const Width2 &halo_width,
                                  bool diagonal,
                                  bool periodic) const;
#ifdef NEIGHBOR_EXCHANGE_MULTI_STAGE  
  virtual void ExchangeBoundariesStage1(GridType *grid,
                                        int dim,
                                        unsigned halo_fw_width,
                                        unsigned halo_bw_width,
                                        bool diagonal,
                                        bool periodic) const;
  virtual void ExchangeBoundariesStage2(GridType *grid,
                                        int dim,
                                        unsigned halo_fw_width,
                                        unsigned halo_bw_width,
                                        bool diagonal,
                                        bool periodic) const;
#endif // NEIGHBOR_EXCHANGE_MULTI_STAGE
  //using GridSpaceMPI::LoadNeighbor;
  virtual GridType *LoadNeighbor(GridType *g,
                                 int member,
                                 const IndexArray &offset_min,
                                 const IndexArray &offset_max,
                                 bool diagonal,
                                 bool reuse,
                                 bool periodic,
                                 cudaStream_t cuda_stream);
  virtual GridType *LoadNeighbor(GridType *g,
                                 const IndexArray &offset_min,
                                 const IndexArray &offset_max,
                                 bool diagonal,
                                 bool reuse,
                                 bool periodic,
                                 cudaStream_t cuda_stream);
  
  
#ifdef NEIGHBOR_EXCHANGE_MULTI_STAGE  
  virtual GridType *LoadNeighborStage1(GridType *g,
                                       const IndexArray &offset_min,
                                       const IndexArray &offset_max,
                                       bool diagonal,
                                       bool reuse,
                                       bool periodic,
                                       cudaStream_t cuda_stream);
  virtual GridType *LoadNeighborStage2(GridType *g,
                                       const IndexArray &offset_min,
                                       const IndexArray &offset_max,
                                       bool diagonal,
                                       bool reuse,
                                       bool periodic,
                                       cudaStream_t cuda_stream);
#endif // NEIGHBOR_EXCHANGE_MULTI_STAGE
  
#ifdef RUNTIME_LOAD_SUBGRID
  virtual void HandleFetchRequest(GridRequest &req, GridType *g);
  virtual void HandleFetchReply(GridRequest &req, GridType *g,
                                std::map<int, FetchInfo> &fetch_map,  GridType *sg);
#endif
  
  virtual std::ostream& PrintLoadNeighborProf(std::ostream &os) const;

  //! Reduce a grid with binary operator op.
  /*
   * \param out The destination scalar buffer.
   * \param op The binary operator to apply.   
   * \param g The grid to reduce.
   * \return The number of reduced elements.
   */
  virtual int ReduceGrid(void *out, PSReduceOp op, GridType *g);
  
 protected:
#ifdef RUNTIME_LOAD_SUBGRID  
  BufferHost *fetch_host_buf_;
#endif  

};

template <class GridType>
GridSpaceMPICUDA<GridType>::GridSpaceMPICUDA(
    int num_dims, const IndexArray &global_size,
    int proc_num_dims, const IntArray &proc_size, 
    InterProcComm &ipc):
    GridSpaceMPI<GridType>(num_dims, global_size, proc_num_dims,
                           proc_size, ipc) {
#ifdef RUNTIME_LOAD_SUBGRID  
  fetch_host_buf_ = new BufferHost();
#endif  
}

template <class GridType>
GridSpaceMPICUDA<GridType>::~GridSpaceMPICUDA() {
#ifdef RUNTIME_LOAD_SUBGRID  
  delete fetch_host_buf_;
#endif  
}

#ifdef NEIGHBOR_EXCHANGE_MULTI_STAGE
// Just copy out halo from GPU memory
template <class GridType>
void GridSpaceMPICUDA<GridType>::ExchangeBoundariesStage1(
    GridMPI *g, int dim, unsigned halo_fw_width,
    unsigned halo_bw_width, bool diagonal,
    bool periodic) const {
  
  GridType *grid = static_cast<GridType*>(g);
  if (grid->empty_) return;

  LOG_DEBUG() << "Periodic grid?: " << periodic << "\n";

  DataCopyProfile *profs = find<int, DataCopyProfile*>(
      load_neighbor_prof_, g->id(), NULL);
  DataCopyProfile &prof_upw = profs[dim*2];
  DataCopyProfile &prof_dwn = profs[dim*2+1];  
  Stopwatch st;
  // Sends out the halo for backward access

  if (halo_bw_width > 0 &&
      (periodic ||
       grid->local_offset_[dim] + grid->local_size_[dim] < grid->size_[dim])) {
#if defined(PS_VERBOSE)            
    LOG_DEBUG() << "Sending halo of " << bw_size << " elements"
              << " for bw access to " << fw_peer << "\n";
#endif    
    st.Start();
    grid->CopyoutHalo(dim, halo_bw_width, false, diagonal);
    prof_upw.gpu_to_cpu += st.Stop();
  }
  
  // Sends out the halo for forward access
  if (halo_fw_width > 0 &&
      (periodic || grid->local_offset_[dim] > 0)) {
#if defined(PS_VERBOSE)            
    LOG_DEBUG() << "Sending halo of " << fw_size << " elements"
                << " for fw access to " << bw_peer << "\n";
#endif    
    st.Start();    
    grid->CopyoutHalo(dim, halo_fw_width, true, diagonal);
    prof_dwn.gpu_to_cpu += st.Stop();
  }
}

// Exchanges halos on CPU memory, and copy in the new halo to GPU
// memory. Assumes the new halo to exchange is already copied out of
// GPU memory at Stage 1.
// Note: width is unsigned.
template <class GridType>
void GridSpaceMPICUDA<GridType>::ExchangeBoundariesStage2(
    GridType *g, int dim, unsigned halo_fw_width,
    unsigned halo_bw_width, bool diagonal, bool periodic) const {

  GridType *grid = static_cast<GridType*>(g);
  if (grid->empty_) return;

  int fw_peer = fw_neighbors_[dim];
  int bw_peer = bw_neighbors_[dim];
  ssize_t fw_size = grid->CalcHaloSize(dim, halo_fw_width, diagonal);
  ssize_t bw_size = grid->CalcHaloSize(dim, halo_bw_width, diagonal);
  LOG_DEBUG() << "Periodic grid?: " << periodic << "\n";

  DataCopyProfile *profs = find<int, DataCopyProfile*>(
      load_neighbor_prof_, g->id(), NULL);
  DataCopyProfile &prof_upw = profs[dim*2];
  DataCopyProfile &prof_dwn = profs[dim*2+1];  
  Stopwatch st;

  bool req_bw_active = false;
  bool req_fw_active = false;
  MPI_Request req_bw;
  MPI_Request req_fw;
  
  if (halo_bw_width > 0 &&
      (periodic ||
       grid->local_offset_[dim] + grid->local_size_[dim] < grid->size_[dim])) {
    st.Start();
    grid->halo_self_mpi_[dim][0]->MPIIsend(fw_peer, comm_, &req_bw,
                                           IndexArray(), IndexArray(bw_size));
    prof_upw.cpu_out += st.Stop();
    req_bw_active = true;
  }
  
  // Sends out the halo for forward access
  if (halo_fw_width > 0 &&
      (periodic || grid->local_offset_[dim] > 0)) {
    st.Start();        
    grid->halo_self_mpi_[dim][1]->MPIIsend(bw_peer, comm_, &req_fw,
                                           IndexArray(), IndexArray(fw_size));
    prof_dwn.cpu_out += st.Stop();
    req_fw_active = true;
  }

  // Receiving halo for backward access
  RecvBoundaries(grid, dim, halo_width, false, diagonal,
                 periodic, bw_size, prof_dwn);
  
  // Receiving halo for forward access
  RecvBoundaries(grid, dim, halo_width, true, diagonal,
                 periodic, fw_size, prof_upw);

  // Ensure the exchanges are done. Otherwise, the send buffers can be
  // overwritten by the next iteration.
  if (req_fw_active) CHECK_MPI(MPI_Wait(&req_fw, MPI_STATUS_IGNORE));
  if (req_bw_active) CHECK_MPI(MPI_Wait(&req_bw, MPI_STATUS_IGNORE));

  grid->FixupBufferPointers();
  return;
}
#endif // NEIGHBOR_EXCHANGE_MULTI_STAGE

template <class GridType>
bool GridSpaceMPICUDA<GridType>::SendBoundaries(
    GridType *grid, int member, int dim, const Width2 &halo_width,
    bool forward, bool diagonal, bool periodic, DataCopyProfile &prof,
    MPI_Request &req) const {
  unsigned width = forward ? halo_width.fw[dim] : halo_width.bw[dim];
  // Nothing to do since the width is zero  
  if (width == 0) {
    return false;
  }
  // No other process to share this dimension
  if (this->proc_size_[dim] == 1) {
    return false;
  }
  // Do nothing if this process is on the end of the dimension and
  // periodic boundary is not set.
  if (!periodic &&
      ((!forward && grid->local_offset_[dim] + grid->local_size_[dim]
        == grid->size_[dim]) ||
       (forward && grid->local_offset_[dim] == 0))) {
    return false;
  }
  int peer = forward ? this->bw_neighbors_[dim] : this->fw_neighbors_[dim];
  
  LOG_DEBUG() << "Sending to " << peer << "\n";
  
  Stopwatch st;  
  st.Start();
  grid->CopyoutHalo(dim, halo_width, forward, diagonal, member);
  prof.gpu_to_cpu += st.Stop();
#if defined(PS_VERBOSE)
  LOG_VERBOSE() << "own halo copied to host ->";
  grid->GetHaloSelfHost(dim, forward)->print<float>(std::cerr);
#endif

  // Now the halo to be sent to the peer is on the host memory, ready
  // to be sent to the peer.

  BufferHost *halo_buf = grid->GetHaloSelfHost(dim, forward, member);
  size_t halo_size = halo_buf->size();

  LOG_DEBUG() << "Sending to " << peer << "\n";
  LOG_DEBUG() << "Sending halo on host of " << halo_size
              << " bytes to " << peer << "\n";
  
  st.Start();
  CHECK_MPI(PS_MPI_Isend(
      halo_buf->Get(), halo_size, MPI_BYTE, peer, 0, this->comm_, &req));
  prof.cpu_out += st.Stop();
  return true;
}

template <class GridType>
bool GridSpaceMPICUDA<GridType>::RecvBoundaries(
    GridType *grid, int member, int dim, const Width2 &halo_width,
    bool forward, bool diagonal, bool periodic, DataCopyProfile &prof) const {
  int peer = forward ? this->fw_neighbors_[dim] : this->bw_neighbors_[dim];
  bool is_last_process =
      grid->local_offset_[dim] + grid->local_size_[dim]
      == grid->size_[dim];
  bool is_first_process = grid->local_offset_[dim] == 0;
  Stopwatch st;
  const unsigned width = halo_width(forward)[dim];
  if (width == 0 ||
      (!periodic && ((forward && is_last_process) ||
                        (!forward && is_first_process)))) {
    return false;
  }
  // No other process to share this dimension
  if (this->proc_size_[dim] == 1) {
    return false;
  }

#if defined(PS_DEBUG)
  if (!periodic) {
    if( (forward && grid->local_offset_[dim] +
         grid->local_size_[dim] + (PSIndex)width > grid->size_[dim]) ||
        (!forward && grid->local_offset_[dim] - width < 0)
        ) {
      LOG_ERROR() << "Off limit accesses: "
                  << "local_offset: " << grid->local_offset_[dim]
                  << ", local_size: " << grid->local_size_[dim]
                  << ", width: " << width
                  << ", grid size: " << grid->size_[dim]
                  << "\n";
      LOG_ERROR() << "is_first: " << is_first_process
                  << ", is_last: " << is_last_process
                  << ", forward: " << forward
                  << "\n";
      PSAbort(1);
    }
  }
#endif  

  BufferHost *halo_buf = grid->GetHaloPeerHost(dim, forward, member);
  LOG_DEBUG() << "Receiving halo of " << halo_buf->size()
              << " bytes from " << peer << "\n";

  // Receive from the peer MPI process
  st.Start();
  CHECK_MPI(PS_MPI_Recv(
      halo_buf->Get(), halo_buf->size(), MPI_BYTE, peer, 0,
      this->comm_, MPI_STATUS_IGNORE));
  prof.cpu_in += st.Stop();

  // Copy to device buffer
  st.Start();
  grid->CopyinHalo(dim, halo_width, forward, diagonal, member);
  prof.cpu_to_gpu += st.Stop();
  return true;
}

// Note: width is unsigned.
template <class GridType>
void GridSpaceMPICUDA<GridType>::ExchangeBoundaries(
    GridType *grid, int member, int dim, const Width2 &halo_width,
    bool diagonal, bool periodic) const {

  if (grid->empty_) return;

  LOG_DEBUG() << "Periodic grid?: " << periodic << "\n";

  DataCopyProfile *profs = find<int, DataCopyProfile*>(
      this->load_neighbor_prof_, grid->id(), NULL);
  DataCopyProfile &prof_upw = profs[dim*2];
  DataCopyProfile &prof_dwn = profs[dim*2+1];  
  MPI_Request req_bw, req_fw;

  // Sends out the halo for backward access
  bool req_bw_active =
      SendBoundaries(grid, member, dim, halo_width, false, diagonal,
                     periodic, prof_upw, req_bw);
  
  // Sends out the halo for forward access
  bool req_fw_active =
      SendBoundaries(grid, member, dim, halo_width, true, diagonal,
                     periodic, prof_dwn, req_fw);

  // Receiving halo for backward access
  RecvBoundaries(grid, member, dim, halo_width, false, diagonal,
                 periodic, prof_dwn);
  
  // Receiving halo for forward access
  RecvBoundaries(grid, member, dim, halo_width, true, diagonal,
                 periodic, prof_upw);

  // Ensure the exchanges are done. Otherwise, the send buffers can be
  // overwritten by the next iteration.
  if (req_fw_active) CHECK_MPI(MPI_Wait(&req_fw, MPI_STATUS_IGNORE));
  if (req_bw_active) CHECK_MPI(MPI_Wait(&req_bw, MPI_STATUS_IGNORE));

  grid->FixupBufferPointers();
  return;
}

#ifdef RUNTIME_LOAD_SUBGRID
template <class GridType>
void GridSpaceMPICUDA<GridType>::HandleFetchRequest(GridRequest &req, GridType *g) {
  LOG_DEBUG() << "HandleFetchRequest\n";
  GridType *gm = static_cast<GridType*>(g);
  FetchInfo finfo;
  int nd = num_dims_;
  CHECK_MPI(MPI_Recv(&finfo, sizeof(FetchInfo), MPI_BYTE,
                     req.my_rank, 0, comm_, MPI_STATUS_IGNORE));
  size_t bytes = finfo.peer_size.accumulate(nd) * g->elm_size();
  fetch_host_buf->EnsureCapacity(bytes);
  static_cast<BufferCUDADevExp*>(gm->buffer())->Copyout(
      *buf, finfo.peer_offset - g->local_offset(), finfo.peer_size);
  SendGridRequest(my_rank_, req.my_rank, comm_, FETCH_REPLY);
  MPI_Request mr;
  buf->MPIIsend(req.my_rank, comm_, &mr, IndexArray(), IndexArray(bytes));
  //CHECK_MPI(PS_MPI_Isend(buf, bytes, MPI_BYTE, req.my_rank, 0, comm_, &mr));
}

template <class GridType>
void GridSpaceMPICUDA<GridType>::HandleFetchReply(GridRequest &req, GridType *g,
                                                  std::map<int, FetchInfo> &fetch_map,
                                                  GridType *sg) {
  LOG_DEBUG() << "HandleFetchReply\n";
  GridType *sgm = static_cast<GridType*>(sg);
  const FetchInfo &finfo = fetch_map[req.my_rank];
  PSAssert(GetProcessRank(finfo.peer_index) == req.my_rank);
  size_t bytes = finfo.peer_size.accumulate(num_dims_) * g->elm_size();
  LOG_DEBUG() << "Fetch reply data size: " << bytes << "\n";
  buf->EnsureCapacity(bytes);
  buf->Buffer::MPIRecv(req.my_rank, comm_, IndexArray(bytes));
  LOG_DEBUG() << "Fetch reply received\n";
  static_cast<BufferCUDADev3D*>(sgm->buffer())->Copyin(
      *buf, finfo.peer_offset - sg->local_offset(),
      finfo.peer_size);
  sgm->FixupBufferPointers();
  return;
}
#endif // RUNTIME_LOAD_SUBGRID

inline PSIndex GridCalcOffsetExp(const IndexArray &index,
                                 const IndexArray &size,
                                 size_t pitch) {
  return index[0] + index[1] * pitch + index[2] * pitch * size[1];
}


template <class GridType>
GridType *GridSpaceMPICUDA<GridType>::LoadNeighbor(
    GridType *g, int member,
    const IndexArray &offset_min,
    const IndexArray &offset_max,
    bool diagonal,  bool reuse, bool periodic,
    cudaStream_t cuda_stream) {
  
  // set the stream of buffer by the stream parameter
  //g->SetCUDAStream(cuda_stream);
  GridType *rg = GridSpaceMPI<GridType>::LoadNeighbor(
      g, member, offset_min, offset_max,
      diagonal, reuse, periodic);

  //g->SetCUDAStream(0);
  return rg;
}

template <class GridType>
GridType *GridSpaceMPICUDA<GridType>::LoadNeighbor(
    GridType *g,
    const IndexArray &offset_min,
    const IndexArray &offset_max,
    bool diagonal,  bool reuse, bool periodic,
    cudaStream_t cuda_stream) {
  
  // set the stream of buffer by the stream parameter
  //g->SetCUDAStream(cuda_stream);
  GridType *rg = GridSpaceMPI<GridType>::LoadNeighbor(
      g, offset_min, offset_max,
      diagonal, reuse, periodic);

  //g->SetCUDAStream(0);
  return rg;
}


#ifdef NEIGHBOR_EXCHANGE_MULTI_STAGE
template <class GridType>
GridMPI *GridSpaceMPICUDA<GridType>::LoadNeighborStage1(
    GridType *g,
    const IndexArray &offset_min,
    const IndexArray &offset_max,
    bool diagonal,  bool reuse, bool periodic,
    cudaStream_t cuda_stream) {
  
  // set the stream of buffer by the stream parameter
  GridType *gmc = static_cast<GridType*>(g);
  gmc->SetCUDAStream(cuda_stream);

  IndexArray halo_fw_width(offset_max);
  halo_fw_width.SetNoLessThan(0);  
  IndexArray halo_bw_width(offset_min);
  halo_bw_width = halo_bw_width * -1;
  halo_bw_width.SetNoLessThan(0);
  
  //for (int i = g->num_dims_ - 1; i >= 0; --i) {
  for (int i = g->num_dims_ - 2; i >= 0; --i) {  
    LOG_VERBOSE() << "Exchanging dimension " << i << " data\n";
    ExchangeBoundariesStage1(g, i, halo_fw_width[i], halo_bw_width[i],
                             diagonal, periodic);
  }
  gmc->SetCUDAStream(0);
  return NULL;
}

template <class GridType>
GridMPI *GridSpaceMPICUDA<GridType>::LoadNeighborStage2(
    GridType *g,
    const IndexArray &offset_min,
    const IndexArray &offset_max,
    bool diagonal,  bool reuse, bool periodic,
    cudaStream_t cuda_stream) {
  
  // set the stream of buffer by the stream parameter
  GridType *gmc = static_cast<GridType*>(g);
  gmc->SetCUDAStream(cuda_stream);

  IndexArray halo_fw_width(offset_max);
  halo_fw_width.SetNoLessThan(0);  
  IndexArray halo_bw_width(offset_min);
  halo_bw_width = halo_bw_width * -1;
  halo_bw_width.SetNoLessThan(0);

  // Does not perform staging for the continuous dimension. 
  int i = g->num_dims_ - 1;
  ExchangeBoundaries(g, i, halo_fw_width[i],
                     halo_bw_width[i], diagonal, periodic);

  // For the non-continuous dimensions, finish the exchange steps 
  for (int i = g->num_dims_ - 2; i >= 0; --i) {
    LOG_VERBOSE() << "Exchanging dimension " << i << " data\n";
    ExchangeBoundariesStage2(g, i, halo_fw_width[i],
                             halo_bw_width[i], diagonal, periodic);
  }
  gmc->SetCUDAStream(0);
  return NULL;
}
#endif // NEIGHBOR_EXCHANGE_MULTI_STAGE

template <class GridType>
std::ostream& GridSpaceMPICUDA<GridType>::PrintLoadNeighborProf(std::ostream &os) const {
  StringJoin sj("\n");
  FOREACH (it, this->load_neighbor_prof_.begin(), this->load_neighbor_prof_.end()) {
    int grid_id = it->first;
    DataCopyProfile *profs = it->second;
    StringJoin sj_grid;
    for (int i = 0; i < this->num_dims_*2; i+=2) {
      sj_grid << "upw: " << profs[i] << ", dwn: " << profs[i+1];
    }
    sj << grid_id << ": " << sj_grid.str();
  }
  return os << sj.str() << "\n";
}

template <class GridType>
int GridSpaceMPICUDA<GridType>::ReduceGrid(void *out, PSReduceOp op,
                                           GridType *g) {
  void *p = malloc(g->elm_size());
  if (g->Reduce(op, p) == 0) {
    switch (g->type()) {
      case PS_FLOAT:
        *(float*)p = GetReductionDefaultValue<float>(op);
        break;
      case PS_DOUBLE:
        *(double*)p = GetReductionDefaultValue<float>(op);
        break;
      case PS_INT:
        *(int*)p = GetReductionDefaultValue<int>(op);
        break;
      case PS_LONG:
        *(long*)p = GetReductionDefaultValue<long>(op);
        break;
      default:
        PSAbort(1);
    }
  }
  MPI_Datatype type = GetMPIDataType(g->type());
  MPI_Op mpi_op = GetMPIOp(op);
  PS_MPI_Reduce(p, out, 1, type, mpi_op, 0, this->comm_);
  free(p);
  return g->num_elms();
}
  
} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_GRID_SPACE_MPI_CUDA_H_ */

