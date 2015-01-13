// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_GRID_SPACE_MPI_TEMPLATE_H_
#define PHYSIS_RUNTIME_GRID_SPACE_MPI_TEMPLATE_H_

#define __STDC_LIMIT_MACROS
#include "mpi.h"

#include "runtime/runtime_common.h"
#include "runtime/grid.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_wrapper.h"
#include "runtime/grid_mpi.h"
#include "runtime/timing.h"
#include "runtime/ipc.h"

#include <utility>

namespace physis {
namespace runtime {

struct FetchInfo {
  IntArray peer_index;
  IndexArray peer_offset;  
  IndexArray peer_size;
};

enum GRID_REQUEST_KIND {INVALID, DONE, FETCH_REQUEST, FETCH_REPLY};

struct GridRequest {
  int my_rank;
  GRID_REQUEST_KIND kind;
  GridRequest(): my_rank(-1), kind(INVALID) {}
  GridRequest(int rank, GRID_REQUEST_KIND k): my_rank(rank), kind(k) {}
};

void SendGridRequest(int my_rank, int peer_rank, MPI_Comm comm,
                     GRID_REQUEST_KIND kind);
GridRequest RecvGridRequest(MPI_Comm comm);


inline void ConvertStencilOffsetToStencilWidth(const IndexArray &halo_bw,
                                               const IndexArray &halo_fw,
                                               Width2 &width,
                                               int rank,
                                               const IndexArray &local_size,
                                               const IndexArray &proc_size) {

  IndexArray tb = halo_bw * -1;
  IndexArray tf = halo_fw;
  for (int i = 0; i < rank; ++i) {
    if (local_size[i] == 0 || proc_size[i] == 1) {
      tb[i] = 0;
      tf[i] = 0;
    }
    tb[i] = std::max(tb[i], 0);
    tf[i] = std::max(tf[i], 0);
  }
  width.bw = tb;  
  width.fw = tf;
  return;
}

//class GridMPI;

template <class GT>
class GridSpaceMPI: public GridSpace {
 public:
  typedef GT GridType;
  GridSpaceMPI(int num_dims, const IndexArray &global_size,
               int proc_num_dims, const IntArray &proc_size,
               InterProcComm &ipc);
  virtual ~GridSpaceMPI();

  static void Partition(int num_dims, int num_procs,
                        const IndexArray &size, 
                        const IntArray &num_partitions,
                        PSIndex **partitions, PSIndex **offsets,
                        std::vector<IntArray> &proc_indices,
                        IndexArray &min_partition);

  virtual GT *CreateGrid(PSType type, int elm_size,
                         int num_dims,
                         const IndexArray &size,
                         const IndexArray &global_offset,
                         const IndexArray &stencil_offset_min,
                         const IndexArray &stencil_offset_max,
                         int attr);
  
  virtual GT *CreateGrid(const __PSGridTypeInfo *type_info,
                         int num_dims,
                         const IndexArray &size,
                         const IndexArray &global_offset,
                         const IndexArray &stencil_offset_min,
                         const IndexArray &stencil_offset_max,
                         const int *stencil_offset_min_member,
                         const int *stencil_offset_max_member,
                         int attr);

  //! Exchange halo on one dimension of a grid asynchronously.
  /*!
    Completion is not guaranteed. Used from the synchronous
    version--ExchangeBoundaries. 
    
    \param grid Grid to exchange
    \param dim Halo dimension
    \param halo_width Halo width
    \param diagonal True if diagonal points are accessed.
    \param periodic True if periodic access is used.
    \param requests MPI request vector to poll completion
   */
  virtual void ExchangeBoundariesAsync(
      GT *grid, int member, int dim,
      const Width2 &halo_width,
      bool diagonal, bool periodic,
      std::vector<MPI_Request> &requests) const;
  

  //! Exchange halo on one dimension of a grid.
  /*!
    \param grid Grid to exchange
    \param dim Halo dimension
    \param halo_width Halo width
    \param diagonal True if diagonal points are accessed.
    \param periodic True if periodic access is used.
   */
  virtual void ExchangeBoundaries(GT *grid, int member, int dim,
                                  const Width2 &halo_width,
                                  bool diagonal,
                                  bool periodic) const;

  //! Exchange all boundaries of a grid.
  /*!
    \param grid Grid to exchange
    \param halo_width Halo width.
    \param diagonal True if diagonal points are accessed.
    \param periodic True if periodic access is used.
    \param reuse True if reuse is enabled.
   */
  virtual void ExchangeBoundaries(GT *grid, int member,
                                  const Width2 &halo_width,
                                  bool diagonal,
                                  bool periodic,
                                  bool reuse=false) const;

  virtual GT *LoadNeighbor(GT *g,
                           const IndexArray &offset_min,
                           const IndexArray &offset_max,
                           bool diagonal,
                           bool reuse,
                           bool periodic);
  virtual GT *LoadNeighbor(GT *g, int member,
                           const IndexArray &offset_min,
                           const IndexArray &offset_max,
                           bool diagonal,
                           bool reuse,
                           bool periodic);
  
  

  virtual int FindOwnerProcess(GT *g, const IndexArray &index);
  
  virtual std::ostream &Print(std::ostream &os) const;
  
  int num_dims() const { return num_dims_; }
  const IndexArray &global_size() const { return global_size_; }
  const IntArray &proc_size() const { return proc_size_; }
  int num_procs() const { return num_procs_; }
  const IntArray &my_idx() const { return my_idx_; }
  PSIndex **partitions() const { return partitions_; }
  PSIndex **offsets() const { return offsets_; }
  int my_rank() const { return my_rank_; }
  const IndexArray &my_size() { return my_size_; }
  const IndexArray &my_offset() { return my_offset_; }  
  const std::vector<IntArray> &proc_indices() const { return proc_indices_; }
  int GetProcessRank(const IntArray &proc_index) const;
  //! Reduce a grid with binary operator op.
  /*
   * \param out The destination scalar buffer.
   * \param op The binary operator to apply.   
   * \param g The grid to reduce.
   * \return The number of reduced elements.
   */
  virtual int ReduceGrid(void *out, PSReduceOp op, GT *g);

  //virtual void Save() const;
  //virtual void Restore();

 protected:
  int num_dims_;
  IndexArray global_size_;
  int proc_num_dims_;
  IntArray proc_size_;
  InterProcComm &ipc_;  
  int my_rank_;
  int num_procs_;
  IndexArray my_size_;  
  IndexArray my_offset_;
  PSIndex **partitions_;  
  PSIndex **offsets_;
  IndexArray min_partition_;
  IntArray my_idx_; //! Process index
  IntArray fw_neighbors_;
  IntArray bw_neighbors_;
  //! Indices for all processes; proc_indices_[my_rank] == my_idx_
  std::vector<IntArray> proc_indices_;
  MPI_Comm comm_;
  // Timing profile for halo exchange
  std::map<int, DataCopyProfile*> load_neighbor_prof_;
  
  // To support fetching of subgrids from distributed processes. Used
  // by LoadSubgrid
  virtual void CollectPerProcSubgridInfo(
      const GT *g,
      const IndexArray &grid_offset,
      const IndexArray &grid_size,
      std::vector<FetchInfo> &finfo_holder) const;
  virtual bool SendFetchRequest(FetchInfo &finfo) const;
  virtual void HandleFetchRequest(GridRequest &req, GT *g);
  virtual void HandleFetchReply(GridRequest &req, GT *g,
                                std::map<int, FetchInfo> &fetch_map,  GT *sg);
  
  void *buf;
  size_t cur_buf_size;

  //! Calculate paritioning of a grid into sub grids.
  virtual void PartitionGrid(int num_dims, const IndexArray &size,
                             const IndexArray &global_offset,
                             IndexArray &local_offset, IndexArray &local_size) {
    for (int i = 0; i < num_dims; ++i) {
      local_offset[i] =
          std::max(my_offset_[i] - global_offset[i], (PSIndex)0);
      PSIndex first = std::max(my_offset_[i], global_offset[i]);
      PSIndex last = std::min(global_offset[i] + size[i],
                              my_offset_[i] + my_size_[i]);
      local_size[i] = std::max(last - first, (PSIndex)0);
    }
    return;
  }
  
}; // class GridSpaceMPI

template <class GridType>
GridSpaceMPI<GridType>::GridSpaceMPI(
    int num_dims, const IndexArray &global_size,
    int proc_num_dims, const IntArray &proc_size,
    InterProcComm &ipc):
    num_dims_(num_dims), global_size_(global_size),
    proc_num_dims_(proc_num_dims), proc_size_(proc_size),
    ipc_(ipc), my_rank_(ipc.GetRank()), buf(NULL), cur_buf_size(0) {
  assert(num_dims_ == proc_num_dims_);
  
  num_procs_ = proc_size_.accumulate(proc_num_dims_); // For example 6

  partitions_ = new PSIndex*[num_dims_];
  offsets_ = new PSIndex*[num_dims_];
  
  Partition(num_dims_, num_procs_, global_size_, proc_size_,
            partitions_, offsets_, proc_indices_, min_partition_);

  my_idx_ = proc_indices_[my_rank_]; // Usually {0,0, my_rank_}
  
  for (int i = 0; i < num_dims_; ++i) {
    my_offset_[i] = offsets_[i][my_idx_[i]]; // For example {0,0,31}
    my_size_[i] = partitions_[i][my_idx_[i]]; // For example {0,0,11}
  }

  comm_ = MPI_COMM_WORLD;
  for (int i = 0; i < num_dims_; ++i) {
    IntArray neighbor = my_idx_; // Usually {0, 0, my_rank_}
    neighbor[i] += 1;
    // wrap around for periodic boundary access
    neighbor[i] %= proc_size_[i];
    fw_neighbors_[i] = GetProcessRank(neighbor);
    neighbor[i] = my_idx_[i] - 1;
    // wrap around for periodic boundary access     
    if (neighbor[i] < 0) {
      neighbor[i] = (neighbor[i] + proc_size_[i]) % proc_size_[i];
    }
    bw_neighbors_[i] = GetProcessRank(neighbor);
  }
}

template <class GridType>
GridSpaceMPI<GridType>::~GridSpaceMPI() {
  FREE(buf);
  FOREACH (it, load_neighbor_prof_.begin(), load_neighbor_prof_.end()) {
    delete[] it->second;
  }
}

template <class GridType>
GridType *GridSpaceMPI<GridType>::CreateGrid(
    PSType type, int elm_size,
    int num_dims,
    const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &stencil_offset_min,
    const IndexArray &stencil_offset_max,
    int attr) {
  __PSGridTypeInfo info = {type, elm_size, 0, NULL};
  return CreateGrid(&info, num_dims, size, global_offset,
                    stencil_offset_min, stencil_offset_max, NULL, NULL, attr);
}

template <class GridType>
GridType *GridSpaceMPI<GridType>::CreateGrid(
    const __PSGridTypeInfo *type_info,
    int num_dims,
    const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &stencil_offset_min,
    const IndexArray &stencil_offset_max,
    const int *stencil_offset_min_member,
    const int *stencil_offset_max_member,
    int attr) {

  IndexArray grid_size = size;
  IndexArray grid_global_offset = global_offset;
  if (num_dims_ != num_dims) {
    // The grid space has smaller dimension.
    LOG_ERROR() << "Cannot create " << num_dims << "-d grid in "
                << num_dims_ << "-d space.\n";
    PSAbort(1);
  }

  IndexArray local_offset, local_size;  
  PartitionGrid(num_dims, grid_size, grid_global_offset,
                local_offset, local_size);

  LOG_DEBUG() << "local_size: " << local_size << "\n";
  LOG_DEBUG() << "stencil_offset_min: "
              << stencil_offset_min << "\n";
  LOG_DEBUG() << "stencil_offset_max: "
              << stencil_offset_max << "\n";

  Width2 halo;
  ConvertStencilOffsetToStencilWidth(
      stencil_offset_min, stencil_offset_max, halo,
      num_dims_, local_size, proc_size());
  LOG_DEBUG() << "halo.fw: " << halo.fw << "\n";
  LOG_DEBUG() << "halo.bw: " << halo.bw << "\n";

  Width2 *halo_member = NULL;
  if (type_info->num_members > 0) {
    halo_member = new Width2[type_info->num_members];
    for (int i = 0; i < type_info->num_members; ++i) {
      IndexArray halo_bw(stencil_offset_min_member + i*PS_MAX_DIM);
      IndexArray halo_fw(stencil_offset_max_member + i*PS_MAX_DIM);
      ConvertStencilOffsetToStencilWidth(
          halo_bw, halo_fw, halo_member[i], num_dims_, local_size, proc_size());
    }
  }
  
  GridType *g = GridType::Create(
      type_info, num_dims, grid_size,
      grid_global_offset, local_offset, local_size,
      halo, halo_member, attr);
  LOG_DEBUG() << "grid created\n";
  RegisterGrid(g);
  LOG_DEBUG() << "grid registered\n";
  DataCopyProfile *profs =
      new DataCopyProfile[num_dims*2];
  load_neighbor_prof_.insert(std::make_pair(g->id(), profs));
  return g;
}

template <class GridType>
std::ostream &GridSpaceMPI<GridType>::Print(
    std::ostream &os) const {
  os << "GridSpaceMPI {"
     << "#grid dims: " << num_dims_
     << ", grid size: " << global_size_
     << ", #proc dims: " << proc_num_dims_
     << ", proc size: " << proc_size_
     << ", my rank: " << my_rank_
     << ", #procs: " << num_procs_
     << ", my idx: " << my_idx_
     << ", size: " << my_size_
     << ", offset: " << my_offset_
     << ", #grids: " << grids_.size()
     << "}";
  return os;
}

template <class GridType>
void GridSpaceMPI<GridType>::ExchangeBoundariesAsync(
    GridType *grid, int member, int dim, const Width2 &halo_width,
    bool diagonal, bool periodic,
    std::vector<MPI_Request> &requests) const {
  
  if (grid->empty()) return;

  int fw_peer = fw_neighbors_[dim];
  int bw_peer = bw_neighbors_[dim];
  int tag = 0;
  const unsigned halo_fw_width = halo_width.fw[dim];
  const unsigned halo_bw_width = halo_width.bw[dim];  
  size_t fw_size = grid->CalcHaloSize(dim, halo_fw_width, diagonal)
      * grid->elm_size();
  size_t bw_size = grid->CalcHaloSize(dim, halo_bw_width, diagonal)
      * grid->elm_size();

  //LOG_DEBUG() << "Periodic?: " << periodic << "\n";

  /*
    Send and receive ordering must match. First get the halo for the
    forward access, and then the halo for the backward access.
   */

  // Note: If no decomposition is done for this dimension, periodic
  // access is implemented without halo buffer, but with wrap-around
  // offsets. 
  if (halo_fw_width > 0 &&
      (grid->local_offset()[dim] + grid->local_size()[dim]
       < grid->size_[dim] ||
       (periodic && proc_size_[dim] > 1))) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Receiving halo of " << fw_size
                << " bytes for fw access from " << fw_peer << "\n";
    MPI_Request req;
    CHECK_MPI(MPI_Irecv(
        grid->GetHaloPeerBuf(dim, true, halo_fw_width),
        fw_size, MPI_BYTE, fw_peer, tag, comm_, &req));
    requests.push_back(req);
  }

  if (halo_bw_width > 0 &&
      (grid->local_offset()[dim] > 0 ||
       (periodic && proc_size_[dim] > 1))) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Receiving halo of " << bw_size
                << " bytes for bw access from " << bw_peer << "\n";
    MPI_Request req;
    CHECK_MPI(MPI_Irecv(
        grid->GetHaloPeerBuf(dim, false, halo_bw_width),
        bw_size, MPI_BYTE, bw_peer, tag, comm_, &req));
    requests.push_back(req);
  }

  // Sends out the halo for forward access
  if (halo_fw_width > 0 &&
      (grid->local_offset()[dim] > 0 ||
       (periodic && proc_size_[dim] > 1))) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Sending halo of " << fw_size << " bytes"
                << " for fw access to " << bw_peer << "\n";
    LOG_DEBUG() << "grid: " << grid << "\n";
    grid->CopyoutHalo(dim, halo_width, true, diagonal);
    LOG_DEBUG() << "grid2: " << grid->data() << "\n";
    LOG_DEBUG() << "dim: " << dim << "\n";        
    MPI_Request req;
    LOG_DEBUG() << "send buf: " <<
        (void*)(grid->halo_self_fw_[dim])
                << "\n";        
    CHECK_MPI(PS_MPI_Isend(grid->halo_self_fw_[dim], fw_size, MPI_BYTE,
                           bw_peer, tag, comm_, &req));
  }

   // Sends out the halo for backward access
  if (halo_bw_width > 0 &&
      (grid->local_offset()[dim] + grid->local_size()[dim]
       < grid->size_[dim] ||
       (periodic && proc_size_[dim] > 1))) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Sending halo of " << bw_size << " bytes"
                << " for bw access to " << fw_peer << "\n";
    grid->CopyoutHalo(dim, halo_width, false, diagonal);
    MPI_Request req;
    CHECK_MPI(PS_MPI_Isend(grid->halo_self_bw_[dim], bw_size, MPI_BYTE,
                           fw_peer, tag, comm_, &req));
  }

  return;
}

template <class GridType>
void GridSpaceMPI<GridType>::ExchangeBoundaries(
    GridType *grid, int member, int dim, const Width2 &halo_width,
    bool diagonal, bool periodic) const {
  std::vector<MPI_Request> requests;
  ExchangeBoundariesAsync(grid, member, dim, halo_width, diagonal,
                          periodic, requests);
  FOREACH (it, requests.begin(), requests.end()) {
    MPI_Request *req = &(*it);
    CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));
    grid->CopyinHalo(dim, halo_width, false, diagonal);
    grid->CopyinHalo(dim, halo_width, true, diagonal);
  }
  return;
}

#if 0
template <class GridType>
void GridSpaceMPI<GridType>::ExchangeBoundariesAsync(
    int grid_id,  const UnsignedArray &halo_fw_width,
    const UnsignedArray &halo_bw_width, bool diagonal,
    std::vector<MPI_Request> &requests) const {

  LOG_ERROR() << "This does not correctly copy halo regions. "
              << "Halo exchange of previous dimension must be completed"
              << " before sending out next halo region.";
  PSAbort(1);
  
  
  GridType *g = static_cast<GridType*>(FindGrid(grid_id));
  
  for (int i = g->num_dims_ - 1; i >= 0; --i) {
    LOG_DEBUG() << "Exchanging dimension " << i << " data\n";
    ExchangeBoundariesAsync(*g, i, halo_fw_width[i],
                            halo_bw_width[i], diagonal,
                            periodic,requests);
  }
  return;
}
#endif

// REFACTORING: reuse is used?
template <class GridType>
void GridSpaceMPI<GridType>::ExchangeBoundaries(
    GridType *g, int member, const Width2 &halo_width,
    bool diagonal, bool periodic, 
    bool reuse) const {
  LOG_DEBUG() << "GridSpaceMPI::ExchangeBoundaries\n";

  //GridType *g = static_cast<GridType*>(FindGrid(grid_id));
  for (int i = g->num_dims_ - 1; i >= 0; --i) {
    LOG_VERBOSE() << "Exchanging dimension " << i << " data\n";
    ExchangeBoundaries(g, member, i, halo_width, diagonal, periodic);
  }
  return;
}

inline
void SendGridRequest(int my_rank, int peer_rank,
                     MPI_Comm comm,
                     GRID_REQUEST_KIND kind) {
  LOG_DEBUG() << "Sending request " << kind << " to " << peer_rank << "\n";
  GridRequest req(my_rank, kind);
  MPI_Request mpi_req;
  CHECK_MPI(PS_MPI_Isend(&req, sizeof(GridRequest), MPI_BYTE,
                      peer_rank, 0, comm, &mpi_req));
}

inline
GridRequest RecvGridRequest(MPI_Comm comm) {
  GridRequest req;
  CHECK_MPI(MPI_Recv(&req, sizeof(GridRequest), MPI_BYTE, MPI_ANY_SOURCE, 0,
                     comm, MPI_STATUS_IGNORE));
  return req;
}


static inline void *ensure_buffer_capacity(void *buf, size_t cur_size,
                                           size_t required_size) {
  if (cur_size < required_size) {
    FREE(buf);
    cur_size = required_size;
    buf = malloc(cur_size);
  }
  return buf;
}

template <class GridType>
void GridSpaceMPI<GridType>::CollectPerProcSubgridInfo(
    const GridType *g,  const IndexArray &grid_offset,
    const IndexArray &grid_size,
    std::vector<FetchInfo> &finfo_holder) const {
  LOG_DEBUG() << "Collecting per-process subgrid info\n";
  IndexArray grid_lim = grid_offset + grid_size;
  std::vector<FetchInfo> *fetch_info = new std::vector<FetchInfo>;
  std::vector<FetchInfo> *fetch_info_next = new std::vector<FetchInfo>;
  FetchInfo dummy;
  fetch_info->push_back(dummy);
  for (int d = 0; d < num_dims_; ++d) {
    PSIndex x = grid_offset[d] + g->global_offset()[d];
    for (int pidx = 0; pidx < proc_size_[d] && x < grid_lim[d]; ++pidx) {
      if (x < offsets_[d][pidx] + partitions_[d][pidx]) {
        LOG_VERBOSE_MPI() << "inclusion: " << pidx
                          << ", dim: " << d << "\n";
        FOREACH (it, fetch_info->begin(), fetch_info->end()) {
          FetchInfo info = *it;
          // peer process index
          info.peer_index[d] = pidx;
          // peer offset
          info.peer_offset[d] = x - g->global_offset_[d];
          // peer size
          info.peer_size[d] =
              std::min(offsets_[d][pidx] + partitions_[d][pidx] - x,
                       grid_lim[d] - x);
          fetch_info_next->push_back(info);
        }
        x = offsets_[d][pidx] + partitions_[d][pidx];
      }
    }
    std::swap(fetch_info, fetch_info_next);
    fetch_info_next->clear();
#if defined(PS_VERBOSE)
    FOREACH (it, fetch_info->begin(), fetch_info->end()) {
      const FetchInfo &finfo = *it;
      LOG_DEBUG() << "[" << my_rank_ << "] finfo peer index: "
                  << finfo.peer_index << "\n";
    }
#endif    
  }
  finfo_holder = *fetch_info;
  delete fetch_info;
  delete fetch_info_next;
  return;  
}

// Returns true if sent out
template <class GridType>
bool GridSpaceMPI<GridType>::SendFetchRequest(FetchInfo &finfo) const {
  int peer_rank = GetProcessRank(finfo.peer_index);
  size_t size = finfo.peer_size.accumulate(num_dims_);
  if (size == 0) return false; 
  SendGridRequest(my_rank_, peer_rank, comm_, FETCH_REQUEST);
  MPI_Request mpi_req;  
  CHECK_MPI(PS_MPI_Isend(&finfo, sizeof(FetchInfo), MPI_BYTE,
                      peer_rank, 0, comm_, &mpi_req));
  return true;
}

template <class GridType>
void GridSpaceMPI<GridType>::HandleFetchRequest(GridRequest &req, GridType *g) {
  LOG_DEBUG() << "HandleFetchRequest\n";
  FetchInfo finfo;
  int nd = num_dims_;
  CHECK_MPI(MPI_Recv(&finfo, sizeof(FetchInfo), MPI_BYTE,
                     req.my_rank, 0, comm_, MPI_STATUS_IGNORE));
  size_t bytes = finfo.peer_size.accumulate(nd) * g->elm_size();
  buf = ensure_buffer_capacity(buf, cur_buf_size, bytes);
  CopyoutSubgrid(g->elm_size(), nd, g->data(), g->local_size(),
                 buf, finfo.peer_offset - g->local_offset(),
                 finfo.peer_size);
  SendGridRequest(my_rank_, req.my_rank, comm_, FETCH_REPLY);
  MPI_Request mr;
  CHECK_MPI(PS_MPI_Isend(buf, bytes, MPI_BYTE, req.my_rank, 0, comm_, &mr));
  return;
}

template <class GridType>
void GridSpaceMPI<GridType>::HandleFetchReply(
    GridRequest &req, GridType *g, std::map<int, FetchInfo> &fetch_map,
    GridType *sg) {
  LOG_DEBUG() << "HandleFetchReply\n";
  const FetchInfo &finfo = fetch_map[req.my_rank];
  PSAssert(GetProcessRank(finfo.peer_index) == req.my_rank);
  size_t bytes = finfo.peer_size.accumulate(num_dims_) * g->elm_size();
  LOG_DEBUG() << "Fetch reply data size: " << bytes << "\n";
  buf = ensure_buffer_capacity(buf, cur_buf_size, bytes);
  CHECK_MPI(MPI_Recv(buf, bytes, MPI_BYTE, req.my_rank, 0,
                     comm_, MPI_STATUS_IGNORE));
  LOG_DEBUG() << "Fetch reply received\n";
  CopyinSubgrid(sg->elm_size(), num_dims_, sg->data(),
                sg->local_size(), buf,
                finfo.peer_offset - sg->local_offset(), finfo.peer_size);
  return;;
}

template <class GridType>
int GridSpaceMPI<GridType>::GetProcessRank(const IntArray &proc_index) const {
  int rank = 0;
  int offset = 1;
  for (int i = 0; i < num_dims(); ++i) {
    rank += proc_index[i] * offset;
    offset *= proc_size_[i];
  }
  return rank;
}


template <class GridType>
GridType *GridSpaceMPI<GridType>::LoadNeighbor(
    GridType *g, const IndexArray &offset_min,
    const IndexArray &offset_max,
    bool diagonal, bool reuse, bool periodic) {
  for (int i = 0; i < g->num_members(); ++i) {
    LoadNeighbor(g, i, offset_min, offset_max,
                 diagonal, reuse, periodic);
  }
  return NULL;
}

template <class GridType>
GridType *GridSpaceMPI<GridType>::LoadNeighbor(
    GridType *g, int member, const IndexArray &offset_min,
    const IndexArray &offset_max,
    bool diagonal, bool reuse, bool periodic) {
  Width2 hw;
  for (int i = 0; i < PS_MAX_DIM; ++i) {
    hw.bw[i] = (offset_min[i] <= 0) ? (unsigned)(abs(offset_min[i])) : 0;
    hw.fw[i] = (offset_max[i] >= 0) ? (unsigned)(offset_max[i]) : 0;
  }
  ExchangeBoundaries(g, member, hw, diagonal, periodic, reuse);
  return NULL;
}


template <class GridType>
int GridSpaceMPI<GridType>::FindOwnerProcess(GridType *g, const IndexArray &index) {
  std::vector<FetchInfo> fetch_requests;
  IndexArray one;
  one.Set(1);
  CollectPerProcSubgridInfo(g, index, one, fetch_requests);
  PSAssert(fetch_requests.size() == 1);
  return GetProcessRank(fetch_requests[0].peer_index);
}

template <class GridType>
int GridSpaceMPI<GridType>::ReduceGrid(void *out, PSReduceOp op,
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
        LOG_ERROR() << "Unsupported type\n";
        PSAbort(1);
    }
  }
  MPI_Datatype type = GetMPIDataType(g->type());
  MPI_Op mpi_op = GetMPIOp(op);
  PS_MPI_Reduce(p, out, 1, type, mpi_op, 0, comm_);
  free(p);
  return g->num_elms();
}

template <class GridType>
void GridSpaceMPI<GridType>::Partition(
    int num_dims, int num_procs,
    const IndexArray &size, 
    const IntArray &num_partitions,
    PSIndex **partitions, PSIndex **offsets,
    std::vector<IntArray> &proc_indices,
    IndexArray &min_partition)  {
  for (int i = 0; i < num_procs; ++i) {
    IntArray pidx;
    for (int j = 0, t = i; j < num_dims; ++j) {
      pidx[j] = t % num_partitions[j];
      t /= num_partitions[j];
    }
    proc_indices.push_back(pidx);
  }

  min_partition.Set(PSINDEX_MAX);
  
  for (int i = 0; i < num_dims; i++) {
    partitions[i] = new PSIndex[num_partitions[i]];
    offsets[i] = new PSIndex[num_partitions[i]];    
    int offset = 0;
    for (int j = 0; j < num_partitions[i]; ++j) {
      int rem = size[i] % num_partitions[i]; // <0, 0, 4>
      partitions[i][j] = size[i] / num_partitions[i]; // {{64}, {64}, {10,10,10,10,10,10}}
      if (num_partitions[i] - j <= rem) {
        ++partitions[i][j]; // {{64}, {64}, {10,10,11,11,11,11}}
      }
      min_partition[i] = std::min(min_partition[i], partitions[i][j]); // {64,64,10}
      offsets[i][j] = offset;
      offset += partitions[i][j]; // {{0}, {0}, {0,10,20,31,42,53}}
    }
  }
  
  return;
}

} // namespace runtime
} // namespace physis

template <class GridType>
std::ostream &operator<<(std::ostream &os,
                         const physis::runtime::GridSpaceMPI<GridType> &gs) {
  return gs.Print(os);
}

#endif /* PHYSIS_RUNTIME_GRID_SPACE_MPI_H_ */
