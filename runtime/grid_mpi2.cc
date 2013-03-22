// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/grid_mpi2.h"

#include <limits.h>

#include "runtime/grid_util.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_wrapper.h"

#include <algorithm>

using namespace std;

namespace physis {
namespace runtime {

GridMPI2::GridMPI2(PSType type, int elm_size, int num_dims,
                   const IndexArray &size,
                   const IndexArray &global_offset,
                   const IndexArray &local_offset,
                   const IndexArray &local_size,
                   const Width2 &halo,
                   int attr):
    GridMPI(type, elm_size, num_dims, size, false, global_offset,
            local_offset, local_size, attr) {

  local_real_size_ = local_size_;
  local_real_offset_ = local_offset_;
  for (int i = 0; i < num_dims_; ++i) {
    local_real_size_[i] += halo.fw[i] + halo.bw[i];
    local_real_offset_[i] -= halo.bw[i];
  }
  
  empty_ = local_size_.accumulate(num_dims_) == 0;
  if (empty_) return;

  halo_ = halo;
}

GridMPI2::~GridMPI2() {
  DeleteBuffers();
}

GridMPI2 *GridMPI2::Create(
    PSType type, int elm_size,
    int num_dims, const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &local_offset,
    const IndexArray &local_size,
    const Width2 &halo,
    int attr) {
  GridMPI2 *g = new GridMPI2(
      type, elm_size,
      num_dims, size,
      global_offset,
      local_offset,
      local_size,
      halo,
      attr);
  g->InitBuffers();
  return g;
}

std::ostream &GridMPI2::Print(std::ostream &os) const {
  os << "GridMPI2 {"
     << "elm_size: " << elm_size_
     << ", size: " << size_
     << ", global offset: " << global_offset_
     << ", local offset: " << local_offset_
     << ", local size: " << local_size_
     << ", local real size: " << local_real_size_      
     << "}";
  return os;
}

void GridMPI2::InitBuffers() {
  if (empty_) return;
  data_buffer_[0] = new BufferHost(num_dims_, elm_size_);
  data_buffer_[0]->Allocate(local_real_size_);
  data_buffer_[1] = NULL;
  data_[0] = (char*)data_buffer_[0]->Get();
  LOG_DEBUG() << "buffer addr: " << (void*)(data_[0]) << "\n";
  data_[1] = NULL;
  InitHaloBuffers();
}

void GridMPI2::InitHaloBuffers() {
  // Note that the halo for the last dimension is continuously located
  // in memory, so no separate buffer is necessary.
  for (int i = 0; i < num_dims_ - 1; ++i) {
    halo_self_fw_[i] = halo_self_bw_[i] = NULL;
    halo_peer_fw_[i] = halo_peer_bw_[i] = NULL;
    if (halo_.fw[i]) {
      halo_self_fw_[i] =
          (char*)malloc(CalcHaloSize(i, halo_.fw[i]) * elm_size_);
      assert(halo_self_fw_[i]);
      halo_peer_fw_[i] =
          (char*)malloc(CalcHaloSize(i, halo_.fw[i]) * elm_size_);
      assert(halo_peer_fw_[i]);      
    } 
    if (halo_.bw[i]) {
      halo_self_bw_[i] =
          (char*)malloc(CalcHaloSize(i, halo_.bw[i]) * elm_size_);
      assert(halo_self_bw_[i]);
      halo_peer_bw_[i] =
          (char*)malloc(CalcHaloSize(i, halo_.bw[i]) * elm_size_);
      assert(halo_peer_bw_[i]);      
    } 
  }
  
}

void GridMPI2::DeleteBuffers() {
  if (empty_) return;
  Grid::DeleteBuffers();
}

void GridMPI2::DeleteHaloBuffers() {
  for (int i = 0; i < num_dims_ - 1; ++i) {
    PS_XFREE(halo_self_fw_[i]);
    PS_XFREE(halo_self_bw_[i]);
    PS_XFREE(halo_peer_fw_[i]);
    PS_XFREE(halo_peer_bw_[i]);
  }
}

void *GridMPI2::GetAddress(const IndexArray &indices_param) {
  IndexArray indices = indices_param;
  indices -= local_real_offset_;
  return (void*)(_data() +
                 GridCalcOffset3D(indices, local_real_size_)
                 * elm_size());
}

PSIndex GridMPI2::CalcOffset(const IndexArray &indices_param) {
  IndexArray indices = indices_param;
  indices -= local_real_offset_;
  return GridCalcOffset3D(indices, local_real_size_);
}

PSIndex GridMPI2::CalcOffsetPeriodic(const IndexArray &indices_param) {
  IndexArray indices = indices_param;
  for (int i = 0; i < num_dims_; ++i) {
    // No halo if no domain decomposition is done for a
    // dimension. Periodic access must be done by wrap around the offset
    if (local_size_[i] == size_[i]) {
      indices[i] = (indices[i] + size_[i]) % size_[i];
    } else {
      indices[i] -= local_real_offset_[i];
    }
  }
  return GridCalcOffset3D(indices, local_real_size_);
}

void GridMPI2::Copyout(void *dst) const {
  const void *src = buffer()->Get();
  if (HasHalo()) {
    IndexArray offset(halo().bw);
    CopyoutSubgrid(elm_size(), num_dims(),
                   src, local_real_size(),
                   dst, offset, local_size());
  } else {
    memcpy(dst, src, GetLocalBufferSize());
  }
  return;
}

void GridMPI2::Copyin(const void *src) {
  void *dst = buffer()->Get();
  if (HasHalo()) {
    CopyinSubgrid(elm_size(), num_dims(),
                  dst, local_real_size(),
                  src, halo().bw, local_size());
  } else {
    memcpy(dst, src, GetLocalBufferSize());
  }
  return;
}

GridSpaceMPI2::GridSpaceMPI2(int num_dims, const IndexArray &global_size,
                             int proc_num_dims, const IntArray &proc_size,
                             int my_rank):
    GridSpaceMPI(num_dims, global_size, proc_num_dims, proc_size,
                 my_rank) {
}

GridSpaceMPI2::~GridSpaceMPI2() {
}

GridMPI2 *GridSpaceMPI2::CreateGrid(PSType type, int elm_size, int num_dims,
                                    const IndexArray &size, 
                                    const IndexArray &global_offset,
                                    const IndexArray &stencil_offset_min,
                                    const IndexArray &stencil_offset_max,
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

  IndexArray halo_fw = stencil_offset_max;
  IndexArray halo_bw = stencil_offset_min;  
  halo_bw = halo_bw * -1;
  for (int i = 0; i < num_dims_; ++i) {
    if (local_size[i] == 0 || proc_size()[i] == 1) {
      halo_bw[i] = 0;
      halo_fw[i] = 0;
    }
    if (halo_bw[i] < 0) halo_bw[i] = 0;
    if (halo_fw[i] < 0) halo_fw[i] = 0;    
  }
  Width2 halo = {halo_bw, halo_fw};
  LOG_DEBUG() << "halo.fw: " << halo.fw << "\n";
  LOG_DEBUG() << "halo.bw: " << halo.bw << "\n";  
  
  GridMPI2 *g = GridMPI2::Create(
      type, elm_size, num_dims, grid_size,
      grid_global_offset, local_offset, local_size,
      halo, attr);
  LOG_DEBUG() << "grid created\n";
  RegisterGrid(g);
  return g;
}

size_t GridMPI2::CalcHaloSize(int dim, unsigned width) {
  IndexArray halo_size = local_real_size_;
  halo_size[dim] = width;
  return halo_size.accumulate(num_dims_);
}

// REFACTORING: halo_fw|bw-size_ are Used?
void GridMPI2::SetHaloSize(int dim, bool fw, unsigned width, bool diagonal) {
  PSAbort(1);
  IndexArray s = local_real_size_;
  s[dim] = width;  
  if (fw) {
    halo_fw_size_[dim] = s;
  } else {
    halo_bw_size_[dim] = s;    
  }
  return;
}

char *GridMPI2::GetHaloPeerBuf(int dim, bool fw, unsigned width) {
  if (dim == num_dims_ - 1) {
    IndexArray offset(0);
    if (fw) {
      offset[dim] = local_real_size_[dim] - halo_.fw[dim];
    } else {
      offset[dim] = halo_.fw[dim] - width;
    }
    return _data() + GridCalcOffset3D(offset, local_real_size_) * elm_size_;
  } else {
    if (fw) return halo_peer_fw_[dim];
    else  return halo_peer_bw_[dim];
  }
  
}

// Note: width is unsigned. 
void GridSpaceMPI2::ExchangeBoundariesAsync(
    GridMPI *grid, int dim, unsigned halo_fw_width, unsigned halo_bw_width,
    bool diagonal, bool periodic,
    std::vector<MPI_Request> &requests) const {
  
  if (grid->empty()) return;

  GridMPI2 *g2 = static_cast<GridMPI2*>(grid);
  int fw_peer = fw_neighbors_[dim];
  int bw_peer = bw_neighbors_[dim];
  int tag = 0;
  size_t fw_size = g2->CalcHaloSize(dim, halo_fw_width)
      * grid->elm_size_;
  size_t bw_size = g2->CalcHaloSize(dim, halo_bw_width)
      * grid->elm_size_;

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
        g2->GetHaloPeerBuf(dim, true, halo_fw_width),
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
        g2->GetHaloPeerBuf(dim, false, halo_bw_width),
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
    grid->CopyoutHalo(dim, halo_fw_width, true, diagonal);
    LOG_DEBUG() << "grid2: " << (void*)(grid->_data()) << "\n";
    LOG_DEBUG() << "dim: " << dim << "\n";        
    MPI_Request req;
    LOG_DEBUG() << "send buf: " <<
        (void*)(grid->_halo_self_fw()[dim])
                << "\n";        
    CHECK_MPI(PS_MPI_Isend(grid->_halo_self_fw()[dim], fw_size, MPI_BYTE,
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
    grid->CopyoutHalo(dim, halo_bw_width, false, diagonal);
    MPI_Request req;
    CHECK_MPI(PS_MPI_Isend(grid->_halo_self_bw()[dim], bw_size, MPI_BYTE,
                           fw_peer, tag, comm_, &req));
  }

  return;
}

template <class T>
int ReduceGridMPI3D(GridMPI2 *g, PSReduceOp op, T *out) {
  size_t nelms = g->local_size().accumulate(g->num_dims());
  if (nelms == 0) return 0;
  boost::function<T (T, T)> func = GetReducer<T>(op);
  T *d = (T *)g->_data();
  T v = GetReductionDefaultValue<T>(op);
  for (int k = 0; k < g->local_size()[2]; ++k) {
    for (int j = 0; j < g->local_size()[1]; ++j) {
      for (int i = 0; i < g->local_size()[0]; ++i) {
        intptr_t offset =
            GridCalcOffset3D(i + g->halo().bw[0], j + g->halo().bw[1],
                             k + g->halo().bw[2], g->local_real_size());
        v = func(v, d[offset]);
      }
    }
  }
  *out = v;
  return nelms;
}

int GridMPI2::Reduce(PSReduceOp op, void *out) {
  int rv = 0;
  PSAssert(num_dims_ == 3);
  switch (type_) {
    case PS_FLOAT:
      rv = ReduceGridMPI3D<float>(this, op, (float*)out);
      break;
    case PS_DOUBLE:
      rv = ReduceGridMPI3D<double>(this, op, (double*)out);
      break;
    default:
      PSAbort(1);
  }
  return rv;
}

void GridSpaceMPI2::ExchangeBoundaries(GridMPI *grid,
                                       int dim,
                                       unsigned halo_fw_width,
                                       unsigned halo_bw_width,
                                       bool diagonal,
                                       bool periodic) const {
  std::vector<MPI_Request> requests;
  ExchangeBoundariesAsync(grid, dim, halo_fw_width,
                          halo_bw_width, diagonal,
                          periodic, requests);
  GridMPI2 *g2 = static_cast<GridMPI2*>(grid);
  FOREACH (it, requests.begin(), requests.end()) {
    MPI_Request *req = &(*it);
    CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));
    g2->CopyinHalo(dim, halo_bw_width, false, diagonal);
    g2->CopyinHalo(dim, halo_fw_width, true, diagonal);
  }
  
  return;
}

GridMPI *GridSpaceMPI2::LoadNeighbor(GridMPI *g,
                                     const IndexArray &offset_min,
                                     const IndexArray &offset_max,
                                     bool diagonal,
                                     bool reuse,
                                     bool periodic) {
  UnsignedArray halo_fw_width, halo_bw_width;
  for (int i = 0; i < PS_MAX_DIM; ++i) {
    halo_bw_width[i] = (offset_min[i] <= 0) ? (unsigned)(abs(offset_min[i])) : 0;
    halo_fw_width[i] = (offset_max[i] >= 0) ? (unsigned)(offset_max[i]) : 0;
  }
  GridSpaceMPI::ExchangeBoundaries(g->id(), halo_fw_width,
                                   halo_bw_width, diagonal, periodic, reuse);
  return NULL;
}

// fw: copy in halo buffer received for forward access if true
void GridMPI2::CopyinHalo(int dim, unsigned width, bool fw, bool diagonal) {
  // The slowest changing dimension does not need actual copying
  // because it's directly copied into the grid buffer.
  if (dim == num_dims_ - 1) {
    return;
  }
  
  IndexArray halo_offset(0);
  if (fw) {
    halo_offset[dim] = local_real_size_[dim] - halo_.fw[dim];
  } else {
    halo_offset[dim] = halo_.bw[dim] - width;
  }
  
  char *halo_buf = fw ? halo_peer_fw_[dim] : halo_peer_bw_[dim];

  IndexArray halo_size = local_real_size_;
  halo_size[dim] = width;
  
  CopyinSubgrid(elm_size_, num_dims_, data_[0], local_real_size_,
                halo_buf, halo_offset, halo_size);
}

// fw: prepare buffer for sending halo for forward access if true
void GridMPI2::CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal) {
#if 0
  LOG_DEBUG() << "FW?: " << fw << ", width: " << width
              << ", local size: " << local_size_
              << ", halo fw: " << halo_.fw
              << ", halo bw: " << halo_.bw << "\n";
#endif

  IndexArray halo_offset(0);
  if (fw) {
    halo_offset[dim] = halo_.bw[dim];
  } else {
    halo_offset[dim] = local_real_size_[dim] - halo_.fw[dim] - width;
  }

  LOG_DEBUG() << "halo offset: "
              << halo_offset << "\n";
  
  char **halo_buf = fw ? &(halo_self_fw_[dim]) : &(halo_self_bw_[dim]);

  // The slowest changing dimension does not need actual copying
  // because its halo region is physically continuous.
  if (dim == (num_dims_ - 1)) {
    char *p = data_[0]
        + GridCalcOffset3D(halo_offset, local_real_size_) * elm_size_;
    LOG_DEBUG() << "halo_offset: " << halo_offset << "\n";
    *halo_buf = p;
    LOG_DEBUG() << "p: " << (void*)p << "\n";
    return;
  } else {
    IndexArray halo_size = local_real_size_;
    halo_size[dim] = width;
    CopyoutSubgrid(elm_size_, num_dims_, data_[0], local_real_size_,
                   *halo_buf, halo_offset, halo_size);
    return;
  }
}

} // namespace runtime
} // namespace physis


