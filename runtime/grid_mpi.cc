// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/grid_mpi.h"

#include <limits.h>

#include "runtime/grid_util.h"
#include "runtime/mpi_util.h"
#include "runtime/mpi_wrapper.h"

using namespace std;

namespace physis {
namespace runtime {

size_t GridMPI::CalcHaloSize(int dim, unsigned width, bool diagonal) {
  size_t s = 1;
  for (int i = 0; i < num_dims_; ++i) {
    if (i == dim) {
      s *= width;
    } else {
      size_t edge_width = local_size_[i];
      if (diagonal && i > dim)
        edge_width += halo_fw_width_[i] + halo_bw_width_[i];
      s *= edge_width;
    }
  }
  return s;
}

void GridMPI::SetHaloSize(int dim, bool fw, unsigned width, bool diagonal) {
  IndexArray s;
  for (int i = 0; i < num_dims_; ++i) {
    if (i == dim) {
      s[i] = width;
    } else {
      PSIndex edge_width = local_size_[i];
      if (diagonal && i > dim)
        edge_width += halo_fw_width_[i] + halo_bw_width_[i];
      s[i] = edge_width;
    }
  }
  if (fw) {
    halo_fw_size_[dim] = s;
  } else {
    halo_bw_size_[dim] = s;    
  }
  return;
}

GridMPI::GridMPI(PSType type, int elm_size, int num_dims,
                 const IndexArray &size,
                 bool double_buffering, const IndexArray &global_offset,
                 const IndexArray &local_offset,                 
                 const IndexArray &local_size, int attr):
    Grid(type, elm_size, num_dims, size, double_buffering, attr),
    global_offset_(global_offset),  
    local_offset_(local_offset), local_size_(local_size),
    halo_has_diagonal_(false),
    remote_grid_(NULL), remote_grid_active_(false) {
  
  empty_ = local_size_.accumulate(num_dims_) == 0;
  if (empty_) return;

  halo_self_fw_ = new char*[num_dims_];
  halo_self_bw_ = new char*[num_dims_];
  halo_peer_fw_ = new char*[num_dims_];
  halo_peer_bw_ = new char*[num_dims_];
  
  for (int i = 0; i < num_dims_; ++i) {
    halo_self_fw_[i] = NULL;
    halo_self_bw_[i] = NULL;
    halo_peer_fw_[i] = NULL;
    halo_peer_bw_[i] = NULL;
  }
}
// This is the only interface to create a GridMPI grid
GridMPI *GridMPI::Create(PSType type, int elm_size,
                         int num_dims, const IndexArray &size,
                         bool double_buffering,
                         const IndexArray &global_offset,
                         const IndexArray &local_offset,
                         const IndexArray &local_size,
                         int attr) {
  GridMPI *gm = new GridMPI(type, elm_size, num_dims, size,
                            double_buffering, global_offset,
                            local_offset, local_size, attr);
  gm->InitBuffers();
  return gm;
}

void GridMPI::InitBuffers() {
  data_buffer_[0] = new BufferHost(num_dims_, elm_size_);
  data_buffer_[0]->Allocate(local_size_);
  if (double_buffering_) {
    data_buffer_[1] = new BufferHost(num_dims_, elm_size_);
    data_buffer_[1]->Allocate(local_size_);    
  } else {
    data_buffer_[1] = data_buffer_[0];
  }
  data_[0] = (char*)data_buffer_[0]->Get();
  data_[1] = (char*)data_buffer_[1]->Get();
}

GridMPI::~GridMPI() {
  DeleteBuffers();
}

void GridMPI::DeleteBuffers() {
  if (empty_) return;
  // The buffers for the last dimension is the same as data buffer, so
  // freeing is not required.
  for (int i = 0; i < num_dims_-1; ++i) {
    if (halo_self_fw_ && halo_self_fw_[i]) {
      FREE(halo_self_fw_[i]);
    }
    if (halo_self_bw_ && halo_self_bw_[i]) {
      FREE(halo_self_bw_[i]);
    }
    if (halo_peer_fw_ && halo_peer_fw_[i]) {
      FREE(halo_peer_fw_[i]);
    }
    if (halo_peer_bw_ && halo_peer_bw_[i]) {
      FREE(halo_peer_bw_[i]);
    }
  }
  delete[] halo_self_fw_;
  halo_self_fw_ = NULL;
  delete[] halo_self_bw_;
  halo_self_bw_ = NULL;
  delete[] halo_peer_fw_;
  halo_peer_fw_ = NULL;
  delete[] halo_peer_bw_;
  halo_peer_bw_ = NULL;
  delete remote_grid_;
  remote_grid_ = NULL;
  Grid::DeleteBuffers();
}

void GridMPI::EnsureRemoteGrid(const IndexArray &local_offset,
                               const IndexArray &local_size) {
  if (remote_grid_ == NULL) {
    remote_grid_ = GridMPI::Create(type_, elm_size_, num_dims_,
                                   size_, false, global_offset_,
                                   local_offset, local_size, 0);
  } else {
    remote_grid_->Resize(local_offset, local_size);
  }
}

// Copy out halo with diagonal points
// PRECONDITION: halo for the first dim is already exchanged
void GridMPI::CopyoutHalo2D0(unsigned width, bool fw) {
  LOG_DEBUG() << "2d copyout halo\n";
  char *buf = (fw) ? halo_self_fw_[0] : halo_self_bw_[0];    
  size_t line_size = width * elm_size_;
  // copy diag
  size_t offset = fw ? 0 : (local_size_[0] - width) * elm_size_;
  char *diag_buf = halo_peer_bw_[1] + offset;
  for (unsigned i = 0; i < halo_bw_width_[1]; ++i) {
    memcpy(buf, diag_buf, line_size);
    buf += line_size;
    diag_buf += local_size_[0] * elm_size_;
  }
  
  // copy halo
  IndexArray sg_offset;
  if (!fw) sg_offset[0] = local_size_[0] - width;
  IndexArray sg_size(width, local_size_[1]);
  CopyoutSubgrid(elm_size_, 2, data_[0], local_size_,
                 buf, sg_offset, sg_size);
  buf += line_size * local_size_[1];

  // copy diag
  diag_buf = halo_peer_fw_[1] + offset;
  for (unsigned i = 0; i < halo_fw_width_[1]; ++i) {
    memcpy(buf, diag_buf, line_size);
    buf += line_size;
    diag_buf += local_size_[0] * elm_size_;
  }
  
  return;
}

char *GridMPI::GetHaloBuf(int dim, unsigned width,
                          bool fw, bool diagonal) {
  char **halo_buf;
  SizeArray *halo_cur_size;
  if (fw) {
    halo_buf = &(halo_self_fw_[dim]);
    halo_cur_size = &halo_self_fw_buf_size_;
  } else {
    halo_buf = &(halo_self_bw_[dim]);
    halo_cur_size = &halo_self_bw_buf_size_;
  }

  // Ensure large enough buffer
  size_t halo_bytes = CalcHaloSize(dim, width, diagonal) * elm_size_;
  if ((*halo_cur_size)[dim] < halo_bytes) {
    (*halo_cur_size)[dim] = halo_bytes;
    *halo_buf = (char*)malloc(halo_bytes);
  }
  return *halo_buf;
}
    

void GridMPI::CopyoutHalo3D0(unsigned width, bool fw) {
  char *buf = (fw) ? halo_self_fw_[0] : halo_self_bw_[0];
  size_t line_size = elm_size_ * width;  
  size_t xoffset = fw ? 0 : local_size_[0] - width;
  char *halo_bw_1 = halo_peer_bw_[1] + xoffset * elm_size_;
  char *halo_fw_1 = halo_peer_fw_[1] + xoffset * elm_size_;
  char *halo_bw_2 = halo_peer_bw_[2] + xoffset * elm_size_;
  char *halo_fw_2 = halo_peer_fw_[2] + xoffset * elm_size_;
  char *halo_0 = data_[0] + xoffset * elm_size_;
  for (PSIndex k = 0; k < ((PSIndex)halo_bw_width_[2])+local_size_[2]+
           ((PSIndex)halo_fw_width_[2]); ++k) {
    // copy from the 2nd-dim backward halo 
    char *t = halo_bw_1;
    for (unsigned j = 0; j < halo_bw_width_[1]; j++) {
      memcpy(buf, t, line_size);
      buf += line_size;      
      t += local_size_[0] * elm_size_;
    }
    halo_bw_1 += local_size_[0] * halo_bw_width_[1] * elm_size_;

    char **h;
    // select halo source
    if (k < (PSIndex)halo_bw_width_[2]) {
      h = &halo_bw_2;
    } else if (k < (PSIndex)halo_bw_width_[2] + local_size_[2]) {
      h = &halo_0;
    } else {
      h = &halo_fw_2;
    }
    t = *h;
    for (PSIndex j = 0; j < local_size_[1]; ++j) {
      memcpy(buf, t, line_size);
      buf += line_size;
      t += local_size_[0] * elm_size_;
    }
    *h += local_size_[0] * local_size_[1] * elm_size_;

    // copy from the 2nd-dim forward halo 
    t = halo_fw_1;
    for (unsigned j = 0; j < halo_fw_width_[1]; j++) {
      memcpy(buf, t, line_size);
      buf += line_size;      
      t += local_size_[0] * elm_size_;
    }
    halo_fw_1 += local_size_[0] * halo_fw_width_[1] * elm_size_;
  }
  
}

void GridMPI::CopyoutHalo3D1(unsigned width, bool fw) {
  char *buf = (fw) ? halo_self_fw_[1] : halo_self_bw_[1];  
  int nd = 3;
  // copy diag
  IndexArray sg_offset;
  if (!fw) sg_offset[1] = local_size_[1] - width;
  IndexArray sg_size(local_size_[0], width, halo_bw_width_[2]);
  IndexArray halo_size(local_size_[0], local_size_[1],
                       halo_bw_width_[2]);
  // different
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_bw_[2],
                 halo_size, buf, sg_offset, sg_size);
  buf += sg_size.accumulate(nd) * elm_size_;

  // copy halo
  sg_size[2] = local_size_[2];
  CopyoutSubgrid(elm_size_, num_dims_, data_[0], local_size_,
                 buf, sg_offset, sg_size);
  buf += sg_size.accumulate(nd) * elm_size_;

  // copy diag
  sg_size[2] = halo_fw_width_[2];
  halo_size[2] = halo_fw_width_[2];
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_fw_[2],
                 halo_size, buf, sg_offset, sg_size);
  return;
}

// fw: prepare buffer for sending halo for forward access if true
void GridMPI::CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal) {
  halo_has_diagonal_ = diagonal;

  if (dim == num_dims_ - 1) {
    if (fw) {
      halo_self_fw_[dim] = data_[0];
      return;
    } else {
      IndexArray s = local_size_;
      s[dim] -= width;
      halo_self_bw_[dim] = data_[0]
          + s.accumulate(num_dims_) * elm_size_;
      return;
    }
  }

  char *halo_buf = GetHaloBuf(dim, width, fw, diagonal);
  
#if 0  
  LOG_DEBUG() << "FW?: " << fw << ", width: " << width <<
      "local size: " << local_size_ << "\n";
#endif
  
  // If diagonal points are not required, copyoutSubgrid can be used
  if (!diagonal) {
    IndexArray halo_offset;
    if (!fw) halo_offset[dim] = local_size_[dim] - width;
    IndexArray halo_size = local_size_;
    halo_size[dim] = width;
    CopyoutSubgrid(elm_size_, num_dims_, data_[0], local_size_,
                   halo_buf, halo_offset, halo_size);
    return;
  }

  switch (num_dims_) {
    case 2:
      CopyoutHalo2D0(width, fw);
      break;
    case 3:
      if (dim == 0) {
        CopyoutHalo3D0(width, fw);
      } else if (dim == 1) {
        CopyoutHalo3D1(width, fw);
      } else {
        LOG_ERROR() << "This case should have been handled in the first block of this function.\n";
        PSAbort(1);
      }
      break;
    default:
      LOG_ERROR() << "Unsupported dimension: " << num_dims_ << "\n";
      PSAbort(1);
  }
  
  return;
}

void GridMPI::FixupBufferPointers() {
  data_[0] = (char*)data_buffer_[0]->Get();
  data_[1] = (char*)data_buffer_[1]->Get();
}

// Note: Halo regions are not updated.
void GridMPI::Resize(const IndexArray &local_offset,
                     const IndexArray &local_size) {
  // Resizing double buffered object not supported
  PSAssert(!double_buffering_);
  local_size_ = local_size;
  local_offset_ = local_offset;
  buffer()->EnsureCapacity(local_size_);
  FixupBufferPointers();
}


std::ostream &GridMPI::Print(std::ostream &os) const {
  os << "GridMPI {"
     << "elm_size: " << elm_size_
     << ", size: " << size_
     << ", global offset: " << global_offset_
     << ", local offset: " << local_offset_
     << ", local size: " << local_size_
     << "}";
  return os;
}

template <class T>
int ReduceGridMPI(GridMPI *g, PSReduceOp op, T *out) {
  size_t nelms = g->local_size().accumulate(g->num_dims());
  if (nelms == 0) return 0;
  boost::function<T (T, T)> func = GetReducer<T>(op);
  T *d = (T *)g->_data();
  T v = d[0];
  for (size_t i = 1; i < nelms; ++i) {
    v = func(v, d[i]);
  }
  *out = v;
  return nelms;
}

int GridMPI::Reduce(PSReduceOp op, void *out) {
  int rv = 0;
  switch (type_) {
    case PS_FLOAT:
      rv = ReduceGridMPI<float>(this, op, (float*)out);
      break;
    case PS_DOUBLE:
      rv = ReduceGridMPI<double>(this, op, (double*)out);
      break;
    default:
      PSAbort(1);
  }
  return rv;
}



std::ostream &GridSpaceMPI::Print(std::ostream &os) const {
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

// For example
// num_dims: 3 <dimension>
// num_procs: 6 = 1*1*6
// size = global_size: {64, 64, 64}
// num_partitions = proc_size: {1, 1, 6}
static void partition(int num_dims, int num_procs,
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

// For example
// num_dims: 3 <dimension>
// global_size: {64, 64, 64}
// proc_num_dims: 3 <dimension>
// proc_size: {1, 1, 6}
GridSpaceMPI::GridSpaceMPI(int num_dims, const IndexArray &global_size,
                           int proc_num_dims, const IntArray &proc_size,
                           int my_rank):
    num_dims_(num_dims), global_size_(global_size),
    proc_num_dims_(proc_num_dims), proc_size_(proc_size),
    my_rank_(my_rank), buf(NULL), cur_buf_size(0) {
  assert(num_dims_ == proc_num_dims_);
  
  num_procs_ = proc_size_.accumulate(proc_num_dims_); // For example 6

  partitions_ = new PSIndex*[num_dims_];
  offsets_ = new PSIndex*[num_dims_];
  
  partition(num_dims_, num_procs_, global_size_, proc_size_,
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

GridSpaceMPI::~GridSpaceMPI() {
  FREE(buf);
}

void GridSpaceMPI::PartitionGrid(int num_dims, const IndexArray &size,
                                 const IndexArray &global_offset,
                                 IndexArray &local_offset, IndexArray &local_size) {
  for (int i = 0; i < num_dims; ++i) {
    local_offset[i] = std::max(my_offset_[i] - global_offset[i], (PSIndex)0);
    PSIndex first = std::max(my_offset_[i], global_offset[i]);
    PSIndex last = std::min(global_offset[i] + size[i],
                        my_offset_[i] + my_size_[i]);
    local_size[i] = std::max(last - first, (PSIndex)0);
  }
  return;
}

GridMPI *GridSpaceMPI::CreateGrid(PSType type, int elm_size,
                                  int num_dims,
                                  const IndexArray &size,
                                  bool double_buffering,
                                  const IndexArray &global_offset,
                                  int attr) {
  IndexArray grid_size = size;
  IndexArray grid_global_offset = global_offset;
  if (num_dims_ < num_dims) {
    // The grid space has smaller dimension.
    LOG_ERROR() << "Cannot create " << num_dims << "-d grid in "
                << num_dims_ << "-d space.\n";
    PSAbort(1);
  } else if (num_dims_ > num_dims) {
    // Adapt the dimension of grid to the grid space
    for (int i = num_dims; i < num_dims_; ++i) {
      grid_size[i] = 1;
      grid_global_offset[i] = 0;
    }
    num_dims = num_dims_;
  }

  IndexArray local_offset, local_size;  
  PartitionGrid(num_dims, grid_size, grid_global_offset,
                local_offset, local_size);

  LOG_DEBUG() << "local_size: " << local_size << "\n";
  GridMPI *g = GridMPI::Create(type, elm_size, num_dims, grid_size,
                               double_buffering,
                               grid_global_offset, local_offset,
                               local_size, attr);
  LOG_DEBUG() << "grid created\n";
  RegisterGrid(g);
  return g;
}

// Note: width is unsigned. 
void GridSpaceMPI::ExchangeBoundariesAsync(
    GridMPI *grid, int dim, unsigned halo_fw_width, unsigned halo_bw_width,
    bool diagonal, bool periodic,
    std::vector<MPI_Request> &requests) const {
  
  if (grid->empty_) return;
  
  int fw_peer = fw_neighbors_[dim];
  int bw_peer = bw_neighbors_[dim];
  int tag = 0;
  size_t fw_size = grid->CalcHaloSize(dim, halo_fw_width, diagonal)
      * grid->elm_size_;
  size_t bw_size = grid->CalcHaloSize(dim, halo_bw_width, diagonal)
      * grid->elm_size_;

  LOG_DEBUG() << "Periodic?: " << periodic << "\n";

  /*
    Send and receive ordering must match. First get the halo for the
    forward access, and then the halo for the backward access.
   */
  
  if (halo_fw_width > 0 &&
      (periodic ||
       grid->local_offset_[dim] + grid->local_size_[dim]
       < grid->size_[dim])) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Receiving halo of " << fw_size
                << " bytes for fw access from " << fw_peer << "\n";
    if (grid->halo_peer_fw_buf_size_[dim] < fw_size) {
      LOG_DEBUG() << "Allocating buffer\n";
      FREE(grid->halo_peer_fw_[dim]);
      grid->halo_peer_fw_[dim] = (char*)malloc(fw_size);
      grid->halo_peer_fw_buf_size_[dim] = fw_size;
    }
    grid->halo_fw_width_[dim] = halo_fw_width;
    grid->SetHaloSize(dim, true, halo_fw_width, diagonal);
    MPI_Request req;
    CHECK_MPI(MPI_Irecv(grid->halo_peer_fw_[dim], fw_size, MPI_BYTE,
                        fw_peer, tag, comm_, &req));
    requests.push_back(req);
  } else {
    grid->halo_fw_width_[dim] = 0;
    grid->halo_fw_size_[dim].Set(0);
  }

  if (halo_bw_width > 0 &&
      (periodic || grid->local_offset_[dim] > 0)) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Receiving halo of " << bw_size
                << " bytes for bw access from " << bw_peer << "\n";
    if (grid->halo_peer_bw_buf_size_[dim] < bw_size) {
      LOG_DEBUG() << "Allocating buffer\n";
      FREE(grid->halo_peer_bw_[dim]);
      grid->halo_peer_bw_[dim] = (char*)malloc(bw_size);
      grid->halo_peer_bw_buf_size_[dim] = bw_size;
    }
    grid->halo_bw_width_[dim] = halo_bw_width;
    grid->SetHaloSize(dim, false, halo_bw_width, diagonal);    
    MPI_Request req;
    CHECK_MPI(MPI_Irecv(grid->halo_peer_bw_[dim], bw_size, MPI_BYTE,
                        bw_peer, tag, comm_, &req));
    requests.push_back(req);
  } else {
    grid->halo_bw_width_[dim] = 0;
    grid->halo_bw_size_[dim].Set(0);
  }

  // Sends out the halo for forward access
  if (halo_fw_width > 0 &&
      (periodic || grid->local_offset_[dim] > 0)) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Sending halo of " << fw_size << " bytes"
                << " for fw access to " << bw_peer << "\n";
    grid->CopyoutHalo(dim, halo_fw_width, true, diagonal);
    MPI_Request req;
    CHECK_MPI(PS_MPI_Isend(grid->halo_self_fw_[dim], fw_size, MPI_BYTE,
                        bw_peer, tag, comm_, &req));
  }

   // Sends out the halo for backward access
  if (halo_bw_width > 0 &&
      (periodic || 
       grid->local_offset_[dim] + grid->local_size_[dim]
       < grid->size_[dim])) {
    LOG_DEBUG() << "[" << my_rank_ << "] "
                << "Sending halo of " << bw_size << " bytes"
                << " for bw access to " << fw_peer << "\n";
    grid->CopyoutHalo(dim, halo_bw_width, false, diagonal);
    MPI_Request req;
    CHECK_MPI(PS_MPI_Isend(grid->halo_self_bw_[dim], bw_size, MPI_BYTE,
                        fw_peer, tag, comm_, &req));
  }

  return;
}

void GridSpaceMPI::ExchangeBoundaries(GridMPI *grid,
                                      int dim,
                                      unsigned halo_fw_width,
                                      unsigned halo_bw_width,
                                      bool diagonal,
                                      bool periodic) const {
  std::vector<MPI_Request> requests;
  ExchangeBoundariesAsync(grid, dim, halo_fw_width,
                          halo_bw_width, diagonal,
                          periodic, requests);
  FOREACH (it, requests.begin(), requests.end()) {
    MPI_Request *req = &(*it);
    CHECK_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));
  }
  
  return;
}
#if 0
void GridSpaceMPI::ExchangeBoundariesAsync(
    int grid_id,  const UnsignedArray &halo_fw_width,
    const UnsignedArray &halo_bw_width, bool diagonal,
    std::vector<MPI_Request> &requests) const {

  LOG_ERROR() << "This does not correctly copy halo regions. "
              << "Halo exchange of previous dimension must be completed"
              << " before sending out next halo region.";
  PSAbort(1);
  
  
  GridMPI *g = static_cast<GridMPI*>(FindGrid(grid_id));
  
  for (int i = g->num_dims_ - 1; i >= 0; --i) {
    LOG_DEBUG() << "Exchanging dimension " << i << " data\n";
    ExchangeBoundariesAsync(*g, i, halo_fw_width[i],
                            halo_bw_width[i], diagonal,
                            periodic,requests);
  }
  return;
}
#endif

// TODO: reuse is used?
void GridSpaceMPI::ExchangeBoundaries(int grid_id,
                                      const UnsignedArray &halo_fw_width,
                                      const UnsignedArray &halo_bw_width,
                                      bool diagonal,
                                      bool periodic, 
                                      bool reuse) const {
  LOG_DEBUG() << "GridSpaceMPI::ExchangeBoundaries\n";

  GridMPI *g = static_cast<GridMPI*>(FindGrid(grid_id));
  for (int i = g->num_dims_ - 1; i >= 0; --i) {
    LOG_VERBOSE() << "Exchanging dimension " << i << " data\n";
    PSAssert(halo_fw_width[i] >=0);
    PSAssert(halo_bw_width[i] >=0);
    ExchangeBoundaries(g, i, halo_fw_width[i],
                       halo_bw_width[i], diagonal, periodic);
  }
  return;
}

void SendGridRequest(int my_rank, int peer_rank,
                     MPI_Comm comm,
                     GRID_REQUEST_KIND kind) {
  LOG_DEBUG() << "Sending request " << kind << " to " << peer_rank << "\n";
  GridRequest req(my_rank, kind);
  MPI_Request mpi_req;
  CHECK_MPI(PS_MPI_Isend(&req, sizeof(GridRequest), MPI_BYTE,
                      peer_rank, 0, comm, &mpi_req));
}

GridRequest RecvGridRequest(MPI_Comm comm) {
  GridRequest req;
  CHECK_MPI(MPI_Recv(&req, sizeof(GridRequest), MPI_BYTE, MPI_ANY_SOURCE, 0,
                     comm, MPI_STATUS_IGNORE));
  return req;
}

static void *ensure_buffer_capacity(void *buf, size_t cur_size,
                                    size_t required_size) {
  if (cur_size < required_size) {
    FREE(buf);
    cur_size = required_size;
    buf = malloc(cur_size);
  }
  return buf;
}

void GridSpaceMPI::CollectPerProcSubgridInfo(
    const GridMPI *g,  const IndexArray &grid_offset,
    const IndexArray &grid_size,
    std::vector<FetchInfo> &finfo_holder) const {
  LOG_DEBUG() << "Collecting per-process subgrid info\n";
  IndexArray grid_lim = grid_offset + grid_size;
  std::vector<FetchInfo> *fetch_info = new std::vector<FetchInfo>;
  std::vector<FetchInfo> *fetch_info_next = new std::vector<FetchInfo>;
  FetchInfo dummy;
  fetch_info->push_back(dummy);
  for (int d = 0; d < num_dims_; ++d) {
    PSIndex x = grid_offset[d] + g->global_offset_[d];
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
bool GridSpaceMPI::SendFetchRequest(FetchInfo &finfo) const {
  int peer_rank = GetProcessRank(finfo.peer_index);
  size_t size = finfo.peer_size.accumulate(num_dims_);
  if (size == 0) return false; 
  SendGridRequest(my_rank_, peer_rank, comm_, FETCH_REQUEST);
  MPI_Request mpi_req;  
  CHECK_MPI(PS_MPI_Isend(&finfo, sizeof(FetchInfo), MPI_BYTE,
                      peer_rank, 0, comm_, &mpi_req));
  return true;
}

void GridSpaceMPI::HandleFetchRequest(GridRequest &req, GridMPI *g) {
  LOG_DEBUG() << "HandleFetchRequest\n";
  FetchInfo finfo;
  int nd = num_dims_;
  CHECK_MPI(MPI_Recv(&finfo, sizeof(FetchInfo), MPI_BYTE,
                     req.my_rank, 0, comm_, MPI_STATUS_IGNORE));
  size_t bytes = finfo.peer_size.accumulate(nd) * g->elm_size();
  buf = ensure_buffer_capacity(buf, cur_buf_size, bytes);
  CopyoutSubgrid(g->elm_size(), nd, g->_data(), g->local_size(),
                 buf, finfo.peer_offset - g->local_offset(),
                 finfo.peer_size);
  SendGridRequest(my_rank_, req.my_rank, comm_, FETCH_REPLY);
  MPI_Request mr;
  CHECK_MPI(PS_MPI_Isend(buf, bytes, MPI_BYTE, req.my_rank, 0, comm_, &mr));
  return;
}

void GridSpaceMPI::HandleFetchReply(GridRequest &req, GridMPI *g,
                                    std::map<int, FetchInfo> &fetch_map,
                                    GridMPI *sg) {
  LOG_DEBUG() << "HandleFetchReply\n";
  const FetchInfo &finfo = fetch_map[req.my_rank];
  PSAssert(GetProcessRank(finfo.peer_index) == req.my_rank);
  size_t bytes = finfo.peer_size.accumulate(num_dims_) * g->elm_size();
  LOG_DEBUG() << "Fetch reply data size: " << bytes << "\n";
  buf = ensure_buffer_capacity(buf, cur_buf_size, bytes);
  CHECK_MPI(MPI_Recv(buf, bytes, MPI_BYTE, req.my_rank, 0,
                     comm_, MPI_STATUS_IGNORE));
  LOG_DEBUG() << "Fetch reply received\n";
  CopyinSubgrid(sg->elm_size(), num_dims_, sg->_data(),
                sg->local_size(), buf,
                finfo.peer_offset - sg->local_offset(), finfo.peer_size);
  return;;
}

void *GridMPI::GetAddress(const IndexArray &indices_param) {
  IndexArray indices = indices_param;
  // If remote_grid is available, compute the address within the
  // remote grid.
  if (remote_grid_active()) {
    GridMPI *rmg = remote_grid();
    indices -= rmg->local_offset();
    return (void*)(rmg->_data() +
                   GridCalcOffset3D(indices, rmg->local_size())
                   * rmg->elm_size());
  }
  
  indices -= local_offset();
  bool diag = halo_has_diagonal();
  // Check the location corresponds to halo regions
  for (int i = 0; i < num_dims(); ++i) {
    if (indices[i] < 0 || indices[i] >= local_size()[i]) {
      for (int j = i+1; j < PS_MAX_DIM; ++j) {
        if (diag) indices[i] += halo_bw_width()[i];
      }
      PSIndex offset;
      char *buf;
      if (indices[i] < 0) {
        indices[i] += halo_bw_width()[i];
        offset = GridCalcOffset3D(indices, halo_bw_size()[i]);
        buf = _halo_peer_bw()[i];
      } else {
        indices[i] -= local_size()[i];
        offset = GridCalcOffset3D(indices, halo_fw_size()[i]);
        buf = _halo_peer_fw()[i];
      }
      return (void*)(buf + offset*elm_size());
    }
  }

  return (void*)(_data() +
                 GridCalcOffset3D(indices, local_size())
                 * elm_size());
}

// Returns new subgrid when necessary.
GridMPI *GridSpaceMPI::LoadSubgrid(GridMPI *g, const IndexArray &grid_offset,
                                   const IndexArray &grid_size,
                                   bool reuse) {
  LOG_DEBUG() << __FUNCTION__ 
                  << ": grid offset: " << grid_offset << ", grid size: "
                  << grid_size << "\n";

  // This is not required, but just for ensuring all processes be here.
  //CHECK_MPI(MPI_Barrier(comm_));

  PSAssert(grid_offset >= 0);
  PSAssert(grid_size >= 0);
  
  std::vector<FetchInfo> fetch_requests;
  CollectPerProcSubgridInfo(g, grid_offset, grid_size, fetch_requests);
  std::map<int, FetchInfo> fetch_map;

  GridMPI *sg = NULL;

  if (fetch_requests.size()) {
    if (fetch_requests.size() == 1 &&
        GetProcessRank(fetch_requests[0].peer_index) == my_rank_) {
      // This request can just be satisfied by returning the current
      // grid object itself since the requested region falls inside
      // this grid. Return NULL then.
      LOG_DEBUG() << "No actual loading since requested region included in the local grid\n";
      sg = NULL;
      fetch_requests.clear();
    } else {
      if (reuse && g->remote_grid() &&
          g->remote_grid()->local_offset() == grid_offset &&
          g->remote_grid()->local_size() == grid_size) {
        // reuse previously loaded remote grid
        fetch_requests.clear();
        g->remote_grid_active() = true;
      } else {
        g->EnsureRemoteGrid(grid_offset, grid_size);
        sg = g->remote_grid();
        g->remote_grid_active() = true;
      }
    }
  } else {
    LOG_DEBUG() << "No fetch needed for this process\n";
  }

  // Sending out requests for copying subgrids
  int remaining_requests = 0;  
  FOREACH (it, fetch_requests.begin(), fetch_requests.end()) {
    // Note: this can be the dummy info
    FetchInfo &finfo = *it;
    StringJoin sj;
    sj << finfo.peer_index;
    sj << finfo.peer_offset;
    sj << finfo.peer_size;
    LOG_DEBUG() << "Fetch info: " << sj << "\n";
    int peer_rank = GetProcessRank(finfo.peer_index);
    if (SendFetchRequest(finfo)) {
      ++remaining_requests;
      fetch_map.insert(make_pair(peer_rank, finfo));
    }
  }

  int done_count = 0;
  bool done_sent = false;
  while (done_count != num_procs_) {
    // If finished, send out DONE messages
    if (remaining_requests == 0 && !done_sent) {
      LOG_DEBUG() << "Notify the others this process is done\n";
      for (int i = 0; i < num_procs_; ++i) {
        if (i == my_rank_)  ++done_count;
        else SendGridRequest(my_rank_, i, comm_, DONE);
      }
      done_sent = true;
      continue;
    }

    LOG_DEBUG() << "[" << my_rank_ << "] " << "Listening"
                    << " (remaining: " << remaining_requests << ")\n";
    GridRequest req = RecvGridRequest(comm_);
    switch (req.kind) {
      case DONE:
        LOG_DEBUG() << "Done\n";
        ++done_count;
        break;
      case FETCH_REQUEST:
        HandleFetchRequest(req, g);
        break;
      case FETCH_REPLY:
        HandleFetchReply(req, g, fetch_map, sg);
        --remaining_requests;
        break;
      default:
        LOG_ERROR() << "Unknown request\n";
        PSAbort(1);
    }
  }

  //MPI_Barrier(MPI_COMM_WORLD);
  
  return sg;
}

int GridSpaceMPI::GetProcessRank(const IntArray &proc_index) const {
  int rank = 0;
  int offset = 1;
  for (int i = 0; i < num_dims(); ++i) {
    rank += proc_index[i] * offset;
    offset *= proc_size_[i];
  }
  return rank;
}

GridMPI *GridSpaceMPI::LoadNeighbor(GridMPI *g,
                                    const IndexArray &offset_min,
                                    const IndexArray &offset_max,
                                    bool diagonal,
                                    bool reuse,
                                    bool periodic) {
  int nd = g->num_dims();

  // Check whether exchangeNeighbor can be used.
  // TODO: What stencil is not possible to handle with
  // exchangeNeighbor? Does this actually happen?
  bool overlap = true;
  for (int i = 0; i < nd; ++i) {
    if (offset_max[i] + min_partition_[i] >= 0 ||
        offset_min[i] <= min_partition_[i]) {
      LOG_VERBOSE() << "Dim " << i << " overlapping\n";
      continue;
    } else {
      LOG_DEBUG() << "Dim " << i << " not overlapping\n";
      LOG_DEBUG() << "min_partition[" << i << "]: "
                  << min_partition_[i] << ", offset_min: "
                  << offset_min[i] << ", offset_max: "
                  << offset_max[i] << "\n";
      overlap = false;
      break;
    }
  }

  // Don't know why overlap can be negative. Assume it's true for
  // now. 
  PSAssert(overlap);

  if (overlap) {
    // Use ExchangeBoundaries

    // OPTIMIZATION: Not all processes need to exchange boundaries,
    // so this function accepts the two flags to signal whether they
    // are necessary. But since ExchangeBoudnaries methods do not
    // provide such an interface, we currently let all processes join
    // exchange communication. This can be more efficient by allowing
    // selective message exchange. It would be particularly
    // adavantageous if only a very small sub set of processes join
    // the communication.
    UnsignedArray halo_fw_width, halo_bw_width;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      halo_bw_width[i] = (offset_min[i] <= 0) ? (unsigned)(abs(offset_min[i])) : 0;
      halo_fw_width[i] = (offset_max[i] >= 0) ? (unsigned)(offset_max[i]) : 0;
    }
    ExchangeBoundaries(g->id(), halo_fw_width,
                       halo_bw_width, diagonal, periodic, reuse);
    return NULL;
  } else {
    // Use LoadSubgrid
    // NOTE: periodic is not supported.
    PSAssert(!periodic);
    IndexArray grid_offset(g->local_offset());
    grid_offset = grid_offset + offset_min;
    grid_offset.SetNoLessThan(0);
    IndexArray grid_y = g->local_offset() + g->local_size() + offset_max;
    return LoadSubgrid(g, grid_offset, grid_y - grid_offset, reuse);
  }
}

int GridSpaceMPI::FindOwnerProcess(GridMPI *g, const IndexArray &index) {
  std::vector<FetchInfo> fetch_requests;
  IndexArray one;
  one.Set(1);
  CollectPerProcSubgridInfo(g, index, one, fetch_requests);
  PSAssert(fetch_requests.size() == 1);
  return GetProcessRank(fetch_requests[0].peer_index);
}

void LoadSubgrid(GridMPI *gm, GridSpaceMPI *gs,
                 int *dims, const IndexArray &min_offsets,
                 const IndexArray &max_offsets,
                 bool reuse) {
  int nd = gm->num_dims();
  IndexArray base_offset;
  IndexArray base_size;    
  for (int i = 0; i < nd; i++) {
    if (dims[i]) {
      base_offset[i] = gm->local_offset()[abs(dims[i])-1];
      base_size[i] = gm->local_size()[abs(dims[i])-1];
    } else {
      // accessing a fixed point      
      base_offset[i] = 0;
      base_size[i] = 1;
    }
  }

  IndexArray base_right = base_offset + base_size;
  IndexArray offset;
  for (int i = 0; i < nd; ++i) {
    if (dims[i] >= 0) {
      offset[i] = base_offset[i] + min_offsets[i];
    } else {
      offset[i] = -base_right[i] + 1 + min_offsets[i];
    }
  }
  IndexArray right = offset + base_size;
  offset.SetNoLessThan(0);
  right.SetNoMoreThan(gm->size());
    
  gs->LoadSubgrid(gm, offset, right - offset, reuse);
  return;
}




int GridSpaceMPI::ReduceGrid(void *out, PSReduceOp op,
                             GridMPI *g) {
  void *p = malloc(g->elm_size());
  if (g->Reduce(op, p) == 0) {
    switch (g->type()) {
      case PS_FLOAT:
        *(float*)p = GetReductionDefaultValue<float>(op);
        break;
      case PS_DOUBLE:
        *(double*)p = GetReductionDefaultValue<float>(op);
        break;
      default:
        PSAbort(1);
    }
  }
  MPI_Datatype type = GetMPIDataType(g->type());
  MPI_Op mpi_op = GetMPIOp(op);
  PS_MPI_Reduce(p, out, 1, type, mpi_op, 0, comm_);
  free(p);
  return g->num_elms();
}

} // namespace runtime
} // namespace physis


