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
  g->InitBuffer();
  return g;
}

std::ostream &GridMPI2::Print(std::ostream &os) const {
  os << "GridMPI2 {"
     << "elm_size: " << elm_size_
     << ", size: " << size_
     << ", global offset: " << global_offset_
     << ", local offset: " << local_offset_
     << ", local size: " << local_size_
     << "}";
  return os;
}

void GridMPI2::InitBuffer() {
  data_buffer_[0] = new BufferHost(num_dims_, elm_size_);
  data_buffer_[0]->Allocate(local_size_);
  data_buffer_[1] = NULL;
  data_[0] = (char*)data_buffer_[0]->Get();
  data_[1] = NULL;
}

void GridMPI2::DeleteBuffers() {
  if (empty_) return;
  Grid::DeleteBuffers();
}

void *GridMPI2::GetAddress(const IndexArray &indices_param) {
  IndexArray indices = indices_param;
  indices -= local_real_offset_;
  return (void*)(_data() +
                 GridCalcOffset3D(indices, local_real_size_)
                 * elm_size());
}

void GridMPI2::Copyout(void *dst) const {
  // REFACTORING: Implement offset access in the buffer classes and
  // use it instead.
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
                                    const Width2 &stencil_width,
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
  
  Width2 halo = stencil_width;
  for (int i = 0; i < num_dims_; ++i) {
    if (local_size[i] == 0 || proc_size()[i] == 1) {
      halo.bw[i] = 0;
      halo.fw[i] = 0;
    }
  }
  
  GridMPI2 *g = GridMPI2::Create(
      type, elm_size, num_dims, grid_size,
      grid_global_offset, local_offset, local_size,
      halo, attr);
  LOG_DEBUG() << "grid created\n";
  RegisterGrid(g);
  return g;
  
  return NULL;
}



} // namespace runtime
} // namespace physis


