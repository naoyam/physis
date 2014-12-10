// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/grid_mpi.h"

#include <limits.h>
#include <algorithm>

#include "runtime/grid_util.h"

using namespace std;

namespace physis {
namespace runtime {

size_t GridMPI::CalcHaloSize(int dim, unsigned width) {
  IndexArray halo_size = local_real_size_;
  halo_size[dim] = width;
  return halo_size.accumulate(num_dims_);
}

GridMPI::GridMPI(PSType type, int elm_size, int num_dims,
                 const IndexArray &size,
                 const IndexArray &global_offset,
                 const IndexArray &local_offset,                 
                 const IndexArray &local_size,
                 const Width2 &halo,
                 int attr):
    Grid(type, elm_size, num_dims, size, false, attr),
    global_offset_(global_offset),  
    local_offset_(local_offset), local_size_(local_size),
    halo_self_fw_(NULL), halo_self_bw_(NULL),
    halo_peer_fw_(NULL), halo_peer_bw_(NULL) {
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

GridMPI *GridMPI::Create(
    PSType type, int elm_size,
    int num_dims, const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &local_offset,
    const IndexArray &local_size,
    const Width2 &halo,
    int attr) {
  GridMPI *g = new GridMPI(
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

void GridMPI::InitBuffers() {
  if (empty_) return;  
  data_buffer_[0] = new BufferHost();
  data_buffer_[0]->Allocate(GetLocalBufferRealSize());
  data_buffer_[1] = NULL;
  data_[0] = (char*)data_buffer_[0]->Get();
  LOG_DEBUG() << "buffer addr: " << (void*)(data_[0]) << "\n";
  data_[1] = NULL;
  InitHaloBuffers();
}

void GridMPI::InitHaloBuffers() {
  // Note that the halo for the last dimension is continuously located
  // in memory, so no separate buffer is necessary.

  if (num_dims_ > 1) {
    halo_self_fw_ = new char*[num_dims_-1];
    halo_self_bw_ = new char*[num_dims_-1];
    halo_peer_fw_ = new char*[num_dims_-1];
    halo_peer_bw_ = new char*[num_dims_-1];
  }
  
  for (int i = 0; i < num_dims_ - 1; ++i) {
    // Initialize to NULL by default
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

GridMPI::~GridMPI() {
  DeleteBuffers();
}

void GridMPI::DeleteBuffers() {
  if (empty_) return;
  DeleteHaloBuffers();
  Grid::DeleteBuffers();
}

void GridMPI::DeleteHaloBuffers() {
  if (empty_) return;
  
  for (int i = 0; i < num_dims_ - 1; ++i) {
    if (halo_self_fw_) PS_XFREE(halo_self_fw_[i]);
    if (halo_self_bw_) PS_XFREE(halo_self_bw_[i]);
    if (halo_peer_fw_) PS_XFREE(halo_peer_fw_[i]);
    if (halo_peer_bw_) PS_XFREE(halo_peer_bw_[i]);
  }
  PS_XDELETEA(halo_self_fw_);
  PS_XDELETEA(halo_self_bw_);
  PS_XDELETEA(halo_peer_fw_);
  PS_XDELETEA(halo_peer_bw_);
}

char *GridMPI::GetHaloPeerBuf(int dim, bool fw, unsigned width) {
  if (dim == num_dims_ - 1) {
    IndexArray offset(0);
    if (fw) {
      offset[dim] = local_real_size_[dim] - halo_.fw[dim];
    } else {
      offset[dim] = halo_.fw[dim] - width;
    }
    return _data() + GridCalcOffset(offset, local_real_size_, num_dims_)
        * elm_size_;
  } else {
    if (fw) return halo_peer_fw_[dim];
    else  return halo_peer_bw_[dim];
  }
  
}
// fw: copy in halo buffer received for forward access if true
void GridMPI::CopyinHalo(int dim, unsigned width, bool fw, bool diagonal) {
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
void GridMPI::CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal) {
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
        + GridCalcOffset(halo_offset, local_real_size_, num_dims_) * elm_size_;
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


std::ostream &GridMPI::Print(std::ostream &os) const {
  os << "GridMPI {"
     << "num_dims: " << num_dims_
     << "elm_size: " << elm_size_
     << ", size: " << size_
     << ", global offset: " << global_offset_
     << ", local offset: " << local_offset_
     << ", local size: " << local_size_
     << ", local real size: " << local_real_size_      
     << "}";
  return os;
}

template <class T>
int ReduceGridMPI(GridMPI *g, PSReduceOp op, T *out, int dim) {
  PSAssert(dim <= 3);
  size_t nelms = g->local_size().accumulate(g->num_dims());
  if (nelms == 0) return 0;
  boost::function<T (T, T)> func = GetReducer<T>(op);
  T *d = (T *)g->_data();
  T v = GetReductionDefaultValue<T>(op);
  int imax = g->local_size()[0];
  int jmax = g->local_size()[1];
  int kmax = g->local_size()[2];
  if (dim == 1) {
    jmax = kmax = 1;
  } else if (dim == 2) {
    kmax = 1;
  }
  for (int k = 0; k < kmax; ++k) {
    for (int j = 0; j < jmax; ++j) {
      for (int i = 0; i < imax; ++i) {
        IndexArray p(i, j, k);
        p += g->halo().bw;
        intptr_t offset =
            GridCalcOffset(p, g->local_real_size(), dim);
        v = func(v, d[offset]);
      }
    }
  }
  *out = v;
  return nelms;
}

int GridMPI::Reduce(PSReduceOp op, void *out) {
  int rv = 0;
  PSAssert(num_dims_ <= 3);
  switch (type_) {
    case PS_FLOAT:
      rv = ReduceGridMPI<float>(this, op, (float*)out, num_dims_);
      break;
    case PS_DOUBLE:
      rv = ReduceGridMPI<double>(this, op, (double*)out, num_dims_);
      break;
    case PS_INT:
      rv = ReduceGridMPI<int>(this, op, (int*)out, num_dims_);
      break;
    case PS_LONG:
      rv = ReduceGridMPI<long>(this, op, (long*)out, num_dims_);
      break;
    default:
      LOG_ERROR() << "Unsupported type\n";
      PSAbort(1);
  }
  return rv;
}


void GridMPI::Copyout(void *dst) const {
  const void *src = buffer()->Get();
  if (HasHalo()) {
    IndexArray offset(halo_.bw);
    CopyoutSubgrid(elm_size(), num_dims(),
                   src, local_real_size(),
                   dst, offset, local_size());
  } else {
    memcpy(dst, src, GetLocalBufferSize());
  }
  return;
}

void GridMPI::Copyin(const void *src) {
  void *dst = buffer()->Get();
  if (HasHalo()) {
    CopyinSubgrid(elm_size(), num_dims(),
                  dst, local_real_size(),
                  src, halo_.bw, local_size());
  } else {
    memcpy(dst, src, GetLocalBufferSize());
  }
  return;
}




} // namespace runtime
} // namespace physis


