// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/grid_mpi_cuda_exp.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/buffer_cuda.h"
#include "runtime/grid_util.h"

#include <cuda.h>
#include <cuda_runtime.h>

using std::make_pair;

namespace physis {
namespace runtime {

GridMPICUDAExp::GridMPICUDAExp(
    PSType type, int elm_size, int num_dims, const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &local_offset, const IndexArray &local_size,
    const Width2 &halo,                 
    int attr): GridMPI(type, elm_size, num_dims, size, global_offset,
                       local_offset, local_size,
                       halo, attr), dev_(NULL) {
  if (empty_) return;
}

GridMPICUDAExp::~GridMPICUDAExp() {
  DeleteBuffers();
}

GridMPICUDAExp *GridMPICUDAExp::Create(
    PSType type, int elm_size,
    int num_dims, const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &local_offset,
    const IndexArray &local_size,
    const Width2 &halo,
    int attr) {
  GridMPICUDAExp *g = new GridMPICUDAExp(
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

size_t GridMPICUDAExp::CalcHaloSize(int dim, unsigned width, bool diagonal) {
  // This class sends the whole allocated halo region, irrespective of
  // how much of it is actually required, which is specified by
  // width. This avoids non-continuous memory copies and simplifies
  // implementations.
  // This is actually the same as GridMPI. 
  IndexArray halo_size = local_real_size_;
  halo_size[dim] = width;
  return halo_size.accumulate(num_dims_);
}

namespace {
void *MallocPSGridDevType(int dim) {
  size_t s = 0;
  switch (dim) {
    case 1:
      s = sizeof(__PSGrid1D_dev);
      break;
    case 2:
      s = sizeof(__PSGrid2D_dev);
      break;
    case 3:
      s = sizeof(__PSGrid3D_dev);
      break;
    default:
      LOG_ERROR() << "Dimension, " << dim << ", not supported\n";
      PSAbort(1);
  }
  void *p = malloc(s);
  PSAssert(p);
  return p;
}

}

void GridMPICUDAExp::InitBuffers() {
  LOG_DEBUG() << "Initializing grid buffer\n";

  if (empty_) return;
  data_buffer_ = new BufferCUDADev();
  data_buffer_->Allocate(GetLocalBufferRealSize());  
  
  halo_self_host_ = new BufferCUDAHost*[num_dims_][2];
  halo_peer_host_ = new BufferCUDAHost*[num_dims_][2];
  halo_peer_dev_ = new BufferCUDADev*[num_dims_][2];
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
      size_t halo_size =
          CalcHaloSize(i, halo_(j==1)[i], false) *
          elm_size();
      BufferCUDAHost *self_host = NULL;
      BufferCUDAHost *peer_host = NULL;
      if (halo_size) {
        self_host = new BufferCUDAHost();
        self_host->Allocate(halo_size);
        peer_host = new BufferCUDAHost();
        peer_host->Allocate(halo_size);
      }
      GetHaloSelfHost(i, j == 1) = self_host;
      GetHaloPeerHost(i, j == 1) = peer_host;
      
      // REFACTORING: unused
      halo_peer_dev_[i][j] = NULL;
    }
  }

  // buffer to be sent as a pass-by-value parameter for CUDA kernel
  // calls
  PSAssert(dev_ == NULL);
  dev_ = MallocPSGridDevType(num_dims_);

  FixupBufferPointers();
}


// Fix addresses in dev_
namespace {
template <class DevType>
void FixupDevBuffer(GridMPICUDAExp &g) {
  if (!g.GetDev()) {
    LOG_DEBUG() << "dev_ is NULL; nothing to do\n";
    return;
  }

  DevType *dev_ptr = (DevType*)g.GetDev();
  dev_ptr->p = g.buffer()->Get();
  for (int i = 0; i < g.num_dims(); ++i) {
    dev_ptr->dim[i] = g.size()[i];
    dev_ptr->local_size[i] = g.local_size()[i];
    dev_ptr->local_offset[i] = g.local_offset()[i];
  }
}
}

void GridMPICUDAExp::FixupBufferPointers() {
  LOG_VERBOSE() << "Fixup buffer ptrs\n";
  if (data_buffer_) {
    if (data_ != (char*)data_buffer_->Get()) {
      LOG_DEBUG() << "Buf pointer updated by " <<
	data_buffer_->Get() << "\n";
    }
    data_ = (char*)data_buffer_->Get();
  } else {
    LOG_VERBOSE() << "data buffer null\n";
    data_ = NULL;
  }
  
  switch (num_dims_) {
    case 1: FixupDevBuffer<__PSGrid1D_dev>(*this);
      break;
    case 2: FixupDevBuffer<__PSGrid2D_dev>(*this);
      break;
    case 3: FixupDevBuffer<__PSGrid3D_dev>(*this);
      break;
    default:
      LOG_ERROR() << "Dimension, " << num_dims_ << ", not supported\n";
      PSAbort(1);
  }
}



void GridMPICUDAExp::DeleteBuffers() {
  if (empty_) return;
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (halo_self_host_) {
        delete halo_self_host_[i][j];
        halo_self_host_[i][j] = NULL;
      }
      if (halo_peer_host_) {
        delete halo_peer_host_[i][j];
        halo_peer_host_[i][j] = NULL;
      }
      if (halo_peer_dev_) {
        delete halo_peer_dev_[i][j];
        halo_peer_dev_[i][j] = NULL;
      }
    }
  }
  delete[] halo_self_host_;
  halo_self_host_ = NULL;
  delete[] halo_peer_host_;
  halo_peer_host_ = NULL;
  delete[] halo_peer_dev_;
  halo_peer_dev_ = NULL;
  
  if (dev_) {
    free(dev_);
    dev_ = NULL;
  }
  
  GridMPI::DeleteBuffers();
}

//! Copy out interior halo region.
void GridMPICUDAExp::CopyoutHalo(int dim, const Width2 &width,
                                 bool fw, bool diagonal) {
  if (halo_(dim, fw) == 0) {
    LOG_DEBUG() << "No " << (fw ? "forward" : "backward")
                << " halo for dimension " << dim << "\n";
    return;
  }
  LOG_DEBUG() << "Copyout\n";
  PSAssert(num_dims() <= 3);
  
  IndexArray halo_offset(0);
  if (fw) {
    halo_offset[dim] = halo_.bw[dim];
  } else {
    halo_offset[dim] = local_real_size_[dim] - halo_.fw[dim] - halo_.bw[dim];
  }
  LOG_DEBUG() << "halo offset: " << halo_offset << "\n";
    
  IndexArray halo_size = local_real_size_;
  halo_size[dim] = halo_(fw)[dim];
  LOG_DEBUG() << "halo size: " << halo_size << "\n";
  
  // First, copy out of CUDA device memory to CUDA pinned host memory
  void *host_buf = GetHaloSelfHost(dim, fw)->Get();

  
  buffer()->Copyout(elm_size_, num_dims(), local_real_size_,
                    host_buf, halo_offset, halo_size);
  LOG_DEBUG() << "host buf: "
              << ((float*)host_buf)[0] << ", "
              << ((float*)host_buf)[1] << ", "
              << ((float*)host_buf)[2] << ", "
              << ((float*)host_buf)[3] << "\n";
  LOG_DEBUG() << "Before Diag\n";
  
  // If diagonal needs to be exchanged, copies into the host buffer
  // from the buffers for the higher dimensions
  if (diagonal && (dim != (num_dims_ - 1))) {
    CopyDiag(dim, halo_, fw);
  }
}

void GridMPICUDAExp::CopyDiag(int dim, const Width2 &width, bool fw) {
  LOG_DEBUG() << "CopyDiag\n";
  switch (num_dims_) {
    case 2:
      LOG_ERROR() << "not supported\n";
      PSAbort(1); 
      break;
    case 3:
      if (dim == 0) {
        CopyDiag3D1(width, fw);
        PSAbort(1);
      } else if (dim == 1) {
        CopyDiag3D2(width, fw);
        return;
      } else {
        LOG_DEBUG() << "No need to copy diag\n";
        return;
      }
      break;
    default:
      LOG_ERROR() << "Unsupported dimension: " << num_dims_ << "\n";
      PSAbort(1);
  }
}

// Assumes remote 3rd dim halo are available locally at
// halo_self_host_[1] and halo_self_host_[2]
void GridMPICUDAExp::CopyDiag3D1(const Width2 &width, bool fw) {
  LOG_ERROR() << "Not implemented yet\n";
}

// Assumes remote 3rd dim halo are available locally at halo_self_host_[2]
void GridMPICUDAExp::CopyDiag3D2(const Width2 &width, bool fw) {
  const int num_dims = 3;
  const unsigned w = fw ? width.fw[1] : width.bw[1];  
  
  intptr_t halo_cuda_host = (intptr_t)GetHaloSelfHost(1, fw)->Get();
  
  IndexArray sg_offset;
  if (fw) {
    sg_offset[1] = width.bw[1];    
  } else {
    sg_offset[1] = local_real_size_[1] - width.fw[1] - width.bw[1];
  }
  IndexArray sg_size(local_real_size_[0], w, width.bw[2]);
  IndexArray halo_size(local_real_size_[0], local_real_size_[1], width.bw[2]);

  if (sg_size.accumulate(num_dims_)) {
    CopyoutSubgrid(elm_size_, num_dims_, GetHaloPeerHost(2, false)->Get(),
                   halo_size, (void*)halo_cuda_host, sg_offset, sg_size);
  }

  sg_size[2] += local_size_[2];
  halo_cuda_host += sg_size.accumulate(num_dims) * elm_size_;

  sg_size[2] = width.fw[2];
  halo_size[2] = width.fw[2];
  if (sg_size.accumulate(num_dims_)) {
    CopyoutSubgrid(elm_size_, num_dims_, GetHaloPeerHost(2, true)->Get(),
                   halo_size, (void*)halo_cuda_host, sg_offset, sg_size);
  }
  
}

//! Copy in halo region.
void GridMPICUDAExp::CopyinHalo(int dim, const Width2 &width,
                                bool fw, bool diagonal) {
  if (halo_(dim, fw) == 0) {
    LOG_DEBUG() << "No " << (fw ? "forward" : "backward")
                << " halo for dimension " << dim << "\n";
    return;
  }
  
  PSAssert(num_dims() <= 3);
  
  IndexArray halo_offset(0);  
  if (fw) {
    halo_offset[dim] = local_real_size_[dim] - halo_.fw[dim];
  } else {
    halo_offset[dim] = halo_.bw[dim] - width(fw)[dim];
    PSAssert(halo_.bw[dim] == width(fw)[dim]);
  }
  
  IndexArray halo_size = local_real_size_;
  halo_size[dim] = halo_(fw)[dim];

  // The buffer holding peer halo
  void *host_buf = GetHaloPeerHost(dim, fw)->Get();
  
  buffer()->Copyin(elm_size_, num_dims(), local_real_size_,
                   host_buf, halo_offset, halo_size);

  
}

// TODO (Reduction)
int GridMPICUDAExp::Reduce(PSReduceOp op, void *out) {
  LOG_ERROR() << "Not supported\n";
  PSAbort(1);
  return 0;
}
} // namespace runtime
} // namespace physis

