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
    const __PSGridTypeInfo *type_info, int num_dims, const IndexArray &size,
    const IndexArray &global_offset, const IndexArray &local_offset,
    const IndexArray &local_size, const Width2 &halo, const Width2 *halo_member,
    int attr): GridMPI(type_info, num_dims, size, global_offset,
                       local_offset, local_size, halo, halo_member, attr),
               dev_(NULL) {
  if (empty_) return;

  // Hanlde primitive type as a single-member user-defined type
  if (type_ != PS_USER) {
    type_info_.num_members = 1;
    type_info_.members = new __PSGridTypeMemberInfo;
    type_info_.members->type = type_info_.type;
    type_info_.members->size = type_info_.size;
    type_info_.members->rank = 0;
  }
  
  // NOTE: Take maximum of halo width for all members. This is to
  // reduce the size of grid device type. Otherwise, each member has
  // to have its own size info.
  Width2 max_width = halo_member_[0];
  for (int i = 1; i < num_members(); ++i) {
    max_width.fw = std::max(max_width.fw, halo_member_[1].fw);
    max_width.bw = std::max(max_width.bw, halo_member_[1].bw);
  }
  for (int i = 0; i < num_members(); ++i) {
    halo_member_[i] = max_width;
  }
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
  __PSGridTypeInfo info = {type, elm_size, 0, NULL};
  return GridMPICUDAExp::Create(&info, num_dims, size, global_offset,
                                local_offset, local_size, halo,
                                NULL, attr);
}  

GridMPICUDAExp *GridMPICUDAExp::Create(
      const __PSGridTypeInfo *type_info,
      int num_dims,  const IndexArray &size,
      const IndexArray &global_offset, const IndexArray &local_offset,
      const IndexArray &local_size, const Width2 &halo,
      const Width2 *halo_member,
      int attr) {
  GridMPICUDAExp *g = new GridMPICUDAExp(
      type_info, num_dims, size, global_offset, local_offset,
      local_size, halo, halo_member, attr);
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
void *MallocPSGridDevType(int dim, int num_members) {
  size_t s = 0;
  switch (dim) {
    case 1:
      s = sizeof(__PSGrid1D_dev_m);
      break;
    case 2:
      s = sizeof(__PSGrid2D_dev_m);
      break;
    case 3:
      s = sizeof(__PSGrid3D_dev_m);
      break;
    default:
      LOG_ERROR() << "Dimension, " << dim << ", not supported\n";
      PSAbort(1);
  }
  // account for multiple members
  s += sizeof(void*) * (num_members - 1);
  void *p = malloc(s);
  PSAssert(p);
  return p;
}

}

void GridMPICUDAExp::InitBuffers() {
  LOG_DEBUG() << "Initializing grid buffer\n";

  if (empty_) return;
  data_buffer_ = NULL;
  data_buffer_m_ = new BufferCUDADev*[num_members()];
  for (int i = 0; i < num_members(); ++i) {
    data_buffer_m_[i] = new BufferCUDADev();
    data_buffer_m_[i]->Allocate(GetLocalBufferRealSize(i));
  }
  
  halo_self_host_ = new BufferCUDAHost*[num_dims_*num_members()][2];
  halo_peer_host_ = new BufferCUDAHost*[num_dims_*num_members()][2];
  for (int k = 0; k < num_members(); ++k) {  
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < 2; ++j) {
        size_t halo_size =
            CalcHaloSize(i, halo(k)(j==1)[i], false) *
            elm_size(k);
        BufferCUDAHost *self_host = NULL;
        BufferCUDAHost *peer_host = NULL;
        if (halo_size) {
          self_host = new BufferCUDAHost();
          self_host->Allocate(halo_size);
          peer_host = new BufferCUDAHost();
          peer_host->Allocate(halo_size);
        }
        GetHaloSelfHost(i, j == 1, k) = self_host;
        GetHaloPeerHost(i, j == 1, k) = peer_host;
      }
    }
  }

  // buffer to be sent as a pass-by-value parameter for CUDA kernel
  // calls
  PSAssert(dev_ == NULL);
  dev_ = MallocPSGridDevType(num_dims_, num_members());

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
  for (int i = 0; i < g.num_dims(); ++i) {
    dev_ptr->dim[i] = g.size()[i];
    // Pass the real size, not logically assigned size. Note that
    // since halo size is the same in all members, any member ID
    // shoudl work.
    dev_ptr->local_size[i] = g.local_real_size(0)[i];
    dev_ptr->local_offset[i] = g.local_real_offset(0)[i];
  }
  for (int i = 0; i < g.num_members(); ++i) {
    dev_ptr->p[i] = g.buffer(i)->Get();
  }
}

}

void GridMPICUDAExp::FixupBufferPointers() {
  LOG_VERBOSE() << "Fixup buffer ptrs\n";
  
  switch (num_dims_) {
    case 1: FixupDevBuffer<__PSGrid1D_dev_m>(*this);
      break;
    case 2: FixupDevBuffer<__PSGrid2D_dev_m>(*this);
      break;
    case 3: FixupDevBuffer<__PSGrid3D_dev_m>(*this);
      break;
    default:
      LOG_ERROR() << "Dimension, " << num_dims_ << ", not supported\n";
      PSAbort(1);
  }
}

void GridMPICUDAExp::DeleteBuffers() {
  if (empty_) return;
  for (int k = 0; k < num_members(); ++k) {
    for (int i = 0; i < num_dims_; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (halo_self_host_) {
          delete GetHaloSelfHost(i, j==1, k);
          GetHaloSelfHost(i, j==1, k) = NULL;
        }
        if (halo_peer_host_) {
          delete GetHaloPeerHost(i, j==1, k);
          GetHaloPeerHost(i, j==1, k) = NULL;
        }
      }
    }
  }
  delete[] halo_self_host_;
  halo_self_host_ = NULL;
  delete[] halo_peer_host_;
  halo_peer_host_ = NULL;
  
  if (dev_) {
    free(dev_);
    dev_ = NULL;
  }

  for (int i = 0; i < num_members(); ++i) {
    delete data_buffer_m_[i];
  }
  delete[] data_buffer_m_;
  
  GridMPI::DeleteBuffers();
}

void GridMPICUDAExp::CopyoutHalo(int dim, const Width2 &width,
                                 bool fw, bool diagonal) {
  for (int i = 0; i < num_members(); ++i) {
    CopyoutHalo(dim, width, fw, diagonal, i);
  }
}

//! Copy out interior halo region.
void GridMPICUDAExp::CopyoutHalo(int dim, const Width2 &width,
                                 bool fw, bool diagonal, int member) {
  if (halo(member)(dim, fw) == 0) {
    LOG_DEBUG() << "No " << (fw ? "forward" : "backward")
                << " halo for dimension " << dim << "\n";
    return;
  }
  LOG_DEBUG() << "Copyout\n";
  PSAssert(num_dims() <= 3);

  IndexArray halo_offset(0);
  if (fw) {
    halo_offset[dim] = halo(member).bw[dim];
  } else {
    halo_offset[dim] = local_size_[dim];
  }
  LOG_DEBUG() << "halo offset: " << halo_offset << "\n";
    
  IndexArray halo_size = local_real_size(member);
  halo_size[dim] = halo(member)(fw)[dim];
  LOG_DEBUG() << "halo size: " << halo_size << "\n";
  
  // First, copy out of CUDA device memory to CUDA pinned host memory
  void *host_buf = GetHaloSelfHost(dim, fw, member)->Get();

  
  buffer(member)->Copyout(elm_size(member), num_dims(),
                          local_real_size(member),
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
    CopyDiag(dim, fw, member);
  }
}

void GridMPICUDAExp::CopyDiag(int dim, bool fw, int member) {
  LOG_DEBUG() << "CopyDiag\n";
  switch (num_dims_) {
    case 2:
      LOG_ERROR() << "not supported\n";
      PSAbort(1); 
      break;
    case 3:
      if (dim == 0) {
        CopyDiag3D1(fw, member);
        PSAbort(1);
      } else if (dim == 1) {
        CopyDiag3D2(fw, member);
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
void GridMPICUDAExp::CopyDiag3D1(bool fw, int member) {
  LOG_ERROR() << "Not implemented yet\n";
}

// Assumes remote 3rd dim halo are available locally at halo_self_host_[2]
void GridMPICUDAExp::CopyDiag3D2(bool fw, int member) {
  const int num_dims = 3;
  const Width2 &width = halo(member);
  const unsigned w = fw ? width.fw[1] : width.bw[1];  
  
  intptr_t halo_cuda_host = (intptr_t)GetHaloSelfHost(1, fw, member)->Get();
  IndexArray sg_offset;
  if (fw) {
    sg_offset[1] = width.bw[1];    
  } else {
    sg_offset[1] = local_size_[1];
  }

  IndexArray local_real_size_m = local_real_size(member);
  IndexArray sg_size(local_real_size_m[0], w, width.bw[2]);
  IndexArray halo_size(local_real_size_m[0], local_real_size_m[1], width.bw[2]);

  if (sg_size.accumulate(num_dims_)) {
    CopyoutSubgrid(elm_size(member), num_dims_,
                   GetHaloPeerHost(2, false, member)->Get(),
                   halo_size, (void*)halo_cuda_host, sg_offset, sg_size);
  }

  sg_size[2] += local_size_[2];
  halo_cuda_host += sg_size.accumulate(num_dims) * elm_size(member);

  sg_size[2] = width.fw[2];
  halo_size[2] = width.fw[2];
  if (sg_size.accumulate(num_dims_)) {
    CopyoutSubgrid(elm_size(member), num_dims_,
                   GetHaloPeerHost(2, true, member)->Get(),
                   halo_size, (void*)halo_cuda_host, sg_offset, sg_size);
  }
  
}

void GridMPICUDAExp::CopyinHalo(int dim, const Width2 &width,
                                bool fw, bool diagonal) {
  for (int i = 0; i < num_members(); ++i) {
    CopyinHalo(dim, width, fw, diagonal, i);
  }  
}
//! Copy in halo region.
void GridMPICUDAExp::CopyinHalo(int dim, const Width2 &width,
                                bool fw, bool diagonal, int member) {
  if (halo_(dim, fw) == 0) {
    LOG_DEBUG() << "No " << (fw ? "forward" : "backward")
                << " halo for dimension " << dim << "\n";
    return;
  }
  
  PSAssert(num_dims() <= 3);

  const Width2 &h = halo(member);
  IndexArray local_real_size_m = local_real_size(member);
  
  IndexArray halo_offset(0);  
  if (fw) {
    halo_offset[dim] = local_real_size_m[dim] - h.fw[dim];
  } else {
    halo_offset[dim] = h.bw[dim] - width(fw)[dim];
    PSAssert(h.bw[dim] == width(fw)[dim]);
  }
  
  IndexArray halo_size = local_real_size_m;
  halo_size[dim] = h(fw)[dim];

  // The buffer holding peer halo
  void *host_buf = GetHaloPeerHost(dim, fw, member)->Get();
  
  buffer(member)->Copyin(elm_size(member), num_dims(), local_real_size_m,
                         host_buf, halo_offset, halo_size);
  
}

// TODO (Reduction)
int GridMPICUDAExp::Reduce(PSReduceOp op, void *out) {
  LOG_ERROR() << "Not supported\n";
  PSAbort(1);
  return 0;
}

void GridMPICUDAExp::Copyout(void *dst) {
  for (int i = 0; i < num_members(); ++i) {
    Copyout(dst, i);
    dst = (void*)((intptr_t)dst + buffer(i)->size());
  }
  return;
}

void GridMPICUDAExp::Copyout(void *dst, int member) {
  IndexArray offset(halo(member).bw);
  buffer(member)->Copyout(elm_size(member), num_dims(),
                          local_real_size(member),
                          dst, offset, local_size_);
  return;
}

void GridMPICUDAExp::Copyin(const void *src) {
  for (int i = 0; i < num_members(); ++i) {
    Copyin(src, i);
    src = (void*)((intptr_t)src + buffer(i)->size());
  }
  return;
}
void GridMPICUDAExp::Copyin(const void *src, int member) {
  IndexArray offset(halo(member).bw);
  buffer(member)->Copyin(elm_size(member), num_dims(),
                         local_real_size(member),
                         src, offset, local_size_);
  return;
}

} // namespace runtime
} // namespace physis

