// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/grid_mpi_cuda.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/buffer_cuda.h"
#include "runtime/grid_util.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace physis::runtime::performance;
using std::make_pair;

namespace physis {
namespace runtime {

//
// Grid
//
GridMPICUDA3D::GridMPICUDA3D(
    PSType type, int elm_size, int num_dims, const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &local_offset, const IndexArray &local_size,
    int attr):
    GridMPI(type, elm_size, num_dims, size, global_offset,
            local_offset, local_size,
            Width2(), // set zero halo width
            attr) {
  
  if (empty_) return;

  // These pointers are replaced with buffer substrate in this class
  delete[]  halo_self_fw_;
  halo_self_fw_ = NULL;
  delete[] halo_self_bw_;
  halo_self_bw_ = NULL;

  FixupBufferPointers();
}

GridMPICUDA3D *GridMPICUDA3D::Create(
    PSType type, int elm_size, int num_dims, const IndexArray &size,
    const IndexArray &global_offset,
    const IndexArray &local_offset, const IndexArray &local_size,
    int attr) {
  GridMPICUDA3D *gmc = new GridMPICUDA3D(type, elm_size, num_dims, size,
                                         global_offset,
                                         local_offset, local_size,
                                         attr);
  gmc->InitBuffers();
  return gmc;
}


GridMPICUDA3D::~GridMPICUDA3D() {
  DeleteBuffers();
}

void GridMPICUDA3D::DeleteBuffers() {
  if (empty_) return;
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (halo_self_cuda_) {
        delete halo_self_cuda_[i][j];
        halo_self_cuda_[i][j] = NULL;
      }
      if (halo_self_mpi_) {
        delete halo_self_mpi_[i][j];
        halo_self_mpi_[i][j] = NULL;
      }
      if (halo_peer_cuda_) {
        delete halo_peer_cuda_[i][j];
        halo_peer_cuda_[i][j] = NULL;
      }
      if (halo_peer_dev_) {
        delete halo_peer_dev_[i][j];
        halo_peer_dev_[i][j] = NULL;
      }
    }
    halo_peer_fw_[i] = NULL;
    halo_peer_bw_[i] = NULL;    
  }
  delete[] halo_self_cuda_;
  halo_self_cuda_ = NULL;
  delete[] halo_self_mpi_;
  halo_self_mpi_ = NULL;
  delete[] halo_peer_cuda_;
  halo_peer_cuda_ = NULL;
  delete[] halo_peer_dev_;
  halo_peer_dev_ = NULL;
  GridMPI::DeleteBuffers();
}


void GridMPICUDA3D::InitBuffers() {
  LOG_DEBUG() << "Initializing grid buffer\n";

  if (empty_) return;
  //data_buffer_ = new BufferCUDADev3D(num_dims(), elm_size());
  data_buffer_ = new BufferCUDADev();
  data_buffer_->Allocate(local_size());
  
#if USE_MAPPED  
  halo_self_cuda_ = new BufferCUDAHostMapped*[num_dims_][2];
#else
  halo_self_cuda_ = new BufferCUDAHost*[num_dims_][2];  
#endif
  halo_self_mpi_ = new BufferHost*[num_dims_][2];
  halo_peer_cuda_ = new BufferCUDAHost*[num_dims_][2];
  halo_peer_dev_ = new BufferCUDADev*[num_dims_][2];
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
#if USE_MAPPED      
      halo_self_cuda_[i][j] = new BufferCUDAHostMapped(elm_size());
#else
      halo_self_cuda_[i][j] = new BufferCUDAHost();
#endif      
      halo_self_mpi_[i][j] = new BufferHost();
      halo_peer_cuda_[i][j] = new BufferCUDAHost();
      halo_peer_dev_[i][j] = new BufferCUDADev();
    }
  }

  FixupBufferPointers();
}


void GridMPICUDA3D::SetCUDAStream(cudaStream_t strm) {
  static_cast<BufferCUDADev3D*>(data_buffer_[0])->strm() = strm;
  for (int i = 0; i < num_dims_; ++i) {
    for (int j = 0; j < 2; ++j) {
      halo_peer_dev_[i][j]->strm() = strm;
    }
  }
}

#if 0
// Assumes the dst address is in the device memory
void GridMPICUDA3D::Copyin(void *dst, const void *src, size_t size) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, size,
                            cudaMemcpyHostToDevice));
}

// Assumes the source address is in the device memory
void GridMPICUDA3D::Copyout(void *dst, const void *src, size_t size) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, src, size,
                            cudaMemcpyDeviceToHost));
}
#endif
std::ostream &GridMPICUDA3D::Print(std::ostream &os) const {
  os << "GridMPICUDA {"
     << "elm_size: " << elm_size_
     << ", size: " << size_
     << ", global offset: " << global_offset_
     << ", local offset: " << local_offset_
     << ", local size: " << local_size_
     << "}";
  return os;
}


void GridMPICUDA3D::CopyoutHalo(int dim, unsigned width, bool fw,
                                bool diagonal) {
  PSAssert(num_dims() <= 3);
  halo_has_diagonal_ = diagonal;

  int fw_idx = (fw) ? 1: 0;

  IndexArray halo_offset;
  if (!fw) halo_offset[dim] = local_size_[dim] - width;
  IndexArray halo_size = local_size_;
  halo_size[dim] = width;
  size_t linear_size = halo_size.accumulate(num_dims());

  // First, copy out of CUDA device memory to CUDA pinned host memory
#if USE_MAPPED
  BufferCUDAHostMapped *halo_cuda_host = halo_self_cuda_[dim][fw_idx];    
#else
  BufferCUDAHost *halo_cuda_host = halo_self_cuda_[dim][fw_idx];  
#endif
  BufferHost *halo_mpi_host = halo_self_mpi_[dim][fw_idx];
  halo_cuda_host->EnsureCapacity(linear_size);
  static_cast<BufferCUDADev3D*>(buffer())->Copyout(
      *halo_cuda_host, halo_offset, halo_size);

  // Next, copy out to the halo buffer from CUDA pinned host memory
  halo_mpi_host->EnsureCapacity(CalcHaloSize(dim, width, diagonal));
  if (dim == num_dims_ - 1 || !diagonal) {
    halo_cuda_host->Copyout(*halo_mpi_host,
                            IndexArray(halo_size.accumulate(num_dims_)));
    return;
  }

  switch (num_dims_) {
    case 2:
      //CopyoutHalo2D0(width, fw, *halo_buf);
      LOG_ERROR() << "not supported\n";
      PSAbort(1); 
      break;
    case 3:
      if (dim == 0) {
        CopyoutHalo3D0(width, fw);
      } else if (dim == 1) {
        CopyoutHalo3D1(width, fw);
      } else {
        LOG_ERROR() <<
            "This case should have been handled in the first block"
            " of this function.\n";
        PSAbort(1);
      }
      break;
    default:
      LOG_ERROR() << "Unsupported dimension: " << num_dims_ << "\n";
      PSAbort(1);
  }
}

// REFACTORING: This is almost identical to the parent class implementation. 
void GridMPICUDA3D::CopyoutHalo3D1(unsigned width, bool fw) {
  int nd = 3;
  int fw_idx = fw ? 1 : 0;
  // copy diag
  char *buf = (char*)halo_self_mpi_[1][fw_idx]->Get();
  IndexArray sg_offset;
  if (!fw) sg_offset[1] = local_size_[1] - width;
  IndexArray sg_size(local_size_[0], width, halo_bw_width_[2]);
  IndexArray halo_size(local_size_[0], local_size_[1],
                     halo_bw_width_[2]);
  // different  
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_cuda_[2][0]->Get(),
                 halo_size, buf, sg_offset, sg_size);
  buf += sg_size.accumulate(nd) * elm_size_;

  // copy halo
  sg_size[2] = local_size_[2];
  // different
  halo_self_cuda_[1][fw_idx]->Copyout(buf, IndexArray(sg_size.accumulate(nd)));
  buf += sg_size.accumulate(nd) * elm_size_;
  
  // copy diag
  sg_size[2] = halo_fw_width_[2];
  halo_size[2] = halo_fw_width_[2];
  CopyoutSubgrid(elm_size_, num_dims_, halo_peer_cuda_[2][1]->Get(),
                 halo_size, buf, sg_offset, sg_size);
}

// REFACTORING: This is almost identical to the parent class implementation. 
//void GridMPICUDA3D::CopyoutHalo3D0(unsigned width, int fw) {
void GridMPICUDA3D::CopyoutHalo3D0(unsigned width, bool fw) {
  int fw_idx = fw ? 1 : 0;
  char *buf = (char*)halo_self_mpi_[0][fw_idx]->Get();
  size_t line_size = elm_size_ * width;  
  size_t xoffset = fw ? 0 : local_size_[0] - width;
  char *halo_bw_1 = (char*)halo_peer_cuda_[1][0]->Get() + xoffset * elm_size_;
  char *halo_fw_1 = (char*)halo_peer_cuda_[1][1]->Get() + xoffset * elm_size_;
  char *halo_bw_2 = (char*)halo_peer_cuda_[2][0]->Get() + xoffset * elm_size_;
  char *halo_fw_2 = (char*)halo_peer_cuda_[2][1]->Get() + xoffset * elm_size_;
  //char *halo_0 = data_[0] + xoffset * elm_size_;
  char *halo_0 = (char*)halo_self_cuda_[0][fw_idx]->Get();
  for (PSIndex k = 0; k < (PSIndex)halo_bw_width_[2]+
           local_size_[2]+(PSIndex)halo_fw_width_[2]; ++k) {
    // copy from the 2nd-dim backward halo 
    char *t = halo_bw_1;
    for (unsigned j = 0; j < halo_bw_width_[1]; j++) {
      memcpy(buf, t, line_size);
      buf += line_size;      
      t += local_size_[0] * elm_size_;
    }
    halo_bw_1 += local_size_[0] * halo_bw_width_[1] * elm_size_;

    char **h;
    size_t halo_x_size;
    // select halo source
    if (k < (PSIndex)halo_bw_width_[2]) {
      h = &halo_bw_2;
      halo_x_size = local_size_[0] * elm_size_;
    } else if (k < (PSIndex)halo_bw_width_[2] + local_size_[2]) {
      h = &halo_0;
      halo_x_size = elm_size_ * width;
    } else {
      h = &halo_fw_2;
      halo_x_size = local_size_[0] * elm_size_;
    }
    t = *h;
    for (PSIndex j = 0; j < local_size_[1]; ++j) {
      memcpy(buf, t, line_size);
      buf += line_size;
      t += halo_x_size;
    }
    *h += halo_x_size * local_size_[1];

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

void GridMPICUDA3D::FixupBufferPointers() {
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
  
  dev_.p0 = data_;

  dev_.diag = halo_has_diagonal();
  LOG_VERBOSE() << "Diag: " << dev_.diag << "\n";
  if (data_buffer_) {
    dev_.pitch = static_cast<BufferCUDADev3D*>(buffer())
        ->GetPitch() / elm_size();
    LOG_DEBUG() << "Pitch: " << dev_.pitch << "\n";
    for (int i = 0; i < num_dims(); ++i) {
      dev_.dim[i]  = size()[i];
      dev_.local_size[i] = local_size()[i];
      dev_.local_offset[i] = local_offset()[i];      
      halo_peer_fw_[i] = (char*)halo_peer_dev_[i][1]->Get();
      halo_peer_bw_[i] = (char*)halo_peer_dev_[i][0]->Get();

      for (int j = 0; j < 2; ++j) {
        dev_.halo[i][j] = halo_peer_dev_[i][j]->Get();
      }
      dev_.halo_width[i][1] = halo_fw_width()[i];
      dev_.halo_width[i][0] = halo_bw_width()[i];
    }
  }
}

#ifdef DEPRECATED
void GridMPICUDA3D::EnsureRemoteGrid(const IndexArray &local_offset,
                                     const IndexArray &local_size) {
  if (remote_grid_ == NULL) {
    remote_grid_ = GridMPICUDA3D::Create(type_, elm_size_, num_dims_,
                                         size_, false, global_offset_,
                                         local_offset, local_size,
                                         0);
  } else {
    remote_grid_->Resize(local_offset, local_size);
  }
}
#endif

#ifdef CHECKPOINT_ENABLED
void GridMPICUDA3D::Save() {
  LOG_DEBUG() << "Saving grid\n";
  Buffer *buf = static_cast<BufferCUDADev3D*>(
      data_buffer_[0]);
  // Copyout
  void *data = buf->Copyout();
  SaveToFile(data, buf->GetLinearSize());
  free(data);
  DeleteBuffers();
}

void GridMPICUDA3D::Restore() {
  LOG_DEBUG() << "Restoring grid\n";
  LOG_DEBUG() << "Initializing buffer\n";
  InitBuffers();
  LOG_DEBUG() << "Buffer initialized\n";
  // Read from a file
  BufferCUDADev3D *buf = static_cast<BufferCUDADev3D*>(
      data_buffer_[0]);
  void *data = malloc(buf->GetLinearSize());
  RestoreFromFile(data, buf->GetLinearSize());
  // Copyin
  buf->Copyin(data, IndexArray((PSIndex)0), buf->size());
}
#endif

} // namespace runtime
} // namespace physis

