// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/buffer_cuda.h"

#include <cuda_runtime.h>
#include <cutil.h>

#define CUDA_MEMCPY_ASYNC_SIZE (64 << 10) // 64KB

namespace physis {
namespace runtime {

BufferCUDAHost::BufferCUDAHost(size_t elm_size)
    : Buffer(elm_size) {
  mpi_buf_ = new BufferHost(elm_size);
  deleter_ = BufferCUDAHost::DeleteChunk;
}

BufferCUDAHost::BufferCUDAHost(int num_dims,  size_t elm_size)
    : Buffer(num_dims, elm_size) {
  mpi_buf_ = new BufferHost(num_dims, elm_size);
  deleter_ = BufferCUDAHost::DeleteChunk;  
}

BufferCUDAHost::~BufferCUDAHost() {
  delete mpi_buf_;
}

void BufferCUDAHost::DeleteChunk(void *ptr) {
  CUDA_SAFE_CALL(cudaFreeHost(ptr));
  return;
}

void *BufferCUDAHost::GetChunk(const IndexArray &size) {
  void *ptr = NULL;
  if (size.accumulate(num_dims_) > 0) {
    LOG_INFO() << "Trying to allocate host pinned memory of "
	       << GetLinearSize(size) << " bytes.\n";
    CUDA_SAFE_CALL(cudaMallocHost(&ptr, GetLinearSize(size)));
  }
  return ptr;
}

void BufferCUDAHost::Copyin(const void *buf, const IndexArray &offset,
                            const IndexArray &size) {
  EnsureCapacity(offset+size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(Get(), buf, GetLinearSize(size));
}

void BufferCUDAHost::Copyin(const BufferHost &buf, const IndexArray &offset,
                            const IndexArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferCUDAHost::Copyout(void *buf, const IndexArray &offset,
                             const IndexArray &s) {
  PSAssert(offset + s <= size());
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(buf, Get(), GetLinearSize(s));
}

void BufferCUDAHost::Copyout(BufferHost &buf,
                             const IndexArray &offset,
                             const IndexArray &size) {
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  Copyout(buf.Get(), offset, size);
}


void BufferCUDAHost::MPIRecv(int src, MPI_Comm comm,
                             const IndexArray &offset,
                             const IndexArray &size) {
  mpi_buf_->MPIRecv(src, comm, IndexArray(), size);
  Copyin(*mpi_buf_, offset, size);
  //mpi_buf_->Delete();
}

void BufferCUDAHost::MPISend(int dst, MPI_Comm comm,
                             const IndexArray &offset,
                             const IndexArray &size) {
  Copyout(*mpi_buf_, offset, size);
  mpi_buf_->MPISend(dst, comm, IndexArray(), size);
  //mpi_buf_->Delete();
}

//
// BufferCUDADev
//
BufferCUDADev::BufferCUDADev(size_t elm_size)
    : Buffer(elm_size), strm_(0) {
  pinned_buf_ = new BufferCUDAHost(elm_size);
  deleter_ = BufferCUDADev::DeleteChunk;
}

BufferCUDADev::BufferCUDADev(int num_dims, size_t elm_size)
    : Buffer(num_dims, elm_size), strm_(0) {
  pinned_buf_ = new BufferCUDAHost(num_dims, elm_size);
  deleter_ = BufferCUDADev::DeleteChunk;  
}

BufferCUDADev::~BufferCUDADev() {
  delete pinned_buf_;  
}

void BufferCUDADev::Copyin(const void *buf, const IndexArray &offset,
                           const IndexArray &size) {
  pinned_buf_->Copyin(buf, size);
  Copyin(*pinned_buf_, offset, size);  
}

void BufferCUDADev::Copyin(const BufferHost &buf, const IndexArray &offset,
                           const IndexArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferCUDADev::Copyin(const BufferCUDAHost &buf,
                           const IndexArray &offset,
                           const IndexArray &size) {
  PSAssert(offset == 0);
  EnsureCapacity(offset+size);
  if (strm_) {
    CUDA_SAFE_CALL(cudaMemcpyAsync(Get(), buf.Get(), GetLinearSize(size),
                                   cudaMemcpyHostToDevice, strm_));
    CUDA_SAFE_CALL(cudaStreamSynchronize(strm_));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy(Get(), buf.Get(), GetLinearSize(size),
                              cudaMemcpyHostToDevice));
    if ((size.accumulate(num_dims_) * elm_size_) <=
        CUDA_MEMCPY_ASYNC_SIZE) {
      CUDA_SAFE_THREAD_SYNC();
    }
  }
}

void BufferCUDADev::Copyout(void *buf, const IndexArray &offset,
                            const IndexArray &size) {
  Copyout(*pinned_buf_, offset, size);
  pinned_buf_->Copyout(buf, size);
}

void BufferCUDADev::Copyout(BufferHost &buf, const IndexArray &offset,
                            const IndexArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferCUDADev::Copyout(BufferCUDAHost &buf, const IndexArray &offset,
                            const IndexArray &size) {
  PSAssert(offset == 0);
  PSAssert(offset + size <= this->size());
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  if (strm_) {
    CUDA_SAFE_CALL(cudaMemcpyAsync(buf.Get(), Get(), GetLinearSize(size),
                                   cudaMemcpyDeviceToHost, strm_));
    CUDA_SAFE_CALL(cudaStreamSynchronize(strm_));    
  } else {
    CUDA_SAFE_CALL(cudaMemcpy(buf.Get(), Get(), GetLinearSize(size),
                              cudaMemcpyDeviceToHost));
    if ((size.accumulate(num_dims_) * elm_size_) <=
        CUDA_MEMCPY_ASYNC_SIZE) {
      CUDA_SAFE_THREAD_SYNC();
    }
  }
}

  
void BufferCUDADev::MPIRecv(int src, MPI_Comm comm, const IndexArray &offset,
                            const IndexArray &size) {
  // First, recv with the host pinned buffer (which also performs
  // internal copying between MPI and CUDA buffers.
  pinned_buf_->Buffer::MPIRecv(src, comm, size);
  // Then use cudaMemcpy to copy into the device memory
  Copyin(*pinned_buf_, offset, size);
}

void BufferCUDADev::MPISend(int dst, MPI_Comm comm, const IndexArray &offset,
                            const IndexArray &size) {
  Copyout(*pinned_buf_, offset, size);
  pinned_buf_->Buffer::MPISend(dst, comm, size);
}

void *BufferCUDADev::GetChunk(const IndexArray &size) {
  void *p = NULL;
  if (size.accumulate(num_dims_) >0)
    CUDA_SAFE_CALL(cudaMalloc(&p, GetLinearSize(size)));
  return p;
}

void BufferCUDADev::DeleteChunk(void *ptr) {
  if (ptr) {
    CUDA_SAFE_CALL(cudaFree(ptr));
  }
}

//
// BufferCUDADev3D
// 
  
BufferCUDADev3D::BufferCUDADev3D(int num_dims,   size_t elm_size)
    : Buffer(num_dims, elm_size), strm_(0) {
  pinned_buf_ = new BufferCUDAHost(num_dims, elm_size);
  deleter_ = BufferCUDADev3D::DeleteChunk;
}

BufferCUDADev3D::~BufferCUDADev3D() {
  delete pinned_buf_;
}

cudaPitchedPtr BufferCUDADev3D::GetChunk3D(const IndexArray &size) {
  // use cudaMalloc3D
  cudaPitchedPtr pp;
  if (size.accumulate(num_dims_)) {
    cudaExtent ext = make_cudaExtent(size[0] * elm_size_,
                                     size[1], size[2]);
    CUDA_SAFE_CALL(cudaMalloc3D(&pp, ext));
  } else {
    pp = make_cudaPitchedPtr(NULL, 0, 0, 0);
  }
  return pp;
}

void BufferCUDADev3D::DeleteChunk(void *ptr) {
  if (ptr) {
    CUDA_SAFE_CALL(cudaFree(ptr));
  }
}

void BufferCUDADev3D::Allocate(int num_dims, size_t elm_size, const IndexArray &size) {
  Delete();
  if (size.accumulate(num_dims)) {
    num_dims_ = num_dims;
    elm_size_ = elm_size;
    pp_ = GetChunk3D(size);
    buf_ = pp_.ptr;
    LOG_DEBUG() << "Pitch: " << pp_.pitch << "\n";
  }
  size_ = size;
}

void BufferCUDADev3D::Allocate(const IndexArray &size) {
  Allocate(num_dims_, elm_size_, size);
}

void BufferCUDADev3D::Copyin(const void *buf, const IndexArray &offset,
                             const IndexArray &size) {
  pinned_buf_->Copyin(buf, size);
  Copyin(*pinned_buf_, offset, size);  
}

void BufferCUDADev3D::Copyin(const BufferHost &buf, const IndexArray &offset,
                             const IndexArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferCUDADev3D::Copyin(const BufferCUDAHost &buf,
                             const IndexArray &offset,
                             const IndexArray &size) {
  EnsureCapacity(offset+size);
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(
      const_cast<void*>(buf.Get()), size[0] * elm_size_,
      size[0], size[1]);
  parms.dstPtr = pp_;
  parms.extent = make_cudaExtent(size[0] * elm_size_, size[1], size[2]);
  parms.dstPos = make_cudaPos(offset[0] * elm_size_, offset[1],
                              offset[2]);
  parms.kind = cudaMemcpyHostToDevice;
  if (strm_) {
    CUDA_SAFE_CALL(cudaMemcpy3DAsync(&parms, strm_));
    CUDA_SAFE_CALL(cudaStreamSynchronize(strm_));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy3D(&parms));
    if ((size.accumulate(num_dims_) * elm_size_) <=
        CUDA_MEMCPY_ASYNC_SIZE) {
      CUDA_SAFE_THREAD_SYNC();
    }
  }
}

void BufferCUDADev3D::Copyout(void *buf, const IndexArray &offset,
                              const IndexArray &size) {
  Copyout(*pinned_buf_, offset, size);
  pinned_buf_->Copyout(buf, size);
}

void BufferCUDADev3D::Copyout(BufferHost &buf, const IndexArray &offset,
                              const IndexArray &size) {
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  Copyout(buf.Get(), offset, size);
}

#if 0
void BufferCUDADev3D::Copyout(BufferCUDAHost &buf, const IndexArray &offset,
                              const IndexArray &size) {
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  PSAssert(offset + size <= this->size());  
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = pp_;
  parms.dstPtr = make_cudaPitchedPtr(
      buf.Get(), size[0] * elm_size_, size[0], size[1]);
  parms.extent = make_cudaExtent(size[0] * elm_size_, size[1], size[2]);
  parms.srcPos = make_cudaPos(offset[0] * elm_size_, offset[1],
                              offset[2]);
  parms.kind = cudaMemcpyDeviceToHost;
  if (strm_) {
    CUDA_SAFE_CALL(cudaMemcpy3DAsync(&parms, strm_));
    CUDA_SAFE_CALL(cudaStreamSynchronize(strm_));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy3D(&parms));
    if ((size.accumulate(num_dims_) * elm_size_) <=
        CUDA_MEMCPY_ASYNC_SIZE) {
      CUDA_SAFE_THREAD_SYNC();
    }
  }
}
#else
void BufferCUDADev3D::Copyout(BufferCUDAHost &buf, const IndexArray &offset,
                              const IndexArray &size) {
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  PSAssert(offset + size <= this->size());  
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = pp_;
  parms.dstPtr = make_cudaPitchedPtr(
      buf.Get(), size[0] * elm_size_, size[0], size[1]);
  parms.extent = make_cudaExtent(size[0] * elm_size_, size[1], size[2]);
  parms.srcPos = make_cudaPos(offset[0] * elm_size_, offset[1],
                              offset[2]);
  parms.kind = cudaMemcpyDeviceToHost;
  if (strm_) {
    CUDA_SAFE_CALL(cudaMemcpy3DAsync(&parms, strm_));
    CUDA_SAFE_CALL(cudaStreamSynchronize(strm_));
  } else {
    CUDA_SAFE_CALL(cudaMemcpy3D(&parms));
    if ((size.accumulate(num_dims_) * elm_size_) <=
        CUDA_MEMCPY_ASYNC_SIZE) {
      CUDA_SAFE_THREAD_SYNC();
    }
  }
}
#endif
  
void BufferCUDADev3D::MPIRecv(int src, MPI_Comm comm,
                              const IndexArray &offset,
                              const IndexArray &size) {
  // First, recv with the host pinned buffer (which also performs
  // internal copying between MPI and CUDA buffers.
  pinned_buf_->Buffer::MPIRecv(src, comm, size);
  // Then use cudaMemcpy to copy into the device memory
  Copyin(*pinned_buf_, offset, size);
}

void BufferCUDADev3D::MPISend(int dst, MPI_Comm comm,
                              const IndexArray &offset,
                              const IndexArray &size) {
  Copyout(*pinned_buf_, offset, size);
  pinned_buf_->Buffer::MPISend(dst, comm, size);
}

  

} // namespace runtime
} // namespace physis
