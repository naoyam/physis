// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/buffer_cuda.h"
#include "runtime/runtime_common_cuda.h"

#include <cuda_runtime.h>

// TODO: This can depend on many things. Need to be refined.
#define CUDA_MEMCPY_ASYNC_SIZE (64 << 10) // 64KB

namespace physis {
namespace runtime {

BufferCUDAHost::BufferCUDAHost()
    : Buffer(BufferCUDAHost::FreeChunk) {
}

BufferCUDAHost::~BufferCUDAHost() {
}

void BufferCUDAHost::FreeChunk(void *ptr) {
  CUDA_SAFE_CALL(cudaFreeHost(ptr));
  return;
}

void *BufferCUDAHost::GetChunk(size_t size) {
  if (size == 0) return NULL;
  void *ptr = NULL;  
  CUDA_SAFE_CALL(cudaMallocHost(&ptr, size));
  return ptr;
}

#if 0
//
// BufferCUDAHostMapped
//

BufferCUDAHostMapped::BufferCUDAHostMapped(size_t elm_size)
    : Buffer(elm_size), dev_ptr_(NULL) {
  mpi_buf_ = new BufferHost(elm_size);
  deleter_ = BufferCUDAHostMapped::DeleteChunk;
}

BufferCUDAHostMapped::BufferCUDAHostMapped(int num_dims,  size_t elm_size)
    : Buffer(num_dims, elm_size), dev_ptr_(NULL) {
  mpi_buf_ = new BufferHost(num_dims, elm_size);
  deleter_ = BufferCUDAHostMapped::DeleteChunk;  
}

BufferCUDAHostMapped::~BufferCUDAHostMapped() {
  delete mpi_buf_;
}

void BufferCUDAHostMapped::DeleteChunk(void *ptr) {
  CUDA_SAFE_CALL(cudaFreeHost(ptr));
  return;
}

void *BufferCUDAHostMapped::GetChunk(const IndexArray &size) {
  void *ptr = NULL;
  if (size.accumulate(num_dims_) > 0) {
    LOG_INFO() << "Trying to allocate host pinned memory of "
	       << GetLinearSize(size) << " bytes.\n";
    CUDA_SAFE_CALL(cudaHostAlloc(&ptr, GetLinearSize(size),
                                 cudaHostAllocMapped));
  }
  return ptr;
}

void BufferCUDAHostMapped::Allocate(int num_dims, size_t elm_size,
                                    const IndexArray &size) {
  Delete();
  if (size.accumulate(num_dims)) {
    num_dims_ = num_dims;
    elm_size_ = elm_size;
    buf_ = GetChunk(size);
    CUDA_SAFE_CALL(cudaHostGetDevicePointer(&dev_ptr_, buf_, 0));
  }
  size_ = size;
}


void BufferCUDAHostMapped::Copyin(const void *buf, const IndexArray &offset,
                            const IndexArray &size) {
  EnsureCapacity(offset+size);
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(Get(), buf, GetLinearSize(size));
}

void BufferCUDAHostMapped::Copyin(const BufferHost &buf, const IndexArray &offset,
                            const IndexArray &size) {
  Copyin(buf.Get(), offset, size);
}

void BufferCUDAHostMapped::Copyout(void *buf, const IndexArray &offset,
                             const IndexArray &s) {
  PSAssert(offset + s <= size());
  // Offset access is not yet supported.
  PSAssert(offset == 0);
  memcpy(buf, Get(), GetLinearSize(s));
}

void BufferCUDAHostMapped::Copyout(BufferHost &buf,
                             const IndexArray &offset,
                             const IndexArray &size) {
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  Copyout(buf.Get(), offset, size);
}


void BufferCUDAHostMapped::MPIRecv(int src, MPI_Comm comm,
                             const IndexArray &offset,
                             const IndexArray &size) {
  mpi_buf_->MPIRecv(src, comm, IndexArray(), size);
  Copyin(*mpi_buf_, offset, size);
  //mpi_buf_->Delete();
}

void BufferCUDAHostMapped::MPISend(int dst, MPI_Comm comm,
                             const IndexArray &offset,
                             const IndexArray &size) {
  Copyout(*mpi_buf_, offset, size);
  mpi_buf_->MPISend(dst, comm, IndexArray(), size);
  //mpi_buf_->Delete();
}
#endif

//
// BufferCUDADev
//
BufferCUDADev::BufferCUDADev()
    : Buffer(BufferCUDADev::FreeChunk) {
}

BufferCUDADev::~BufferCUDADev() {
}

void *BufferCUDADev::GetChunk(size_t size) {
  if (size == 0) return NULL;
  void *p = NULL;
  CUDA_SAFE_CALL(cudaMalloc(&p, size));
  return p;
}

void BufferCUDADev::FreeChunk(void *ptr) {
  CUDA_SAFE_CALL(cudaFree(ptr));
}

#if 0
//
// BufferCUDADev3D
// 
  
BufferCUDADev3D::BufferCUDADev3D(int num_dims,   size_t elm_size)
    : Buffer(num_dims, elm_size), strm_(0) {
  pinned_buf_ = new BufferCUDAHost(num_dims, elm_size);
  mapped_buf_ = new BufferCUDAHostMapped(num_dims, elm_size);  
  deleter_ = BufferCUDADev3D::DeleteChunk;
}

BufferCUDADev3D::~BufferCUDADev3D() {
  delete pinned_buf_;
  delete mapped_buf_;
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

void BufferCUDADev3D::Allocate(int num_dims, size_t elm_size,
                               const IndexArray &size) {
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

/*void BufferCUDADev3D::Allocate(const IndexArray &size) {
  Allocate(num_dims_, elm_size_, size);
  }*/

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
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
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

// REMEMBER to call cudaSetDeviceFlags(cudaDeviceMapHost)
template <typename  T>
__global__ void BufferCUDADev3D_copyout_kernel(
    const T *src, size_t xpos,  size_t ypos, size_t zpos,
    size_t xextent, size_t yextent, size_t zextent,
    size_t xdim, size_t ydim,  T *dst) {
  T *dst_ptr = dst;
  for (size_t k = blockIdx.x; k < zextent; k += gridDim.x) {
    //for (size_t k = blockIdx.x; k == blockIdx.x; k += gridDim.x) {
    size_t offset_k = xdim * ydim * (k + zpos);
    T *dst_ptr_k = dst_ptr + xextent * yextent * k;
    for (size_t j = threadIdx.y; j < yextent; j += blockDim.y) {
      //for (size_t j = threadIdx.y; j == threadIdx.y; j += blockDim.y) {    
      size_t offset_j = offset_k + xdim * (j + ypos);
      T *dst_ptr_j = dst_ptr_k + xextent * j;
      for (size_t i = threadIdx.x; i < xextent; i += blockDim.x) {
      //for (size_t i = threadIdx.x; i == threadIdx.x; i += blockDim.x) {
        size_t offset_i = offset_j + (i + xpos);
        dst_ptr_j[i] = src[offset_i];
      }
    }
  }
}

void BufferCUDADev3D::Copyout_Opt(BufferCUDAHostMapped &buf,
                                  const IndexArray &offset,
                                  const IndexArray &size) {
  // TODO: this must be refined.
  dim3 bdim;
  if (size[0] < 4) {
    bdim.x = size[0];
    bdim.y = 64;
  } else {
    bdim.x = 64;
    bdim.y = size[1];
  }    
  dim3 gdim(size[2]);
  if (elm_size_ == 4) {
    BufferCUDADev3D_copyout_kernel<float><<<gdim, bdim, 0, strm_>>>(
        (float*)Get(), offset[0], offset[1], offset[2],
        size[0], size[1], size[2],
        pp_.pitch / elm_size_, pp_.ysize, (float*)buf.GetDevPointer());
  } else if (elm_size_ == 8) {
    BufferCUDADev3D_copyout_kernel<double><<<gdim, bdim, 0, strm_>>>(
        (double*)Get(), offset[0], offset[1], offset[2],
        size[0], size[1], size[2],
        pp_.pitch / elm_size_, pp_.ysize, (double*)buf.GetDevPointer());
  } else {
    PSAssert(0);
  }
  CUDA_SAFE_CALL(cudaStreamSynchronize(strm_));
}

void BufferCUDADev3D::Copyout(BufferCUDAHostMapped &buf,
                              const IndexArray &offset,
                              const IndexArray &size) {
  buf.EnsureCapacity(num_dims_, elm_size_, size);
  PSAssert(offset + size <= this->size());

  // TODO: this must be refined.  
  if (size[0] < 4 || size[1] < 4) {
    LOG_VERBOSE() << "Copyout to mapped host memory\n";
    Copyout_Opt(buf, offset, size);
    return;
  }
  
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
      CUDA_SAFE_CALL(cudaDeviceSynchronize());      
    }
  }
}

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
      CUDA_SAFE_CALL(cudaDeviceSynchronize());            
    }
  }
}

  
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
#endif
  

} // namespace runtime
} // namespace physis
