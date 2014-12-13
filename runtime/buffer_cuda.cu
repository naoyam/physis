// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/buffer_cuda.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/grid_util.h"

#include <cuda_runtime.h>

namespace physis {
namespace runtime {

BufferCUDAHost::BufferCUDAHost()
    : BufferHost(BufferCUDAHost::FreeChunk) {
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

//
// BufferCUDADev
//
BufferCUDADev::BufferCUDADev()
    : Buffer(BufferCUDADev::FreeChunk), temp_buffer_(NULL) {
}

BufferCUDADev::~BufferCUDADev() {
}

void *BufferCUDADev::GetChunk(size_t size) {
  if (size == 0) return NULL;
  void *p = NULL;
  CUDA_SAFE_CALL(cudaMalloc(&p, size));
  // zero-clear
  CUDA_SAFE_CALL(cudaMemset(p, 0, size));
  return p;
}

void BufferCUDADev::FreeChunk(void *ptr) {
  CUDA_SAFE_CALL(cudaFree(ptr));
}

__global__ void Pack2D(const int *src, int *dst,
                       PSIndex xdim, PSIndex ydim,
                       PSIndex xoff, PSIndex yoff,
                       PSIndex xextent, PSIndex yextent) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_idx = tid;
  PSIndex xidx = tid % xextent;
  tid = tid / xextent;
  PSIndex yidx = tid;
  if (yidx < yextent) {
    dst[dst_idx] = src[(xidx+xoff) + (yidx+yoff) * xdim];
  }
}

void BufferCUDADev::Copyout(void *dst, size_t size) {
  CUDA_SAFE_CALL(cudaMemcpy(dst, Get(), size,
                            cudaMemcpyDeviceToHost));
}

__global__ void Pack3D(const int *src, int *dst,
                       PSIndex xdim, PSIndex ydim, PSIndex zdim,
                       PSIndex xoff, PSIndex yoff, PSIndex zoff,
                       PSIndex xextent, PSIndex yextent, PSIndex zextent) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_idx = tid;
  PSIndex xidx = tid % xextent;
  tid = tid / xextent;
  PSIndex yidx = tid % yextent;
  tid = tid / yextent;
  PSIndex zidx = tid;
  if (zidx < zextent) {
    xidx += xoff; yidx += yoff; zidx += zoff;
    dst[dst_idx] = src[xidx + yidx * xdim + zidx * xdim * ydim];
  }
}

void BufferCUDADev::Copyout(size_t elm_size, int rank,
                            const IndexArray  &grid_size,
                            void *subgrid,
                            const IndexArray &subgrid_offset,
                            const IndexArray &subgrid_size) {
  LOG_DEBUG() << "BufferCUDADev copyout\n";
  LOG_DEBUG() << "grid size: " << grid_size << "\n";
  LOG_DEBUG() << "subgrid offset: " << subgrid_offset << "\n";
  LOG_DEBUG() << "subgrid size: " << subgrid_size << "\n";
  // check if packing is necessary
  bool packing = false;
  for (int i = 0; i < rank - 1; ++i) {
    if (grid_size[i] != subgrid_size[i]) {
      packing = true;
      break;
    }
  }

  intptr_t buf = (intptr_t)Get();
  
  if (!packing) {
    // call cudaMemcpy
    LOG_DEBUG() << "single cudaMemcpy is sufficient\n";
    size_t copy_size = subgrid_size.accumulate(rank) * elm_size;
    LOG_DEBUG() << "copy size: " << copy_size << "\n";
    intptr_t src = buf + GridCalcOffset(subgrid_offset, grid_size, rank)
        * elm_size;
    CUDA_SAFE_CALL(
        cudaMemcpy(subgrid, (void*)src, copy_size, cudaMemcpyDeviceToHost));
  } else {
    LOG_DEBUG() << "memcpy not usable\n";
    // pack
    const int th_size = 4;
    const int thread_block_size = 512;
  
    size_t total_size = elm_size * subgrid_size.accumulate(rank);
    int num_threads = total_size / th_size;
    // Assumption for simplicity of implementation
    PSAssert((total_size % th_size) == 0);
    int nblocks = num_threads / thread_block_size;
    if (num_threads % thread_block_size) ++nblocks;
    if (temp_buffer_ == NULL) temp_buffer_ = new BufferCUDADev();
    temp_buffer_->EnsureCapacity(total_size);

    if (rank == 3) {
      Pack3D<<<nblocks, thread_block_size>>>(
          (int*)Get(), (int*)(temp_buffer_->Get()),
          grid_size[0], grid_size[1], grid_size[2],
          subgrid_offset[0], subgrid_offset[1], subgrid_offset[2],
          subgrid_size[0], subgrid_size[1], subgrid_size[2]);
    } else if (rank == 2) {
      Pack2D<<<nblocks, thread_block_size>>>(
          (int*)Get(), (int*)(temp_buffer_->Get()),
          grid_size[0], grid_size[1], 
          subgrid_offset[0], subgrid_offset[1],
          subgrid_size[0], subgrid_size[1]);
    } else {
      LOG_ERROR() << "Unsupported\n";
      PSAbort(1);
    }

    // transfer
    CUDA_SAFE_CALL(cudaMemcpy(subgrid, temp_buffer_->Get(),
                              subgrid_size.accumulate(rank) * elm_size,
                              cudaMemcpyDeviceToHost));
  }
}

void BufferCUDADev::Copyin(const void *src, size_t size) {
  CUDA_SAFE_CALL(cudaMemcpy(Get(), src, size,
                            cudaMemcpyHostToDevice));
}

__global__ void Unpack2d(const int *src, int *dst,
                         PSIndex xdim, PSIndex ydim,
                         PSIndex xoff, PSIndex yoff,
                         PSIndex xextent, PSIndex yextent) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int src_idx = tid;
  PSIndex xidx = tid % xextent;
  tid = tid / xextent;
  PSIndex yidx = tid;
  if (yidx < yextent) {
    dst[(xidx+xoff) + (yidx+yoff) * xdim] = src[src_idx];
  }
}

__global__ void Unpack3d(const int *src, int *dst,
                         PSIndex xdim, PSIndex ydim, PSIndex zdim,
                         PSIndex xoff, PSIndex yoff, PSIndex zoff,
                         PSIndex xextent, PSIndex yextent, PSIndex zextent) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int src_idx = tid;
  PSIndex xidx = tid % xextent;
  tid = tid / xextent;
  PSIndex yidx = tid % yextent;
  tid = tid / yextent;
  PSIndex zidx = tid;
  if (zidx < zextent) {
    xidx += xoff; yidx += yoff; zidx += zoff;
    dst[xidx + yidx * xdim + zidx * xdim * ydim] = src[src_idx];
  }
}

void BufferCUDADev::Copyin(size_t elm_size, int rank,
                           const IndexArray  &grid_size,
                           const void *subgrid,
                           const IndexArray &subgrid_offset,
                           const IndexArray &subgrid_size) {
  // check if packing is necessary
  bool packing = false;
  for (int i = 0; i < rank - 1; ++i) {
    if (grid_size[i] != subgrid_size[i]) {
      packing = true;
      break;
    }
  }

  intptr_t buf = (intptr_t)Get();
  
  if (!packing) {
    LOG_DEBUG() << "single cudaMemcpy is sufficient\n";    
    // call cudaMemcpy
    intptr_t dst = buf + GridCalcOffset(subgrid_offset, grid_size, rank)
        * elm_size;
    CUDA_SAFE_CALL(
        cudaMemcpy((void*)dst, subgrid, subgrid_size.accumulate(rank) * elm_size,
                   cudaMemcpyHostToDevice));
  } else {
    LOG_DEBUG() << "memcpy not usable\n";
    // pack
    const int th_size = 4;
    const int thread_block_size = 512;
  
    size_t total_size = elm_size * subgrid_size.accumulate(rank);
    int num_threads = total_size / th_size;
    // Assumption for simplicity of implementation
    PSAssert((total_size % th_size) == 0);
    int nblocks = num_threads / thread_block_size;
    if (num_threads % thread_block_size) ++nblocks;
    if (temp_buffer_ == NULL) temp_buffer_ = new BufferCUDADev();
    temp_buffer_->EnsureCapacity(total_size);


    // transfer
    CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_->Get(), subgrid, 
                              subgrid_size.accumulate(rank) * elm_size,
                              cudaMemcpyHostToDevice));
    
    if (rank == 3) {
      Unpack3d<<<nblocks, thread_block_size>>>(
          (int*)(temp_buffer_->Get()), (int*)Get(),
          grid_size[0], grid_size[1], grid_size[2],
          subgrid_offset[0], subgrid_offset[1], subgrid_offset[2],
          subgrid_size[0], subgrid_size[1], subgrid_size[2]);
    } else if (rank == 2) {
      Unpack2d<<<nblocks, thread_block_size>>>(
          (int*)(temp_buffer_->Get()), (int*)Get(), 
          grid_size[0], grid_size[1], 
          subgrid_offset[0], subgrid_offset[1],
          subgrid_size[0], subgrid_size[1]);
    } else {
      LOG_ERROR() << "Unsupported\n";
      PSAbort(1);
    }
  }
}

} // namespace runtime
} // namespace physis
