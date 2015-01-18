// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/reduce_grid_mpi_cuda_exp.h"
#include "runtime/runtime_common_cuda.h"

#include "cub/cub.cuh"

#include <float.h>
#include <limits.h>

namespace physis {
namespace runtime {

namespace {

size_t reduction_buf_size = 0;
void *reduction_buf = NULL;

void *GetReductionBuf(size_t s) {
  if (reduction_buf_size < s) {
    if (reduction_buf) {
      cudaFree(reduction_buf);
    }
    CUDA_SAFE_CALL(cudaMalloc(&reduction_buf, s));
    reduction_buf_size = s;
  }
  return reduction_buf;
}

}

struct CubReduceOpProd {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a * b;
  }
};

template <class T, int OP>
__device__ T GetInitVal();

// float
template <>
__device__ float GetInitVal<float, PS_MAX>() {
  return FLT_MIN;
}
template <>
__device__ float GetInitVal<float, PS_MIN>() {
  return FLT_MAX;
}
template <>
__device__ float GetInitVal<float, PS_SUM>() {
  return 0.f;
}
template <>
__device__ float GetInitVal<float, PS_PROD>() {
  return 1.f;
}

// double
template <>
__device__ double GetInitVal<double, PS_MAX>() {
  return DBL_MIN;
}
template <>
__device__ double GetInitVal<double, PS_MIN>() {
  return DBL_MAX;
}
template <>
__device__ double GetInitVal<double, PS_SUM>() {
  return 0.0;
}
template <>
__device__ double GetInitVal<double, PS_PROD>() {
  return 1.0;
}

// int
template <>
__device__ int GetInitVal<int, PS_MAX>() {
  return INT_MIN;
}
template <>
__device__ int GetInitVal<int, PS_MIN>() {
  return INT_MAX;
}
template <>
__device__ int GetInitVal<int, PS_SUM>() {
  return 0;
}
template <>
__device__ int GetInitVal<int, PS_PROD>() {
  return 1;
}

// long
template <>
__device__ long GetInitVal<long, PS_MAX>() {
  return LONG_MIN;
}
template <>
__device__ long GetInitVal<long, PS_MIN>() {
  return LONG_MAX;
}
template <>
__device__ long GetInitVal<long, PS_SUM>() {
  return 0L;
}
template <>
__device__ long GetInitVal<long, PS_PROD>() {
  return 1L;
}

template <int OP, class T>
__device__ T GetCubReduceOp();

template <>
__device__ cub::Max GetCubReduceOp<PS_MAX, cub::Max>() {
  return cub::Max();
}
template <>
__device__ cub::Min GetCubReduceOp<PS_MIN, cub::Min>() {
  return cub::Min();
}
template <>
__device__ cub::Sum GetCubReduceOp<PS_SUM, cub::Sum>() {
  return cub::Sum();
}
template <>
__device__ CubReduceOpProd GetCubReduceOp<PS_PROD, CubReduceOpProd>() {
  return CubReduceOpProd();
}


template <class T, int op>
__device__ T ReduceBinary(T x, T y) {
  switch (op) {
    case PS_MAX:
      return (x > y) ? x : y;
    case PS_MIN:
      return (x < y) ? x : y;
    case PS_SUM:
      return x + y;
    case PS_PROD:
      return x * y;
    default:
      return 0;      
  }
}

template <class T, class BlockReduce, int op>
__device__ T ReduceBlock(T v) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T aggregate;
  switch (op) {
    case PS_MAX:
      aggregate = BlockReduce(temp_storage).Reduce(v, cub::Max());
      break;
    case PS_MIN:
      aggregate = BlockReduce(temp_storage).Reduce(v, cub::Min());
      break;
    case PS_SUM:
      aggregate = BlockReduce(temp_storage).Reduce(v, cub::Sum());
      break;
    case PS_PROD:
      aggregate = BlockReduce(temp_storage).Reduce(v, CubReduceOpProd());
      break;
    default:
      // should not reach here
      aggregate = 0;
  }
  return aggregate;
}


template <class T, int op, int BLOCK_DIM_X>
__global__ void ReduceGridMPICUDAExpKernelStage1(T *buf, T *dev_grid,
                                                 const int dim1, const int offset_bw1) {
  // Adapted from CUB reduction examples. See the CUB documentation.
  typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  T v = (i1 < dim1) ? dev_grid[i1 + offset_bw1] : GetInitVal<T, op>();
  T aggregate = ReduceBlock<T, BlockReduce, op>(v);
  if (threadIdx.x == 0) buf[blockIdx.x] = aggregate;  
  return;
}


template <class T, int op, int BLOCK_DIM_X>
__global__ void ReduceGridMPICUDAExpKernelStage2(T *buf, T *dev_grid,
                                                 const int dim1) {
  typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
  int i1 = threadIdx.x;
  T v = GetInitVal<T, op>();
  for (; i1 < dim1; i1 += blockDim.x) {
    T next = (i1 < dim1) ? dev_grid[i1] : GetInitVal<T, op>();
    v = ReduceBinary<T, op>(v, next);
  }
  T aggregate = ReduceBlock<T, BlockReduce, op>(v);
  if (threadIdx.x == 0) *buf = aggregate;  
  return;
}

template <class T, int op, int DIM>
struct ReduceGridMPICUDAExpFunctor {
  int operator()(void *buf, void *dev_grid,
                 const IndexArray &size,
                 const Width2 &width);
};

template <class T, int op>
struct ReduceGridMPICUDAExpFunctor<T, op, 1>{
  int operator()(void *buf, void *dev_grid,
                 const IndexArray &size,
                 const Width2 &width) {
    dim3 tblock(256);
    dim3 grid(ceil(((float)size[0]) / tblock.x));
    if (grid.x > 1) {
      T *reduction_buf = (T*)GetReductionBuf(grid.x * sizeof(T));      
      ReduceGridMPICUDAExpKernelStage1<T, op, 256><<<grid, tblock>>>(
          reduction_buf, (T*)dev_grid, size[0], width(0, false));
      ReduceGridMPICUDAExpKernelStage2<T, op, 256><<<1, tblock>>>(
          reduction_buf, reduction_buf, grid.x);
    } else {
      T *reduction_buf = (T*)GetReductionBuf(sizeof(T));      
      ReduceGridMPICUDAExpKernelStage1<T, op, 256><<<grid, tblock>>>(
          reduction_buf, (T*)dev_grid, size[0], width(0, false));
    }
    CUDA_SAFE_CALL(cudaMemcpy(buf, reduction_buf, sizeof(T), cudaMemcpyDeviceToHost));
    return size[0];
  }
};

template <class T, int op, int BLOCK_DIM_X, int THREAD_DIM_Y>
__global__ void ReduceGridMPICUDAExpKernelStage1(T *buf, T *dev_grid,
                                                 const int dim1, const int dim2,
                                                 const int offset_bw1,
                                                 const int offset_bw2,
                                                 const int dim_real1) {
  // Assumes a 1-D thread block
  typedef cub::BlockReduce<T, BLOCK_DIM_X> BlockReduce;
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * THREAD_DIM_Y;
  T v = GetInitVal<T, op>();
  size_t offset = i1 + offset_bw1 + (i2+offset_bw2) * dim_real1;
  if (i1 < dim1) {
    for (int y = 0; y < THREAD_DIM_Y; ++y) {
      if (i2 >= dim2) break;
      v = ReduceBinary<T, op>(v, dev_grid[offset]);
      ++i2;
      offset += dim_real1;
    }
  }
  T aggregate = ReduceBlock<T, BlockReduce, op>(v);
  if (threadIdx.x == 0) buf[blockIdx.x] = aggregate;  
  return;
}

template <class T, int op>
struct ReduceGridMPICUDAExpFunctor<T, op, 2>{
  int operator()(void *buf, void *dev_grid, const IndexArray &size,
                 const Width2 &width) {
    dim3 tblock(256);
    int y_thread_size = 16;
    dim3 grid(ceil(((float)size[0]) / tblock.x),
              ceil(((float)size[1]) / y_thread_size));
    T *reduction_buf = (T*)GetReductionBuf(grid.x * grid.y * sizeof(T));      
    ReduceGridMPICUDAExpKernelStage1<T, op, 256, 16><<<grid, tblock>>>(
        reduction_buf, (T*)dev_grid, size[0], size[1],
        width(0, false), width(1, false),
        width(0, false) + size[0] + width(0, true));
    ReduceGridMPICUDAExpKernelStage2<T, op, 256><<<1, tblock>>>(
        reduction_buf, reduction_buf, grid.x*grid.y);
    CUDA_SAFE_CALL(cudaMemcpy(buf, reduction_buf, sizeof(T), cudaMemcpyDeviceToHost));
    return size.accumulate(2);
  }
};

template <class T, int op, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void ReduceGridMPICUDAExpKernelStage1(
    T *buf, T *dev_grid,
    const int dim1, const int dim2, const int dim3,
    const int offset_bw1, const int offset_bw2, const int offset_bw3,
    const int dim_real1, const int dim_real2) {
  // Assumes a 2-D thread block. Each thread is responsible for
  // reducing a z-direction line of points.
  typedef cub::BlockReduce<T, BLOCK_DIM_X,
                           cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y> BlockReduce;
  int i1 = blockIdx.x * blockDim.x + threadIdx.x;
  int i2 = blockIdx.y * blockDim.y + threadIdx.y;
  int i3 = 0;
  T v = GetInitVal<T, op>();
  size_t offset = i1 + offset_bw1 + (i2+offset_bw2) * dim_real1
      + (i3 + offset_bw3) * dim_real1 * dim_real2;
  if (i1 < dim1 && i2 < dim2) {
    for (; i3 < dim3; ++i3) {
      v = ReduceBinary<T, op>(v, dev_grid[offset]);
      offset += dim_real1 * dim_real2;
    }
  }
  T aggregate = ReduceBlock<T, BlockReduce, op>(v);
  int grid_offset = blockIdx.x + blockIdx.y * gridDim.x;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    buf[grid_offset] = aggregate;
  }
  return;
}

template <class T, int op>
struct ReduceGridMPICUDAExpFunctor<T, op, 3>{
  int operator()(void *buf, void *dev_grid, const IndexArray &size,
                 const Width2 &width) {
    dim3 tblock(256, 4);
    dim3 grid(ceil(((float)size[0]) / tblock.x),
              ceil(((float)size[1]) / tblock.y));
    T *reduction_buf = (T*)GetReductionBuf(grid.x * grid.y * sizeof(T));      
    ReduceGridMPICUDAExpKernelStage1<T, op, 256, 4><<<grid, tblock>>>(
        reduction_buf, (T*)dev_grid, size[0], size[1], size[2],
        width(0, false), width(1, false), width(2, false),
        width(0, false) + size[0] + width(0, true),
        width(1, false) + size[1] + width(1, true)); 
    ReduceGridMPICUDAExpKernelStage2<T, op, 256><<<1, 256>>>(
        reduction_buf, reduction_buf, grid.x*grid.y);
    CUDA_SAFE_CALL(cudaMemcpy(buf, reduction_buf, sizeof(T), cudaMemcpyDeviceToHost));
    return size.accumulate(3);
  }
};

template <class T, int OP>
int ReduceGridMPICUDAExp(void *buf, void *dev_grid,
                         int dim, const IndexArray &size,
                         const Width2 &width) {
  int ret = 0;
  switch (dim) {
    case 1:
      ret = ReduceGridMPICUDAExpFunctor<T, OP, 1>()(buf, dev_grid, size, width);
      break;
    case 2:
      ret = ReduceGridMPICUDAExpFunctor<T, OP, 2>()(buf, dev_grid, size, width);
      break;
    case 3:
      ret = ReduceGridMPICUDAExpFunctor<T, OP, 3>()(buf, dev_grid, size, width);
      break;
    default:
      LOG_ERROR() << "Unsupported dimension: " << dim << "\n";
      PSAbort(1);
  }
  return ret;
}

template <class T>
int ReduceGridMPICUDAExp(void *buf, PSReduceOp op, void *dev_grid,
                          int dim, const IndexArray &size,
                          const Width2 &width) {
  int ret = 0;
  switch (op) {
    case PS_MAX:
      ret = ReduceGridMPICUDAExp<T, PS_MAX>(buf, dev_grid, dim, size, width);
      break;
    case PS_MIN:
      ret = ReduceGridMPICUDAExp<T, PS_MIN>(buf, dev_grid, dim, size, width);
      break;
    case PS_SUM:
      ret = ReduceGridMPICUDAExp<T, PS_SUM>(buf, dev_grid, dim, size, width);
      break;
    case PS_PROD:
      ret = ReduceGridMPICUDAExp<T, PS_PROD>(buf, dev_grid, dim, size, width);
      break;
    default:
      LOG_ERROR() << "Unsupported op: " << op << "\n";
      PSAbort(1);
  }
  return ret;
}



int ReduceGridMPICUDAExp(void *buf, PSType type, PSReduceOp op,
                         void *dev_grid, int dim, const IndexArray &size,
                          const Width2 &width) {
  int ret = 0;
  switch (type) {
    case PS_FLOAT:
      ret = ReduceGridMPICUDAExp<float>(buf, op, dev_grid, dim, size, width);
      break;
    case PS_DOUBLE:
      ret = ReduceGridMPICUDAExp<double>(buf, op, dev_grid, dim, size, width);
      break;
    case PS_INT:
      ret = ReduceGridMPICUDAExp<int>(buf, op, dev_grid, dim, size, width);
      break;
    case PS_LONG:
      ret = ReduceGridMPICUDAExp<long>(buf, op, dev_grid, dim, size, width);
      break;
    default:
      LOG_ERROR() << "Unsupported dimension: " << dim << "\n";
      PSAbort(1);
  }
  return ret;
}

} //namespace runtime
} //namespace runtime
