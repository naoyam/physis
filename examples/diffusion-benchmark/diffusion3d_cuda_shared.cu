#include "diffusion3d_cuda.h"

#define CUDA_SAFE_CALL(c)                       \
  do {                                          \
    assert(c == cudaSuccess);                   \
  } while (0)

#define block_x (128)
#define block_y (2)

namespace diffusion3d {

__global__ void diffusion_kernel_shared(REAL *f1, REAL *f2,
                                        int nx, int ny, int nz,
                                        REAL ce, REAL cw, REAL cn, REAL cs,
                                        REAL ct, REAL cb, REAL cc) {
#if 1
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int xy = nx * ny;  
  __shared__ REAL sb[block_x * block_y];
  int c = i + j * nx;
  const int c1 = tid_x + tid_y * blockDim.x;
  REAL t1, t2, t3;
  t2 = t3 = f1[c];
  int w = (i == 0)        ? c1 : c1 - 1;
  int e = (i == nx-1)     ? c1 : c1 + 1;
  int n = (j == 0)        ? c1 : c1 - block_x;
  int s = (j == ny-1)     ? c1 : c1 + block_x;
  int bw = tid_x == 0 && i != 0;
  int be = tid_x == block_x-1 && i != nx - 1;
  int bn = tid_y == 0 && j != 0;
  int bs = tid_y == block_y-1 && j != ny - 1;
  //#pragma unroll 4
  for (int k = 0; k < nz; ++k) {
    t1 = t2;
    t2 = t3;
    sb[c1] = t2;    
    t3 = (k < nz-1) ? f1[c+xy] : t3;
    __syncthreads();
    REAL t = cc * t2 + cb * t1 + ct * t3;
    REAL v;
    v = bw ? f1[c-1] : sb[w];
    t += cw * v;
    v = be ? f1[c+1] : sb[e];
    t += ce * v;
    v = bs ? f1[c+nx] : sb[s];
    t += cs * v;
    v = bn ? f1[c-nx] : sb[n];
    t += cn * v;
    f2[c] = t;
    c += xy;
    __syncthreads();
  }
#else
  // prefetching
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  const int i = blockDim.x * blockIdx.x + tid_x;
  const int j = blockDim.y * blockIdx.y + tid_y;
  const int xy = nx * ny;  
  __shared__ REAL sb[block_x * block_y];
  int c = i + j * nx;
  const int c1 = tid_x + tid_y * blockDim.x;
  REAL t1, t2, t3, t4;
  t2 = t3 = f1[c];
  t4 = f1[c+xy];
  int w = (i == 0)        ? c1 : c1 - 1;
  int e = (i == nx-1)     ? c1 : c1 + 1;
  int n = (j == 0)        ? c1 : c1 - block_x;
  int s = (j == ny-1)     ? c1 : c1 + block_x;
  int bw = tid_x == 0 && i != 0;
  int be = tid_x == block_x-1 && i != nx - 1;
  int bn = tid_y == 0 && j != 0;
  int bs = tid_y == block_y-1 && j != ny - 1;
#pragma unroll 
  for (int k = 0; k < nz; ++k) {
    t1 = t2;
    t2 = t3;
    sb[c1] = t2;    
    t3 = t4;
    __syncthreads();
    t4 = (k < nz -2) ? f1[c+xy*2] : t4;
    REAL t = cc * t2 + cb * t1 + ct * t3;
    REAL v;
    v = bw ? f1[c-1] : sb[w];
    t += cw * v;
    v = be ? f1[c+1] : sb[e];
    t += ce * v;
    v = bs ? f1[c+nx] : sb[s];
    t += cs * v;
    v = bn ? f1[c-nx] : sb[n];
    t += cn * v;
    f2[c] = t;
    c += xy;
    __syncthreads();
  }
#endif
  return;
}

void Diffusion3DCUDAShared::InitializeBenchmark() {
  Diffusion3DCUDA::InitializeBenchmark();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(diffusion_kernel_shared,
                                        cudaFuncCachePreferShared));
}

void Diffusion3DCUDAShared::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(block_x, block_y, 1);
  dim3 grid_dim(nx_ / block_x, ny_ / block_y, 1);

  CUDA_SAFE_CALL(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    diffusion_kernel_shared<<<grid_dim, block_dim>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CUDA_SAFE_CALL(cudaEventRecord(ev2_));
  CUDA_SAFE_CALL(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

}

