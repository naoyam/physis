#include "diffusion3d_cuda_temporal_blocking.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(c)                                       \
  do {                                                          \
    cudaError_t _e = c;                                         \
    if (_e != cudaSuccess) {                                    \
      fprintf(stderr, "Error: %s\n", cudaGetErrorString(_e));   \
    }                                                           \
  } while (0)

namespace diffusion3d {

#if 0
__global__ void diffusion_kernel_temporal_blocking_1st_half(
    REAL *f1, REAL *f2,
    int nx, int ny, int nz,
    REAL ce, REAL cw, REAL cn, REAL cs,
    REAL ct, REAL cb, REAL cc) {
  int i = (blockDim.x - 2) * blockIdx.x + threadIdx.x - 1;
  i = max(i, 0);
  i = min(i, nx-1);
  int j = (blockDim.y - 2) * blockIdx.y + threadIdx.y - 1;
  j = max(j, 0);
  j = min(j, ny-1);
  int c = i + j * nx;
  int xy = nx * ny;
  for (int k = 0; k < nz; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int n = (j == 0)        ? c : c - nx;
    int s = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy;
  }
  return;
}

__global__ void diffusion_kernel_temporal_blocking_2nd_half(
    REAL *f1, REAL *f2,
    int nx, int ny, int nz,
    REAL ce, REAL cw, REAL cn, REAL cs,
    REAL ct, REAL cb, REAL cc) {
  int i = (blockDim.x - 2) * blockIdx.x + min(threadIdx.x, blockDim.x - 3);
  i = min(i, nx-1);
  int j = (blockDim.y - 2) * blockIdx.y + min(threadIdx.y, blockDim.y - 3);
  j = min(j, ny-1);
  int c = i + j * nx;
  int xy = nx * ny;
  for (int k = 0; k < nz; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int n = (j == 0)        ? c : c - nx;
    int s = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
#if 0    
    if (threadIdx.x > 0 && threadIdx.x < (blockDim.x - 1) &&
        threadIdx.y > 0 && threadIdx.y < (blockDim.y - 1)) {
#endif      
      f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
          + cn * f1[n] + cb * f1[b] + ct * f1[t];
#if 0      
    }
#endif    
    c += xy;
  }
  return;
}

#endif

__global__ void diffusion_kernel_temporal_blocking(
    REAL *f1, REAL *f2, int nx, int ny, int nz,
    REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, REAL cb, REAL cc) {
  
  int i, j, c, sc;
  int i2, j2, c2, sc2;
  extern __shared__ REAL sb[];
  REAL *sb1 = sb;
  REAL *sb2 = sb + blockDim.x * blockDim.y;
  REAL *sb3 = sb + blockDim.x * blockDim.y * 2;
  i = (blockDim.x - 2) * blockIdx.x + threadIdx.x - 1;
  i = max(i, 0);
  i = min(i, nx-1);
  j = (blockDim.y - 2) * blockIdx.y + threadIdx.y - 1;
  j = max(j, 0);
  j = min(j, ny-1);
  c = i + j * nx;
  sc = threadIdx.x + threadIdx.y * blockDim.x;
  const int xy = nx * ny;

  i2 = (blockDim.x - 2) * blockIdx.x +
      min(threadIdx.x, blockDim.x - 3);
  i2 = min(i2, nx-1);  
  j2 = (blockDim.y - 2) * blockIdx.y +
      min(threadIdx.y, blockDim.y - 3);
  j2 = min(j2, ny-1);
  c2 = i2 + j2 * nx;  
  sc2 = (i2 % (blockDim.x-2)) + 1 + ((j2 % (blockDim.y-2)) + 1) * blockDim.x;
  
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int n = (j == 0)        ? c : c - nx;
  int s = (j == ny-1)     ? c : c + nx;
  int b = c;
  int t = c + xy;
  float v = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * f1[b] + ct * f1[t];
  sb2[sc] = v;
  c += xy;
  
  for (int k = 1; k < nz; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int n = (j == 0)        ? c : c - nx;
    int s = (j == ny-1)     ? c : c + nx;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    float v = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    sb3[sc] = v;
    c += xy;
  
    __syncthreads();

    w = (i2 == 0)        ? sc2 : sc2 - 1;
    e = (i2 == nx-1)     ? sc2 : sc2 + 1;
    n = (j2 == 0)        ? sc2 : sc2 - blockDim.x;
    s = (j2 == ny-1)     ? sc2 : sc2 + blockDim.x;
    REAL *bv = (k-1 == 0)        ? sb2 + sc2 : sb1 + sc2;
    REAL *tv = sb3 + sc2;    
    f2[c2] = cc * sb2[sc2] + cw * sb2[w] + ce * sb2[e] + cs * sb2[s]
        + cn * sb2[n] + cb * (*bv) + ct * (*tv);
    c2 += xy;

    __syncthreads();

    REAL *sb_tmp = sb1;
    sb1 = sb2;
    sb2 = sb3;    
    sb3 = sb_tmp;
  }

  w = (i2 == 0)        ? sc2 : sc2 - 1;
  e = (i2 == nx-1)     ? sc2 : sc2 + 1;
  n = (j2 == 0)        ? sc2 : sc2 - blockDim.x;
  s = (j2 == ny-1)     ? sc2 : sc2 + blockDim.x;
  REAL *bv = sb1 + sc2;
  REAL *tv = sb2 + sc2;
  f2[c2] = cc * sb2[sc2] + cw * sb2[w] + ce * sb2[e] + cs * sb2[s]
      + cn * sb2[n] + cb * (*bv) + ct * (*tv);

  return;
}

__global__ void diffusion_kernel_temporal_blocking2(
    REAL *f1, REAL *f2, int nx, int ny, int nz,
    REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, REAL cb, REAL cc) {
  
  int c, sc;
  int c2, sc2;
  extern __shared__ REAL sb[];
  REAL *sb1 = sb;
  REAL *sb2 = sb + blockDim.x * blockDim.y;
  REAL *sb3 = sb + blockDim.x * blockDim.y * 2;
  const int i = min(
      nx-1, max(0,
                (blockDim.x - 2) * blockIdx.x + threadIdx.x - 1));
  const int j =
      min(ny-1,
          max(0, (blockDim.y - 2) * blockIdx.y + threadIdx.y - 1));
  c = i + j * nx;
  sc = threadIdx.x + threadIdx.y * blockDim.x;
  const int xy = nx * ny;
  
  int w = (i == 0)        ? c : c - 1;
  int e = (i == nx-1)     ? c : c + 1;
  int n = (j == 0)        ? c : c - nx;
  int s = (j == ny-1)     ? c : c + nx;
  int b = c;
  int t = c + xy;
  float v = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * f1[b] + ct * f1[t];
  sb2[sc] = v;
  c += xy;
  w += xy;
  e += xy;
  n += xy;
  s += xy;
  t += xy;

  const int i2 = min(nx-1, (blockDim.x - 2) * blockIdx.x +
                     min(threadIdx.x, blockDim.x - 3));
  const int j2 = min(ny-1, (blockDim.y - 2) * blockIdx.y +
                     min(threadIdx.y, blockDim.y - 3));
  c2 = i2 + j2 * nx;  
  sc2 = (i2 % (blockDim.x-2)) + 1 + ((j2 % (blockDim.y-2)) + 1) * blockDim.x;

  int w2 = (i2 == 0)        ? sc2 : sc2 - 1;
  int e2 = (i2 == nx-1)     ? sc2 : sc2 + 1;
  int n2 = (j2 == 0)        ? sc2 : sc2 - blockDim.x;
  int s2 = (j2 == ny-1)     ? sc2 : sc2 + blockDim.x;

  {
    int k = 1;
    float v = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    sb3[sc] = v;
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;
    b += xy;
    t += xy;
  
    __syncthreads();

    f2[c2] = cc * sb2[sc2] + cw * sb2[w2] + ce * sb2[e2] + cs * sb2[s2]
        + cn * sb2[n2] + cb * sb2[sc2] + ct * sb3[sc2];
    c2 += xy;

    __syncthreads();

    REAL *sb_tmp = sb1;
    sb1 = sb2;
    sb2 = sb3;    
    sb3 = sb_tmp;
  }
  
  for (int k = 2; k < nz-1; ++k) {
    float v = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    sb3[sc] = v;
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;
    b += xy;
    t += xy;
  
    __syncthreads();

    f2[c2] = cc * sb2[sc2] + cw * sb2[w2] + ce * sb2[e2] + cs * sb2[s2]
        + cn * sb2[n2] + cb * sb1[sc2] + ct * sb3[sc2];
    c2 += xy;

    __syncthreads();

    REAL *sb_tmp = sb1;
    sb1 = sb2;
    sb2 = sb3;    
    sb3 = sb_tmp;
  }


  sb3[sc] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
      + cn * f1[n] + cb * f1[b] + ct * f1[c];
  
  __syncthreads();

  f2[c2] = cc * sb2[sc2] + cw * sb2[w2] + ce * sb2[e2] + cs * sb2[s2]
      + cn * sb2[n2] + cb * sb1[sc2] + ct * sb3[sc2];
  c2 += xy;

  __syncthreads();

  REAL *sb_tmp = sb1;
  sb1 = sb2;
  sb2 = sb3;    
  sb3 = sb_tmp;
  

  f2[c2] = cc * sb2[sc2] + cw * sb2[w2] + ce * sb2[e2] + cs * sb2[s2]
      + cn * sb2[n2] + cb * sb1[sc2] + ct * sb2[sc2];

  return;
}


void Diffusion3DCUDATemporalBlocking::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  CUDA_SAFE_CALL(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));
  assert(block_x_ > 2);
  assert(block_y_ > 2);
  dim3 block_dim(block_x_, block_y_);
  dim3 grid_dim(nx_ / (block_x_ - 2), ny_ / (block_y_ - 2), 1);
  if (nx_ % (block_x_ - 2)) ++grid_dim.x;
  if (ny_ % (block_y_ - 2)) ++grid_dim.y;
  size_t shared_size = sizeof(REAL) * block_dim.x * block_dim.y * 3;
  printf("Shared memory size: %ld bytes\n", shared_size);

  CUDA_SAFE_CALL(cudaEventRecord(ev1_));
  for (int i = 0; i < count; i += 2) {

#if 1    
    diffusion_kernel_temporal_blocking2<<<
        grid_dim, block_dim, shared_size>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    //CUDA_SAFE_CALL(cudaGetLastError());
    REAL *f_tmp = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = f_tmp;
     
#elif 0
    diffusion_kernel_temporal_blocking<<<
        grid_dim, block_dim, shared_size>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    //¡ÆCUDA_SAFE_CALL(cudaGetLastError());
    REAL *f_tmp = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = f_tmp;
    
#elif 0
    diffusion_kernel_temporal_blocking_1st_half<<<grid_dim, block_dim>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    diffusion_kernel_temporal_blocking_2nd_half<<<grid_dim, block_dim>>>
        (f2_d_, f1_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
#elif 0
    diffusion_kernel_temporal_blocking_1st_half<<<grid_dim, block_dim>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    diffusion_kernel_temporal_blocking_1st_half<<<grid_dim, block_dim>>>
        (f2_d_, f1_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
#else
    diffusion_kernel_temporal_blocking_2nd_half<<<grid_dim, block_dim>>>
        (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    diffusion_kernel_temporal_blocking_2nd_half<<<grid_dim, block_dim>>>
        (f2_d_, f1_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
#endif
  }
  CUDA_SAFE_CALL(cudaEventRecord(ev2_));
  CUDA_SAFE_CALL(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}


}
