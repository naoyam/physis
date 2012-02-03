#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)
#define PSGridGet(g, x, y, z) ((g)[OFFSET(x, y, z)])

__global__ void kernel(REAL *g1, REAL *g2) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  float c, w, e, n, s, b, t;
  c = PSGridGet(g1, x, y, z);
  if (x == 0)    w = c; else w = PSGridGet(g1, x-1, y, z);
  if (x == N-1) e = c ; else e = PSGridGet(g1, x+1, y, z);
  if (y == 0)    n = c ; else n=PSGridGet(g1, x, y-1, z);
  if (y == N-1) s= c ; else s=PSGridGet(g1, x, y+1, z);
  if (z == 0)    b= c ; else b=PSGridGet(g1, x, y, z-1);
  if (z == N-1) t= c ; else t=PSGridGet(g1, x, y, z+1);
  g2[OFFSET(x, y, z)] = c + w + e + s + n + b + t;  
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g1d;
  REAL *g2d;
  size_t nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  cudaMalloc((void**)&g1d, sizeof(REAL) * nelms);
  cudaMalloc((void**)&g2d, sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }
    
  cudaMemcpy(g1d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  cudaMemcpy(g2d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);  
  
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  kernel<<<grid_dim, block_dim>>>(g1d, g2d);
  cudaMemcpy(g1, g2d, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g1);

  cudaThreadExit();
  return 0;
}

