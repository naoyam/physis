#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET1D(x) (x)
#define OFFSET2D(x, y) ((x) + (y) * N)
#define OFFSET3D(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel(REAL *g1, REAL *g2,
                       REAL *i, REAL *j, REAL *k) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x == 0 || x == N-1 || y == 0 || y == N-1 ||
      z == 0 || z == N-1) return;

  float v =
      g1[OFFSET3D(x, y, z)] +
      g1[OFFSET3D(x-1, y, z)] * i[OFFSET1D(x-1)] + 
      g1[OFFSET3D(x+1, y, z)] * i[OFFSET1D(x+1)] +
      g1[OFFSET3D(x, y-1, z)] * j[OFFSET1D(y-1)] + 
      g1[OFFSET3D(x, y+1, z)] * j[OFFSET1D(y+1)] +
      g1[OFFSET3D(x, y, z-1)] * k[OFFSET1D(z-1)] + 
      g1[OFFSET3D(x, y, z+1)] * k[OFFSET1D(z+1)];
  g2[OFFSET3D(x, y, z)] = v;
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
  REAL *k, *kd;
  size_t nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  k = (REAL *)malloc(sizeof(REAL) * N);  
  cudaMalloc((void**)&g1d, sizeof(REAL) * nelms);
  cudaMalloc((void**)&g2d, sizeof(REAL) * nelms);
  cudaMalloc((void**)&kd, sizeof(REAL) * N);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }
    
  cudaMemcpy(g1d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  cudaMemcpy(g2d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);

  for (i = 0; i < N; ++i) {
    k[i] = 1 + (i%2); // 1 or 2    
  }
  cudaMemcpy(kd, k, sizeof(REAL) * N, cudaMemcpyHostToDevice);
  
  dim3 block_dim(16, 4, 1);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  kernel<<<grid_dim, block_dim>>>(g1d, g2d, kd, kd, kd);
  cudaMemcpy(g1, g2d, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g1);

  cudaDeviceReset();
  return 0;
}

