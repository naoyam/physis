#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 8
#define ITER 1
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel(REAL *f1, REAL *f2,
                       int nx, int ny, int nz) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int xy = nx * ny;
  for (int k = 0; k < nz; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int n = (j == 0)        ? c : c - ny;
    int s = (j == ny-1)     ? c : c + ny;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = f1[c] + f1[w] + f1[e] + f1[s]
        + f1[n] + f1[b] + f1[t];
    c += xy;
  }
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
  int nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  cudaMalloc((void**)&g1d, sizeof(REAL) * nelms);
  cudaMalloc((void**)&g2d, sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }

  int nx = N, ny = N, nz = N;
    
  cudaMemcpy(g1d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  
  dim3 block_dim(4, 2);
  dim3 grid_dim(nx/block_dim.x, ny/block_dim.y);

  for (i = 0; i < ITER; ++i) {
    kernel<<<grid_dim, block_dim>>>(
        g1d, g2d, nx, ny, nz);
    REAL *t = g1d;
    g1d = g2d;
    g2d = t;
  }
  cudaMemcpy(g1, g1d, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g1);

  cudaDeviceReset();
  return 0;
}

