#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel(REAL *g, int rb) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  x = x * 2 + (y + z + rb) % 2;

  int xp = ((x - 1) + N) % N;
  int xn = (x + 1) % N;
  int yp = ((y - 1) + N) % N;
  int yn = (y + 1) % N;
  int zp = ((z - 1) + N) % N;
  int zn = (z + 1) % N;
  float v =
      g[OFFSET(x, y, z)] +
      g[OFFSET(xn, y, z)] +
      g[OFFSET(xp, y, z)] +
      g[OFFSET(x, yn, z)] +
      g[OFFSET(x, yp, z)] +
      g[OFFSET(x, y, zn)] +
      g[OFFSET(x, y, zp)];
  g[OFFSET(x, y, z)] = v;
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g, *gd;
  size_t nelms = N*N*N;
  g = (REAL *)malloc(sizeof(REAL) * nelms);
  cudaMalloc((void**)&gd, sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g[i] = i;
  }
    
  cudaMemcpy(gd, g, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x/2, N/block_dim.y, N/block_dim.z);

  kernel<<<grid_dim, block_dim>>>(gd, 0);
  kernel<<<grid_dim, block_dim>>>(gd, 1);  
  cudaMemcpy(g, gd, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g);

  cudaDeviceReset();
  return 0;
}

