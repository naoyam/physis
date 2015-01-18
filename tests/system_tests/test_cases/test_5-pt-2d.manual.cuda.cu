#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET1D(x) (x)
#define OFFSET2D(x, y) ((x) + (y) * N)
#define OFFSET3D(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel(REAL *g1, REAL *g2) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;   

  if (x == 0 || x == N-1 ||
      y == 0 || y == N-1) return;

  float v =
      g1[OFFSET2D(x, y)] +
      g1[OFFSET2D(x-1, y)] + 
      g1[OFFSET2D(x+1, y)] +
      g1[OFFSET2D(x, y-1)] + 
      g1[OFFSET2D(x, y+1)];
  g2[OFFSET2D(x, y)] = v;
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g1d;
  REAL *g2d;
  size_t nelms = N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  cudaMalloc((void**)&g1d, sizeof(REAL) * nelms);
  cudaMalloc((void**)&g2d, sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }
    
  cudaMemcpy(g1d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  cudaMemcpy(g2d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);  
  
  dim3 block_dim(16, 16);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y);

  kernel<<<grid_dim, block_dim>>>(g1d, g2d);
  cudaMemcpy(g1, g2d, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g1);

  cudaDeviceReset();
  return 0;
}

