#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET1D(x) (x)
#define OFFSET3D(x, y, z) ((x) + (y) * N + (z) * N * N)

 __global__ void kernel(REAL *g1, REAL *g2) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;

   if (x == 0 || x == N-1) return;

   float v = g1[OFFSET1D(x-1)] + g1[OFFSET1D(x)] +
       g1[OFFSET1D(x+1)];
   g2[OFFSET1D(x)] = v;
   return;
 }

void dump(float *input) {
  int i;
  for (i = 0; i < N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g1d;
  REAL *g2d;
  size_t nelms = N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  cudaMalloc((void**)&g1d, sizeof(REAL) * nelms);
  cudaMalloc((void**)&g2d, sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }
    
  cudaMemcpy(g1d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  cudaMemcpy(g2d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);  
  
  dim3 block_dim(4);
  dim3 grid_dim(N/block_dim.x);

  kernel<<<grid_dim, block_dim>>>(g1d, g2d);
  cudaMemcpy(g1, g2d, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g1);

  cudaDeviceReset();
  return 0;
}

