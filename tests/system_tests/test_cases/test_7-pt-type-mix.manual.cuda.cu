#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel(float *g1, double *g2,
                       int *c) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int z = threadIdx.z + blockIdx.z * blockDim.z;

   if (x == 0 || x == N-1 || y == 0 || y == N-1 ||
       z == 0 || z == N-1) return;
  
   float v = (double)g1[OFFSET(x, y, z)] +
       (double)g1[OFFSET(x+1, y, z)] +
       (double)g1[OFFSET(x-1, y, z)] +
       (double)g1[OFFSET(x, y+1, z)] +
       (double)g1[OFFSET(x, y-1, z)] +
       (double)g1[OFFSET(x, y, z-1)] +
       (double)g1[OFFSET(x, y, z+1)];
   g2[OFFSET(x, y, z)] = v * c[OFFSET(x, y, z)];
   return;
}

void dump(double *x) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", x[i]);
  }
}

#define halo_width (1)

int main(int argc, char *argv[]) {
  float *g1, *g1d;
  double *g2, *g2d;
  int *c, *cd;
  size_t nelms = N*N*N;
  g1 = (float *)malloc(sizeof(float) * nelms);
  g2 = (double *)malloc(sizeof(double) * nelms);
  c = (int *)malloc(sizeof(int) * nelms);    
  cudaMalloc((void**)&g1d, sizeof(float) * nelms);
  cudaMalloc((void**)&g2d, sizeof(double) * nelms);
  cudaMalloc((void**)&cd, sizeof(int) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = 0;
    c[i] = i % 10;
  }
    
  cudaMemcpy(g1d, g1, sizeof(float) * nelms, cudaMemcpyHostToDevice);
  cudaMemcpy(g2d, g2, sizeof(double) * nelms, cudaMemcpyHostToDevice);
  cudaMemcpy(cd, c, sizeof(int) * nelms, cudaMemcpyHostToDevice);    
  
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  kernel<<<grid_dim, block_dim>>>(g1d, g2d, cd);
  cudaMemcpy(g2, g2d, sizeof(double) * nelms, cudaMemcpyDeviceToHost);

  dump(g2);
  
  cudaDeviceReset();
  return 0;
}

