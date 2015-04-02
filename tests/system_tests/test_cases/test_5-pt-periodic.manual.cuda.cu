#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET(x, y) ((x) + (y) * N)

 __global__ void kernel(REAL *g1, REAL *g2) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   
   int xp = ((x - 1) + N) % N;
   int xn = (x + 1) % N;
   int yp = ((y - 1) + N) % N;
   int yn = (y + 1) % N;
   float v =
       g1[OFFSET(x, y)] +
       g1[OFFSET(xn, y)] +
       g1[OFFSET(xp, y)] +
       g1[OFFSET(x, yn)] +
       g1[OFFSET(x, yp)];
   g2[OFFSET(x, y)] = v;
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
  
  dim3 block_dim(4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y);

  kernel<<<grid_dim, block_dim>>>(g1d, g2d);
  kernel<<<grid_dim, block_dim>>>(g2d, g1d);
  
  cudaMemcpy(g1, g1d, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g1);

  cudaDeviceReset();
  return 0;
}

