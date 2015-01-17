#include <stdio.h>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32

typedef struct {
  float p;
  float q;
} Point;

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel1(Point *g) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int z = threadIdx.z + blockIdx.z * blockDim.z;

   if (!(x > 0 && x < N-1 && y > 0 && y < N-1 && z > 0 && z < N-1)) {
     return;
   }

   float v = g[OFFSET(x, y, z)].p +
       g[OFFSET(x+1, y, z)].p +
       g[OFFSET(x-1, y, z)].p +
       g[OFFSET(x, y+1, z)].p +
       g[OFFSET(x, y-1, z)].p +
       g[OFFSET(x, y, z+1)].p +
       g[OFFSET(x, y, z-1)].p;
   g[OFFSET(x, y, z)].q = v;

   return;
}

void dump(Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f\n", input[i].p, input[i].q);
  }
}

int main(int argc, char *argv[]) {
  Point *g_h, *g_d;
  size_t nelms = N*N*N;
  g_h = (Point *)malloc(sizeof(Point) * nelms);
  assert(cudaSuccess ==
         cudaMalloc((void**)&g_d, sizeof(Point) * nelms));

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g_h[i].p = i;
    g_h[i].q = 0;
  }
    
  assert(cudaSuccess ==
         cudaMemcpy(g_d, g_h, sizeof(Point) * nelms,
                    cudaMemcpyHostToDevice));
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  kernel1<<<grid_dim, block_dim>>>(g_d);
  
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n",
            cudaGetErrorString(e));
    exit(1);
  }

  assert(cudaSuccess ==
         cudaMemcpy(g_h, g_d, sizeof(Point) * nelms,
                    cudaMemcpyDeviceToHost));

  dump(g_h);

  cudaDeviceReset();
  return 0;
}

