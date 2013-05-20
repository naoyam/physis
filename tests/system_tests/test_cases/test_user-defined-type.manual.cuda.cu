#include <stdio.h>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define ITER 10

typedef struct {
  float x;
  float y;
  float z;
} Point;

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel(Point *g1, Point *g2) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int z = threadIdx.z + blockIdx.z * blockDim.z;

   if (x == 0 || x == N-1 || y == 0 || y == N-1 ||
       z == 0 || z == N-1) return;

   float p = (g1[OFFSET(x, y, z)].x +
              g1[OFFSET(x+1, y, z)].x +
              g1[OFFSET(x-1, y, z)].x) / 3.3f;
   float q = (g1[OFFSET(x, y, z)].y +
              g1[OFFSET(x, y+1, z)].y +
              g1[OFFSET(x, y-1, z)].y +
              g1[OFFSET(x, y, z+1)].y +
              g1[OFFSET(x, y, z-1)].y) / 5.5f;
   float r = (g1[OFFSET(x, y, z)].z +
              g1[OFFSET(x, y, z+1)].z +
              g1[OFFSET(x, y, z-1)].z) / 3.3f;
   Point v = {p, q, r};
   g2[OFFSET(x, y, z)] = v;
   return;
}

void dump(Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f %f\n", input[i].x, input[i].y, input[i].z);
  }
}


#define halo_width (1)

int main(int argc, char *argv[]) {
  Point *g1, *g1d;
  Point *g2d;
  size_t nelms = N*N*N;
  g1 = (Point *)malloc(sizeof(Point) * nelms);
  cudaMalloc((void**)&g1d, sizeof(Point) * nelms);
  cudaMalloc((void**)&g2d, sizeof(Point) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i].x = i;
    g1[i].y = i+1;
    g1[i].z = i+2;
  }
    
  assert(cudaSuccess ==
         cudaMemcpy(g1d, g1, sizeof(Point) * nelms, cudaMemcpyHostToDevice));
  assert(cudaSuccess ==
         cudaMemcpy(g2d, g1, sizeof(Point) * nelms, cudaMemcpyHostToDevice));
  
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  for (i = 0; i < ITER/2; ++i) {
    kernel<<<grid_dim, block_dim>>>(g1d, g2d);
    kernel<<<grid_dim, block_dim>>>(g2d, g1d);
  }
  
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n",
            cudaGetErrorString(e));
    exit(1);
  }

  assert(cudaSuccess ==
         cudaMemcpy(g1, g1d, sizeof(Point) * nelms,
                    cudaMemcpyDeviceToHost));

  dump(g1);

  cudaDeviceReset();
  return 0;
}

