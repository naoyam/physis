#include <stdio.h>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32

typedef struct {
  float r;
  float i;
} Complex;

#define OFFSET(x, y, z) ((((x)+N)%N) + (((y)+N)%N) * N + (((z)+N)%N) * N * N)

__global__ void kernel1(Complex *g1, Complex *g2) {
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int z = threadIdx.z + blockIdx.z * blockDim.z;

   if (!(x < N && y < N && z < N)) {
     return;
   }

   Complex t = g1[OFFSET(x, y, z)];
   Complex t1 = g1[OFFSET(x+1, y, z)];
   Complex t2 = g1[OFFSET(x-1, y, z)];
   Complex t3 = g1[OFFSET(x, y+1, z)];
   Complex t4 = g1[OFFSET(x, y-1, z)];
   Complex t5 = g1[OFFSET(x, y, z+1)];
   Complex t6 = g1[OFFSET(x, y, z-1)];
   
   float r = t.r + t1.r + t2.r + t3.r + t4.r + t5.r + t6.r;
   float i = t.i + t1.i + t2.i + t3.i + t4.i + t5.i + t6.i;        
   Complex v = {r, i};
   g2[OFFSET(x, y, z)] = v;
   return;
}

void dump(Complex *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f\n", input[i].r, input[i].i);
  }
}

int main(int argc, char *argv[]) {
  Complex *g_h, *g1_d, *g2_d;
  size_t nelms = N*N*N;
  g_h = (Complex *)malloc(sizeof(Complex) * nelms);
  assert(cudaSuccess ==
         cudaMalloc((void**)&g1_d, sizeof(Complex) * nelms));
  assert(cudaSuccess ==
         cudaMalloc((void**)&g2_d, sizeof(Complex) * nelms));

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g_h[i].r = i;
    g_h[i].i = i+1;
  }
    
  assert(cudaSuccess ==
         cudaMemcpy(g1_d, g_h, sizeof(Complex) * nelms,
                    cudaMemcpyHostToDevice));
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  kernel1<<<grid_dim, block_dim>>>(g1_d, g2_d);
  
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n",
            cudaGetErrorString(e));
    exit(1);
  }

  assert(cudaSuccess ==
         cudaMemcpy(g_h, g2_d, sizeof(Complex) * nelms,
                    cudaMemcpyDeviceToHost));

  dump(g_h);

  cudaDeviceReset();
  return 0;
}

