#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

struct Point {
  float p[2];
  float q;
  float r;
};

__global__ void kernel(Point *g1, Point *g2) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x == 0 || x == N-1 || y == 0 || y == N-1 ||
      z == 0 || z == N-1) return;
  
  REAL v1 = g1[OFFSET(x, y, z)].p[0] +
      g1[OFFSET(x+1, y, z)].p[0] + g1[OFFSET(x-1, y, z)].p[0] +
      g1[OFFSET(x, y+1, z)].p[0] + g1[OFFSET(x, y-1, z)].p[0] +
      g1[OFFSET(x, y, z-1)].p[0] + g1[OFFSET(x, y, z+1)].p[0];
  REAL v2 = g1[OFFSET(x, y, z)].p[1] +
      g1[OFFSET(x+1, y, z)].p[1] + g1[OFFSET(x-1, y, z)].p[1] +
      g1[OFFSET(x, y+1, z)].p[1] + g1[OFFSET(x, y-1, z)].p[1] +
      g1[OFFSET(x, y, z-1)].p[1] + g1[OFFSET(x, y, z+1)].p[1];
  REAL v3 = g1[OFFSET(x, y, z)].q +
      g1[OFFSET(x+1, y, z)].q + g1[OFFSET(x-1, y, z)].q +
      g1[OFFSET(x, y+1, z)].q + g1[OFFSET(x, y-1, z)].q +
      g1[OFFSET(x, y, z-1)].q + g1[OFFSET(x, y, z+1)].q;
  g2[OFFSET(x, y, z)].r = v1+v2+v3;
  return;
}

void dump(Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i].r);
  }
}

#define halo_width (1)

int main(int argc, char *argv[]) {
  Point *indata, *indata_d;
  Point *outdata, *outdata_d;
  size_t nelms = N*N*N;
  indata = (Point *)malloc(sizeof(Point) * nelms);
  outdata = (Point *)malloc(sizeof(Point) * nelms);  
  cudaMalloc((void**)&indata_d, sizeof(Point) * nelms);
  cudaMalloc((void**)&outdata_d, sizeof(Point) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    indata[i].p[0] = i;
    indata[i].p[1] = i+1;
    indata[i].q = i+2;
    indata[i].r = 0;
    outdata[i].p[0] = 0;
    outdata[i].p[1] = 0;
    outdata[i].q = 0;
    outdata[i].r = 0;
  }
    
  cudaMemcpy(indata_d, indata, sizeof(Point) * nelms, cudaMemcpyHostToDevice);
  cudaMemcpy(outdata_d, outdata, sizeof(Point) * nelms, cudaMemcpyHostToDevice);  
  
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  kernel<<<grid_dim, block_dim>>>(indata_d, outdata_d);
  cudaMemcpy(outdata, outdata_d, sizeof(Point) * nelms, cudaMemcpyDeviceToHost);

  dump(outdata);

  cudaDeviceReset();
  return 0;
}

