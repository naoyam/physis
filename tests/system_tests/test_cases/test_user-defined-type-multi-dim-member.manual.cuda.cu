#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

struct Point {
  float p[3][2];
};

__global__ void kernel(Point *g1) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x == 0 || x == N-1 || y == 0 || y == N-1 ||
      z == 0 || z == N-1) return;
  
  REAL v1 = g1[OFFSET(x, y, z)].p[0][0] +
      g1[OFFSET(x+1, y, z)].p[0][0] + g1[OFFSET(x-1, y, z)].p[0][0] +
      g1[OFFSET(x, y+1, z)].p[0][0] + g1[OFFSET(x, y-1, z)].p[0][0] +
      g1[OFFSET(x, y, z-1)].p[0][0] + g1[OFFSET(x, y, z+1)].p[0][0];
  REAL v2 = g1[OFFSET(x, y, z)].p[2][1] +
      g1[OFFSET(x+1, y, z)].p[2][1] + g1[OFFSET(x-1, y, z)].p[2][1] +
      g1[OFFSET(x, y+1, z)].p[2][1] + g1[OFFSET(x, y-1, z)].p[2][1] +
      g1[OFFSET(x, y, z-1)].p[2][1] + g1[OFFSET(x, y, z+1)].p[2][1];
  g1[OFFSET(x, y, z)].p[1][0] = v1+v2;
  return;
}

void dump(Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i].p[1][0]);
  }
}

#define halo_width (1)

int main(int argc, char *argv[]) {
  Point *indata, *indata_d;
  Point *outdata;
  int nelms = N*N*N;
  indata = (Point *)malloc(sizeof(Point) * nelms);
  outdata = (Point *)malloc(sizeof(Point) * nelms);  
  cudaMalloc((void**)&indata_d, sizeof(Point) * nelms);

  int i;
  for (i = 0; i < nelms; i++) {
    int j;
    for (j = 0; j < 3; ++j) {
      int k;
      for (k = 0; k < 2; ++k) {
        indata[i].p[j][k] = i+j+k;
        outdata[i].p[j][k] = 0;
      }
    }
  }
    
  cudaMemcpy(indata_d, indata, sizeof(Point) * nelms, cudaMemcpyHostToDevice);
  
  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(N/block_dim.x, N/block_dim.y, N/block_dim.z);

  kernel<<<grid_dim, block_dim>>>(indata_d);
  cudaMemcpy(outdata, indata_d, sizeof(Point) * nelms, cudaMemcpyDeviceToHost);

  dump(outdata);

  cudaDeviceReset();
  return 0;
}

