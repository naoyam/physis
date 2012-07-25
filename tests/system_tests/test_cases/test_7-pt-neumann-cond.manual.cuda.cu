#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#define N 32
#define ITER 10
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

__global__ void kernel(REAL *f1, REAL *f2,
                       int nx, int ny, int nz,
                       REAL ce, REAL cw, REAL cn, REAL cs,
                       REAL ct, REAL cb, REAL cc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j * nx;
  int xy = nx * ny;
  for (int k = 0; k < nz; ++k) {
    int w = (i == 0)        ? c : c - 1;
    int e = (i == nx-1)     ? c : c + 1;
    int n = (j == 0)        ? c : c - ny;
    int s = (j == ny-1)     ? c : c + ny;
    int b = (k == 0)        ? c : c - xy;
    int t = (k == nz-1)     ? c : c + xy;
    f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e] + cs * f1[s]
        + cn * f1[n] + cb * f1[b] + ct * f1[t];
    c += xy;
  }
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g1d;
  REAL *g2d;
  int nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  cudaMalloc((void**)&g1d, sizeof(REAL) * nelms);
  cudaMalloc((void**)&g2d, sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }

  int nx = N, ny = N, nz = N;

  REAL l = 1.0;
  REAL kappa = 0.1;
  REAL dx = l / nx;
  REAL dy = l / ny;
  REAL dz = l / nz;
  //REAL kx, ky, kz;
  //kx = ky = kz = 2.0 * M_PI;
  REAL dt = 0.1 * dx * dx / kappa;
  REAL ce, cw;
  ce = cw = kappa*dt/(dx*dx);
  REAL cn, cs;
  cn = cs = kappa*dt/(dy*dy);
  REAL ct, cb;
  ct = cb = kappa*dt/(dz*dz);
  REAL cc = 1.0 - (ce + cw + cn + cs + ct + cb);
    
  cudaMemcpy(g1d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  
  dim3 block_dim(16, 2);
  dim3 grid_dim(nx/block_dim.x, ny/block_dim.y);

  for (i = 0; i < ITER; ++i) {
    kernel<<<grid_dim, block_dim>>>(
        g1d, g2d, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
    REAL *t = g1d;
    g1d = g2d;
    g2d = t;
  }
  cudaMemcpy(g1, g1d, sizeof(REAL) * nelms, cudaMemcpyDeviceToHost);

  dump(g1);

  cudaDeviceReset();
  return 0;
}

