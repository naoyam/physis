/*
 * TEST: 7-point stencil with boundary branches
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

void kernel(const int x, const int y, const int z, PSGrid3DFloat g1,
            PSGrid3DFloat g2) {
  int nx, ny, nz;
  nx = PSGridDim(g1, 0);
  ny = PSGridDim(g1, 1);
  nz = PSGridDim(g1, 2);

  float c, w, e, n, s, b, t;
  c = PSGridGet(g1, x, y, z);
  if (x == 0)    w = c; else w = PSGridGet(g1, x-1, y, z);
  if (x == nx-1) e = c ; else e = PSGridGet(g1, x+1, y, z);
  if (y == 0)    n = c ; else n=PSGridGet(g1, x, y-1, z);
  if (y == ny-1) s= c ; else s=PSGridGet(g1, x, y+1, z);
  if (z == 0)    b= c ; else b=PSGridGet(g1, x, y, z-1);
  if (z == nz-1) t= c ; else t=PSGridGet(g1, x, y, z+1);
  PSGridEmit(g2, c + w + e + s + n + b + t);
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

#define halo_width (0)

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);  

  PSStencilRun(PSStencilMap(kernel, d, g1, g2));
    
  PSGridCopyout(g2, outdata);
  dump(outdata);  

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

