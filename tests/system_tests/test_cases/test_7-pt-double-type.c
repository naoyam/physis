/*
 * TEST: Double-precision 7-point stencil
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 32

#define T double
#define PSGrid3DT PSGrid3DDouble
#define PSGrid3DTNew PSGrid3DDoubleNew

void kernel1(const int x, const int y, const int z,
             PSGrid3DT g1, PSGrid3DT g2) {
  T v = PSGridGet(g1, x, y, z) +
      PSGridGet(g1, x+1, y, z) + PSGridGet(g1, x-1, y, z) +
      PSGridGet(g1, x, y+1, z) + PSGridGet(g1, x, y-1, z) +
      PSGridGet(g1, x, y, z-1) + PSGridGet(g1, x, y, z+1);
  PSGridEmit(g2, v);
  return;
}

#define halo_width (1)

void dump(double *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DT g1 = PSGrid3DTNew(N, N, N);
  PSGrid3DT g2 = PSGrid3DTNew(N, N, N);  

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  T *indata = (T *)malloc(sizeof(T) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  T *outdata = (T *)malloc(sizeof(T) * nelms);  
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g1, g2));
  PSGridCopyout(g2, outdata);

  dump(outdata);

  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

