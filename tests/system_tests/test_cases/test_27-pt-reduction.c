/*
 * TEST: 27-point stencil with reduction
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 16

#define PSGrid3DReal PSGrid3DInt
#define PSGrid3DRealNew PSGrid3DIntNew
#define REAL int

void kernel(const int x, const int y, const int z, PSGrid3DReal g,
            PSGrid3DReal g2) {
  float v =
      // z == -1
      PSGridGet(g, x, y, z-1) + PSGridGet(g, x+1, y, z-1) +
      PSGridGet(g, x-1, y, z-1) + PSGridGet(g, x, y+1, z-1) +
      PSGridGet(g, x+1, y+1, z-1) + PSGridGet(g, x-1, y+1, z-1) +
      PSGridGet(g, x, y-1, z-1) + PSGridGet(g, x+1, y-1, z-1) +
      PSGridGet(g, x-1, y-1, z-1) +
      // z == 0
      PSGridGet(g, x, y, z) + PSGridGet(g, x+1, y, z) +
      PSGridGet(g, x-1, y, z) + PSGridGet(g, x, y+1, z) +
      PSGridGet(g, x+1, y+1, z) + PSGridGet(g, x-1, y+1, z) +
      PSGridGet(g, x, y-1, z) + PSGridGet(g, x+1, y-1, z) +
      PSGridGet(g, x-1, y-1, z) +
      // z == 1
      PSGridGet(g, x, y, z+1) + PSGridGet(g, x+1, y, z+1) +
      PSGridGet(g, x-1, y, z+1) + PSGridGet(g, x, y+1, z+1) +
      PSGridGet(g, x+1, y+1, z+1) + PSGridGet(g, x-1, y+1, z+1) +
      PSGridGet(g, x, y-1, z+1) + PSGridGet(g, x+1, y-1, z+1) +
      PSGridGet(g, x-1, y-1, z+1);
  PSGridEmit(g2, v);
  return;
}

#define halo_width (1)

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DReal g1 = PSGrid3DRealNew(N, N, N);
  PSGrid3DReal g2 = PSGrid3DRealNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  REAL *indata = (REAL *)malloc(sizeof(REAL) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);  

  PSStencilRun(PSStencilMap(kernel, d, g1, g2));
  REAL v;
  PSReduce(&v, PS_SUM, g1);
  printf("%d\n", v);

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  return 0;
}

