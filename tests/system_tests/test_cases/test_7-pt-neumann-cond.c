/*
 * TEST: 7-point Neumann boundary condition
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 8
#define REAL float
#define PSGrid3DReal PSGrid3DFloat
#define PSGrid3DRealNew PSGrid3DFloatNew

static void kernel(const int x, const int y, const int z,
                   PSGrid3DReal g1, PSGrid3DReal g2) {
  int nx, ny, nz;
  nx = PSGridDim(g1, 0);
  ny = PSGridDim(g1, 1);
  nz = PSGridDim(g1, 2);

  REAL c, w, e, n, s, b, t;
  c = PSGridGet(g1, x, y, z);
  if (x == 0)    w = PSGridGet(g1, x, y, z); else w = PSGridGet(g1, x-1, y, z);
  if (x == nx-1) e = PSGridGet(g1, x, y, z); else e = PSGridGet(g1, x+1, y, z);
  if (y == 0)    n = PSGridGet(g1, x, y, z); else n = PSGridGet(g1, x, y-1, z);
  if (y == ny-1) s = PSGridGet(g1, x, y, z); else s = PSGridGet(g1, x, y+1, z);
  if (z == 0)    b = PSGridGet(g1, x, y, z); else b = PSGridGet(g1, x, y, z-1);
  if (z == nz-1) t = PSGridGet(g1, x, y, z); else t = PSGridGet(g1, x, y, z+1);
  PSGridEmit(g2, c + w + e + s + n + b + t);
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DReal g1 = PSGrid3DRealNew(N, N, N);
  PSGrid3DReal g2 = PSGrid3DRealNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  REAL *indata = (REAL *)malloc(sizeof(REAL) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  REAL *outdata = (REAL *)malloc(sizeof(REAL) * nelms);

  PSGridCopyin(g1, indata);
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

