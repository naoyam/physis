/*
 * TEST: 7-point Neumann boundary condition
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32
#define ITER 10
#define REAL float
#define PSGrid3DReal PSGrid3DFloat
#define PSGrid3DRealNew PSGrid3DFloatNew

void kernel(const int x, const int y, const int z,
            PSGrid3DReal g1, PSGrid3DReal g2,
            REAL ce, REAL cw, REAL cn, REAL cs,
            REAL ct, REAL cb, REAL cc) {
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
  PSGridEmit(g2, cc*c + cw*w + ce*e + cs*s
             + cn*n + cb*b + ct*t);
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
    
  PSGridCopyin(g1, indata);

  PSStencilRun(PSStencilMap(kernel, d, g1, g2,
                            ce, cw, cn, cs, ct, cb, cc),
               PSStencilMap(kernel, d, g2, g1,
                            ce, cw, cn, cs, ct, cb, cc),               
               ITER/2);
  
  PSGridCopyout(g1, outdata);

  dump(outdata);  

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

