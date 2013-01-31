#include "physis/physis.h"

#define REAL float
#define PSGrid3DReal PSGrid3DFloat
#define PSGrid3DRealNew PSGrid3DFloatNew

PSGrid3DReal f1g;
PSGrid3DReal f2g;

#ifdef __cplusplus
extern "C" {
#endif

void initialize_physis(int argc, char **argv, int nx, int ny, int nz) {
  PSInit(&argc, &argv, 3, nx, ny, nz);
}

void initialize_benchmark_physis(int nx, int ny, int nz) {
  f1g = PSGrid3DRealNew(nx, ny, nz);
  f2g = PSGrid3DRealNew(nx, ny, nz);
}

void finalize_benchmark_physis() {
  PSGridFree(f1g);
  PSGridFree(f2g);
  PSFinalize();
}

static void kernel_physis(const int x, const int y, const int z,
                          PSGrid3DReal  g1, PSGrid3DReal  g2,
                          REAL ce, REAL cw, REAL cn, REAL cs,
                          REAL ct, REAL cb, REAL cc) {
  int nx, ny, nz;
  nx = PSGridDim(g1, 0);
  ny = PSGridDim(g1, 1);
  nz = PSGridDim(g1, 2);

  REAL c, w, e, n, s, b, t;
  c = PSGridGet(g1, x, y, z);
#if SELECT_SUPPORTED_IN_UNCONDITIONAL_GET
  w = (x == 0)    ? c : PSGridGet(g1, x-1, y, z);
  e = (x == nx-1) ? c : PSGridGet(g1, x+1, y, z);
  n = (y == 0)    ? c : PSGridGet(g1, x, y-1, z);
  s = (y == ny-1) ? c : PSGridGet(g1, x, y+1, z);
  b = (z == 0)    ? c : PSGridGet(g1, x, y, z-1);
  t = (z == nz-1) ? c : PSGridGet(g1, x, y, z+1);
#else
  if (x == 0)    w = PSGridGet(g1, x, y, z); else w = PSGridGet(g1, x-1, y, z);
  if (x == nx-1) e = PSGridGet(g1, x, y, z); else e = PSGridGet(g1, x+1, y, z);
  if (y == 0)    n = PSGridGet(g1, x, y, z); else n = PSGridGet(g1, x, y-1, z);
  if (y == ny-1) s = PSGridGet(g1, x, y, z); else s = PSGridGet(g1, x, y+1, z);
  if (z == 0)    b = PSGridGet(g1, x, y, z); else b = PSGridGet(g1, x, y, z-1);
  if (z == nz-1) t = PSGridGet(g1, x, y, z); else t = PSGridGet(g1, x, y, z+1);
#endif
  PSGridEmit(g2, cc*c + cw*w + ce*e + cs*s
             + cn*n + cb*b + ct*t);
  return;
}

void run_kernel_physis(int count, REAL *f1_host,
                       int nx, int ny, int nz,
                       REAL ce, REAL cw, REAL cn, REAL cs,
                       REAL ct, REAL cb, REAL cc) {
  
  PSDomain3D dom = PSDomain3DNew(0, nx, 0, ny, 0, nz);  
  PSGridCopyin(f1g, f1_host);

  PSStencilRun(PSStencilMap(kernel_physis, dom, f1g, f2g,
                            ce, cw, cn, cs, ct, cb, cc),
               PSStencilMap(kernel_physis, dom, f2g, f1g,
                            ce, cw, cn, cs, ct, cb, cc),               
               count/2);
  
  PSGridCopyout(f1g, f1_host);  
}

#ifdef __cplusplus
}
#endif
