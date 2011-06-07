#include "physis/physis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define REAL float
#define PSGrid3DReal PSGrid3DFloat
#define PSGrid3DRealNew PSGrid3DFloatNew
#define NX (256)
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

void kernel(const int x, const int y, const int z,
            PSGrid3DReal  g,
            REAL ce, REAL cw, REAL cn, REAL cs,
            REAL ct, REAL cb, REAL cc) {
  int nx, ny, nz;
  nx = PSGridDim(g, 0);
  ny = PSGridDim(g, 1);
  nz = PSGridDim(g, 2);

  REAL c, w, e, n, s, b, t;
  c = PSGridGet(g, x, y, z);
  w = (x == 0)    ? c : PSGridGet(g, x-1, y, z);
  e = (x == nx-1) ? c : PSGridGet(g, x+1, y, z);
  n = (y == 0)    ? c : PSGridGet(g, x, y-1, z);
  s = (y == ny-1) ? c : PSGridGet(g, x, y+1, z);
  b = (z == 0)    ? c : PSGridGet(g, x, y, z-1);
  t = (z == nz-1) ? c : PSGridGet(g, x, y, z+1);
  PSGridEmit(g, cc*c + cw*w + ce*e + cs*s
             + cn*n + cb*b + ct*t);
  return;
}

void init(REAL *buff, const int nx, const int ny, const int nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time) {
  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
  for (jz = 0; jz < nz; jz++) {
    for (jy = 0; jy < ny; jy++) {
      for (jx = 0; jx < nx; jx++) {
        int j = jz*nx*ny + jy*nx + jx;
        REAL x = dx*((REAL)(jx + 0.5));
        REAL y = dy*((REAL)(jy + 0.5));
        REAL z = dz*((REAL)(jz + 0.5));
        REAL f0 = (REAL)0.125
                  *(1.0 - ax*cos(kx*x))
                  *(1.0 - ay*cos(ky*y))
                  *(1.0 - az*cos(kz*z));
        buff[j] = f0;
      }
    }
  }
}

REAL accuracy(const REAL *b1, REAL *b2, const int len) {
  REAL err = 0.0;
  int i;
  for (i = 0; i < len; i++) {
    err += (b1[i] - b2[i]) * (b1[i] - b2[i]);
  }
  return (REAL)sqrt(err/len);
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, NX, NX, NX);
  PSGrid3DReal g = PSGrid3DRealNew(NX, NX, NX);
  PSDomain3D d = PSDomain3DNew(0, NX, 0, NX, 0, NX);  

  struct timeval time_begin, time_end;
  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;
  REAL  *buff  = (REAL *)malloc(sizeof(REAL) *nx*ny*nz);
  REAL   time  = 0.0;
  int    count = 1000;
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;

  init(buff, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

  PSGridCopyin(g, buff);

  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  gettimeofday(&time_begin, NULL);
  PSStencilRun(PSStencilMap(kernel, d, g,ce,cw,cn,cs,ct,cb,cc),
               count);
  gettimeofday(&time_end, NULL);

  time += dt * count;
  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  PSGridCopyout(g, buff);
  REAL err = accuracy(buff, answer, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
                        + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL mflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-06;
  double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
                 / elapsed_time / (1000 * 1000 * 1000);

  fprintf(stderr, "elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stderr, "flops        : %.3f (MFlops)\n", mflops);
  fprintf(stderr, "throughput   : %.3f (GB/s)\n", thput);
  fprintf(stderr, "accuracy     : %e\n", err);  
  fprintf(stderr, "count        : %d\n", count);
  free(answer);
  PSGridFree(g);
  PSFinalize();
  return 0;
}
