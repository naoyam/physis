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
#ifndef NX
#define NX (256)
#endif
#ifndef NY
#define NY (256)
#endif
#ifndef NZ
#define NZ (256)
#endif
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#define ENABLE_ERROR_CHECKING 0
#define ENABLE_INIT_ON_GPU 1

void kernel(const int x, const int y, const int z,
            PSGrid3DReal  g1,  PSGrid3DReal  g2,
            REAL ce, REAL cw, REAL cn, REAL cs,
            REAL ct, REAL cb, REAL cc) {
  int nx, ny, nz;
  nx = PSGridDim(g1, 0);
  ny = PSGridDim(g1, 1);
  nz = PSGridDim(g1, 2);

  REAL c, w, e, n, s, b, t;
  c = PSGridGet(g1, x, y, z);
  w = (x == 0)    ? c : PSGridGet(g1, x-1, y, z);
  e = (x == nx-1) ? c : PSGridGet(g1, x+1, y, z);
  n = (y == 0)    ? c : PSGridGet(g1, x, y-1, z);
  s = (y == ny-1) ? c : PSGridGet(g1, x, y+1, z);
  b = (z == 0)    ? c : PSGridGet(g1, x, y, z-1);
  t = (z == nz-1) ? c : PSGridGet(g1, x, y, z+1);
  PSGridEmit(g2, cc*c + cw*w + ce*e + cs*s
             + cn*n + cb*b + ct*t);
  return;
}

void init(REAL *buff, const size_t nx, const size_t ny, const size_t nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time) {
  REAL ax, ay, az;
  size_t jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
  for (jz = 0; jz < nz; jz++) {
    for (jy = 0; jy < ny; jy++) {
      for (jx = 0; jx < nx; jx++) {
        size_t j = jz*nx*ny + jy*nx + jx;
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

void init_kernel(const int jx, const int jy, const int jz,
                 PSGrid3DReal g,
                 REAL kx, REAL ky, REAL kz,
                 REAL dx, REAL dy, REAL dz,
                 REAL kappa, REAL time) {
  REAL ax, ay, az;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
  
  REAL x = dx*((REAL)(jx + 0.5));
  REAL y = dy*((REAL)(jy + 0.5));
  REAL z = dz*((REAL)(jz + 0.5));
  REAL f0 = (REAL)0.125
      *(1.0 - ax*cos(kx*x))
      *(1.0 - ay*cos(ky*y))
      *(1.0 - az*cos(kz*z));
  PSGridEmit(g, f0);
}


REAL accuracy(const REAL *b1, REAL *b2, const size_t len) {
  REAL err = 0.0;
  size_t i;
  for (i = 0; i < len; i++) {
    err += (b1[i] - b2[i]) * (b1[i] - b2[i]);
  }
  return (REAL)sqrt(err/len);
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, NX, NY, NZ);
  PSGrid3DReal g1 = PSGrid3DRealNew(NX, NY, NZ);
  PSGrid3DReal g2 = PSGrid3DRealNew(NX, NY, NZ);  
  PSDomain3D d = PSDomain3DNew(0, NX, 0, NY, 0, NZ);  

  struct timeval time_begin, time_end;
  size_t    nx    = NX;
  size_t    ny    = NY;
  size_t    nz    = NZ;

  REAL   time  = 0.0;
  int    count = 100;
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;

#if ENABLE_INIT_ON_GPU
  PSStencilRun(PSStencilMap(init_kernel, d, g1,
                            kx, ky, kz, dx, dy, dz, kappa, time));
#else
  REAL  *buff  = (REAL *)malloc(sizeof(REAL) *nx*ny*nz);  
  init(buff, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  PSGridCopyin(g1, buff);
#endif

  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);
  // Warming up
  PSStencilRun(PSStencilMap(kernel, d, g1, g2, ce,cw,cn,cs,ct,cb,cc),
               PSStencilMap(kernel, d, g2, g1, ce,cw,cn,cs,ct,cb,cc));
  //assert(count % 2 == 0);
  gettimeofday(&time_begin, NULL);
  PSStencilRun(PSStencilMap(kernel, d, g1, g2, ce,cw,cn,cs,ct,cb,cc),
               PSStencilMap(kernel, d, g2, g1, ce,cw,cn,cs,ct,cb,cc),
               count/2);
  gettimeofday(&time_end, NULL);
  // For the 2 times in warming up
  count += 2;
  time += dt * count;
#if ENABLE_ERROR_CHECKING  
  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  PSGridCopyout(g1, buff);
  REAL err = accuracy(buff, answer, nx*ny*nz);
#else
  REAL err = 0.0;
#endif
  
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
                        + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
  double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
                 / elapsed_time / (1000 * 1000 * 1000);

  fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops);
  fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput);
  fprintf(stdout, "accuracy     : %e\n", err);  
  fprintf(stdout, "count        : %d\n", count);
  //free(answer);
  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();
  return 0;
}
