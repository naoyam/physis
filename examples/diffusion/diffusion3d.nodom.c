#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define __PHYSIS__

#ifdef __PHYSIS__
#include "physis_user.h"
#endif

#define REAL float
#define NX (256)
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifdef __PHYSIS__
DeclareGrid3D(real, REAL);

#else
typedef struct grid3d_real_tag {
  int nx, ny, nz;
  REAL *f, *fn;
} *grid3d_real ;

typedef struct {
  int dummy;
  int x, y, z;
} domain3d_regular;


static inline int grid_dimx(grid3d_real g) { return g->nx; }
static inline int grid_dimy(grid3d_real g) { return g->ny; }
static inline int grid_dimz(grid3d_real g) { return g->nz; }

static inline
REAL grid_get(grid3d_real g, int x, int y, int z) {
  return g->f[z*g->nx*g->ny + y*g->nx + x];
}

static
grid3d_real grid3d_real_new(const int nx, const int ny, const int nz) {
  grid3d_real g = (grid3d_real)malloc(sizeof(struct grid3d_real_tag));
  g->nx = nx;
  g->ny = ny;
  g->nz = nz;
  size_t size = nx*ny*nz*sizeof(REAL);
  g->f = (REAL *)malloc(size);
  g->fn = (REAL *)malloc(size);
  return g;
}

inline
REAL kernel(int x, int y, int z, grid3d_real g,
            REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, REAL cb, REAL cc);

static inline void
grid_update(domain3d_regular d, void *p, grid3d_real g,
            REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, REAL cb, REAL cc) {
  for (int jz = 0; jz < g->nz; jz++) {
    for (int jy = 0; jy < g->ny; jy++) {
      for (int jx = 0; jx < g->nx; jx++) {
        int j = jz*g->nx*g->ny + jy*g->nx + jx;
        g->fn[j] = kernel(jx, jy, jz, g, ce, cw, cn, cs, ct, cb, cc);
      }
    }
  }
  REAL *tmp = g->f;
  g->f = g->fn;
  g->fn = tmp;
}

static inline
void grid_copyin(grid3d_real g, REAL *buff) {
  memcpy(g->f, buff, g->nx*g->ny*g->nz*sizeof(REAL));
}

static inline
void grid_copyout(grid3d_real g, REAL *buff) {
  memcpy(buff, g->f, g->nx*g->ny*g->nz*sizeof(REAL));
}

#endif

void kernel(int x, int y, int z, grid3d_real g,
            REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, REAL cb, REAL cc)
{
  int nx, ny, nz;
  nx = grid_dimx(g);
  ny = grid_dimy(g);
  nz = grid_dimz(g);

  REAL c, w, e, n, s, b, t;
  c = grid_get(g, 0, 0, 0);
  w = (x == 0)    ? c : grid_get(g, -1, 0, 0);
  e = (x == nx-1) ? c : grid_get(g, 1, 0, 0);
  n = (y == 0)    ? c : grid_get(g, 0, -1, 0);
  s = (y == ny-1) ? c : grid_get(g, 0, 1, 0);
  b = (z == 0)    ? c : grid_get(g, 0, 0, -1);
  t = (z == nz-1) ? c : grid_get(g, 0, 0, 1);
  grid_emit(g, cc*c + cw*w + ce*e + cs*s
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

int main(int argc, char *argv[]) 
{
  PhysisInit(&argc, &argv);
  grid3d_real g = grid3d_real_new(NX, NX, NX);
  struct timeval time_begin, time_end;

  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;
  REAL  *buff  = (REAL *)malloc(sizeof(REAL) *nx*ny*nz);
  REAL   time  = 0.0;
  int    count = 0;
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;

  init(buff, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

  grid_copyin(g, buff);

  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  //clock_gettime(CLOCK_REALTIME, &time_begin);
  gettimeofday(&time_begin, NULL);
  do {
    if (count && (count % 100 == 0)) {
      fprintf(stderr, "time(%4d)=%7.5f\n", count, time + dt);
    }
    grid_update2(kernel,g,ce,cw,cn,cs,ct,cb,cc);
    time += dt;
    count++;
  } while (time + 0.5*dt < 0.1);
  //clock_gettime(CLOCK_REALTIME, &time_end);
  gettimeofday(&time_end, NULL);

  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  grid_copyout(g, buff);

  REAL err = accuracy(buff, answer, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
          + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL mflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-06;
  double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
          / elapsed_time * 1.0e-9;

  fprintf(stderr, "elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stderr, "flops        : %.3f (MFlops)\n", mflops);
  fprintf(stderr, "throughput   : %.3f (GB/s)\n", thput);  
  fprintf(stderr, "accuracy     : %e\n", err);
  free(answer);
  PhysisFinalize();
  return 0;
}
