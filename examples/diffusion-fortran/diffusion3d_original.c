#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define REAL float
#define NX (64)
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif


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

typedef void (*diffusion_loop_t)(REAL *f1, REAL *f2, int nx, int ny, int nz,
                                 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                                 REAL cb, REAL cc, REAL dt,
                                 REAL **f_ret, REAL *time_ret, int *count_ret);

static void
diffusion_baseline(REAL *f1, REAL *f2, int nx, int ny, int nz,
                   REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                   REAL cb, REAL cc, REAL dt,
                   REAL **f_ret, REAL *time_ret, int *count_ret) {
  REAL time = 0.0;
  int count = 0;
  
  do {
    int z;
    for (z = 0; z < nz; z++) {
      int y;
      for (y = 0; y < ny; y++) {
        int x;
        for (x = 0; x < nx; x++) {
          int c, w, e, n, s, b, t;
          c =  x + y * nx + z * nx * ny;
          w = (x == 0)    ? c : c - 1;
          e = (x == nx-1) ? c : c + 1;
          n = (y == 0)    ? c : c - nx;
          s = (y == ny-1) ? c : c + nx;
          b = (z == 0)    ? c : c - nx * ny;
          t = (z == nz-1) ? c : c + nx * ny;
          f2[c] = cc * f1[c] + cw * f1[w] + ce * f1[e]
              + cs * f1[s] + cn * f1[n] + cb * f1[b] + ct * f1[t];
        }
      }
    }
    REAL *t = f1;
    f1 = f2;
    f2 = t;
    time += dt;
    count++;
  } while (time + 0.5*dt < 0.1);
  *time_ret = time;
  *f_ret = f1;
  *count_ret = count;
  
  return;
}

static void
diffusion_openmp(REAL *f1, REAL *f2, int nx, int ny, int nz,
                   REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                   REAL cb, REAL cc, REAL dt,
                   REAL **f_ret, REAL *time_ret, int *count_ret) {

  
#pragma omp parallel
  {
    REAL time = 0.0;
    int count = 0;
    REAL *f1_t = f1;
    REAL *f2_t = f2;
    
#pragma omp master
    printf("%d threads running\n", omp_get_num_threads());

    do {
      int z;
#pragma omp for
      for (z = 0; z < nz; z++) {
        int y;
        for (y = 0; y < ny; y++) {
          int x;
          for (x = 0; x < nx; x++) {
            int c, w, e, n, s, b, t;
            c =  x + y * nx + z * nx * ny;
            w = (x == 0)    ? c : c - 1;
            e = (x == nx-1) ? c : c + 1;
            n = (y == 0)    ? c : c - nx;
            s = (y == ny-1) ? c : c + nx;
            b = (z == 0)    ? c : c - nx * ny;
            t = (z == nz-1) ? c : c + nx * ny;
            f2_t[c] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
                + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
          }
        }
      }
      REAL *t = f1_t;
      f1_t = f2_t;
      f2_t = t;
      time += dt;
      count++;
    } while (time + 0.5*dt < 0.1);
#pragma omp master
    {
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
  }

  return;
}


int main(int argc, char *argv[]) 
{
  
  struct timeval time_begin, time_end;

  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;
  REAL *f1 = (REAL *)malloc(sizeof(REAL)*NX*NX*NX);
  REAL *f2 = (REAL *)malloc(sizeof(REAL)*NX*NX*NX);  

  REAL   time  = 0.0;
  int    count = 0;  

  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;

  init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  diffusion_loop_t diffusion_loop = diffusion_baseline;
  if (argc == 2) {
    if (strcmp(argv[1], "openmp") == 0) {
      diffusion_loop = diffusion_openmp;
    }
  }
  
  gettimeofday(&time_begin, NULL);
  diffusion_loop(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
                 &f1, &time, &count);
  gettimeofday(&time_end, NULL);

  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

  REAL err = accuracy(f1, answer, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL mflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-06;
  double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
      / elapsed_time / (1 << 30);

  fprintf(stderr, "elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stderr, "flops        : %.3f (MFlops)\n", mflops);
  fprintf(stderr, "throughput   : %.3f (GB/s)\n", thput);  
  fprintf(stderr, "accuracy     : %e\n", err);
  free(answer);
  free(f1);
  free(f2);
  return 0;
}
