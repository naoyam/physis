#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>

#define REAL float
#define NX (256)
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

void init(REAL *buff, const int nx, const int ny, const int nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time) {
  REAL ax, ay, az;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
  int jz;  
  #pragma omp parallel for  
  for (jz = 0; jz < nz; jz++) {
    int jy;
    for (jy = 0; jy < ny; jy++) {
      int jx;
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
                                 int count, double *);

static void
diffusion_baseline(REAL *f1, REAL *f2, int nx, int ny, int nz,
                   REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                   REAL cb, REAL cc, REAL dt,
                   int count, double *etime) {
  int i;
  for (i = 0; i < count; ++i) {
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
  }
  return;
}

static void
diffusion_openmp(REAL *f1, REAL *f2, int nx, int ny, int nz,
                 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                 REAL cb, REAL cc, REAL dt, int count, double *etime) {

  {
    REAL *f1_t = f1;
    REAL *f2_t = f2;

    int i;
    for (i = 0; i < count; ++i) {
      int z;
#pragma omp parallel for
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
    }
  }
  return;
}

#ifdef __INTEL_COMPILER
__declspec(target(mic))
static double cur_second(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void
diffusion_mic(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
              REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
              REAL cb, REAL cc, REAL dt, int count, double *etime) {
  double runtime;
#pragma offload target(mic) \
  inout(f1:length(nx*ny*nz) align(2*1024*1024))			\
  inout(f2:length(nx*ny*nz) align(2*1024*1024)) out(runtime)
  {
    double start = cur_second();
    int i;
    for (i = 0; i < count; ++i) {
      int y, z;
#pragma omp parallel for  collapse(2) private(y, z)
      for (z = 0; z < nz; z++) {
         for (y = 0; y < ny; y++) {
          int x;
#pragma ivdep
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
    }
    double end = cur_second();
    runtime = end - start;
  }

  *etime = runtime;

  return;
}
#endif

static void dump_result(REAL *f, int nx, int ny, int nz, char *out_path) {
  FILE *out = fopen(out_path, "w");
  assert(out);
  size_t nitems = nx * ny * nz;
  //fwrite(f, sizeof(REAL), nitems, out);
  int i;
  for (i = 0; i < nitems; ++i) {
    fprintf(out, "%f\n", f[i]);
  }
  fclose(out);
}

int main(int argc, char *argv[]) 
{
  
  struct timeval time_begin, time_end;

  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;

#if USE_MM_MALLOC
  REAL *f1 = (REAL *)_mm_malloc(sizeof(REAL)*NX,4096);
  REAL *f2 = (REAL *)_mm_malloc(sizeof(REAL)*NX,4096);
#else
  REAL *f1 = (REAL *)malloc(sizeof(REAL)*NX*NX*NX);
  REAL *f2 = (REAL *)malloc(sizeof(REAL)*NX*NX*NX);
#endif
  
  REAL *f_final = NULL;

  REAL   time  = 0.0;
  int    count = 0;  

  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

  char *version_str;

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;
  //count = 0.1 / dt;
  count = 300;
  f_final = (count % 2)? f2 : f1;

  init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  // use baseline by default
  diffusion_loop_t diffusion_loop = diffusion_baseline;
  version_str = "baseline";
  
  if (argc == 2) {
    if (strcmp(argv[1], "openmp") == 0) {
      diffusion_loop = diffusion_openmp;
      version_str = "openmp";
    }
#ifdef __INTEL_COMPILER
    if (strcmp(argv[1], "mic") == 0) {
      printf("MIC\n");
      diffusion_loop = diffusion_mic;
      version_str = "mic";
    }
#endif    
  }
  
  double inner_elaplsed_time = 0.0;
  printf("Running %s diffusion kernel %d times with %dx%dx%d grid\n",
         version_str, count, nx, ny, nz);
  gettimeofday(&time_begin, NULL);
  diffusion_loop(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc,
                 dt, count, &inner_elaplsed_time);
  gettimeofday(&time_end, NULL);
  time = count * dt;
  char dump_path[128];
  sprintf(dump_path, "%s.%s", "diffusion_result.dat", version_str);
  dump_result(f_final, nx, ny, nz, dump_path);

  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

  REAL err = accuracy(f_final, answer, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL mflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-06;
  double thput = (nx * ny * nz) * sizeof(REAL) * 3.0 * count
      / elapsed_time * 1.0e-09;

  fprintf(stderr, "Elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stderr, "FLOPS        : %.3f (MFlops)\n", mflops);
  fprintf(stderr, "Throughput   : %.3f (GB/s)\n", thput);  
  fprintf(stderr, "Accuracy     : %e\n", err);
  free(answer);
  fprintf(stderr, "Time (w/o PCI): %.3f\n", inner_elaplsed_time);
  fprintf(stderr, "FLOPS (w/o PCI): %.3f (MFLOPS)\n", 
    (nx*ny*nz)*13.0*count/inner_elaplsed_time * 1.0e-06);
  fprintf(stderr, "Throughput (w/o PCI): %.3f\n",
          nx *ny * nz * sizeof(REAL) * 3 * count /
          inner_elaplsed_time * 1.0e-09);
  free(f1);
  free(f2);
  return 0;
}
