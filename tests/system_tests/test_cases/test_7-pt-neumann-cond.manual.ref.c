#include <stdio.h>
#include <stdlib.h>

#define N 32
#define ITER 10
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(REAL *g1, REAL *g2,
            int nx, int ny, int nz,
            REAL ce, REAL cw, REAL cn, REAL cs,
            REAL ct, REAL cb, REAL cc) {
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
        n = (y == 0)    ? c : c - ny;
        s = (y == ny-1) ? c : c + ny;
        b = (z == 0)    ? c : c - nx * ny;
        t = (z == nz-1) ? c : c + nx * ny;
        g2[c] = cc * g1[c] + cw * g1[w] + ce * g1[e]
            + cs * g1[s] + cn * g1[n] + cb * g1[b] + ct * g1[t];
      }
    }
  }
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g2;  
  size_t nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  g2 = (REAL *)malloc(sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < nelms; i++) {
    g1[i] = i;
  }

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

  for (i = 0; i < ITER; ++i) {
    kernel(g1, g2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
    REAL *t = g1;
    g1 = g2;
    g2 = t;
  }
  dump(g1);
  
  free(g1);
  free(g2);
  return 0;
}

