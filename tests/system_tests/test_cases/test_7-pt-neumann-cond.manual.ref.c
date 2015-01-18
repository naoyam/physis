#include <stdio.h>
#include <stdlib.h>

#define N 8
#define ITER 1
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(REAL *g1, REAL *g2,
            int nx, int ny, int nz) {
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
        g2[c] = g1[c] + g1[w] + g1[e]
            + g1[s] + g1[n] + g1[b] + g1[t];
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

  for (i = 0; i < ITER; ++i) {
    kernel(g1, g2, nx, ny, nz);
    REAL *t = g1;
    g1 = g2;
    g2 = t;
  }
  dump(g1);
  
  free(g1);
  free(g2);
  return 0;
}

