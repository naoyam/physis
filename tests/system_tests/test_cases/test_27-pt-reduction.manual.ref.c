#include <stdio.h>
#include <stdlib.h>

#define N 16
#define REAL int

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(REAL *g1, REAL *g2) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        REAL v =
            g1[OFFSET(x, y, z-1)] + g1[OFFSET(x+1, y, z-1)] +
            g1[OFFSET(x-1, y, z-1)] + g1[OFFSET(x, y+1, z-1)] +
            g1[OFFSET(x+1, y+1, z-1)] + g1[OFFSET(x-1, y+1, z-1)] +
            g1[OFFSET(x, y-1, z-1)] + g1[OFFSET(x+1, y-1, z-1)] +
            g1[OFFSET(x-1, y-1, z-1)] +
            // z == 0
            g1[OFFSET(x, y, z)] + g1[OFFSET(x+1, y, z)] +
            g1[OFFSET(x-1, y, z)] + g1[OFFSET(x, y+1, z)] +
            g1[OFFSET(x+1, y+1, z)] + g1[OFFSET(x-1, y+1, z)] +
            g1[OFFSET(x, y-1, z)] + g1[OFFSET(x+1, y-1, z)] +
            g1[OFFSET(x-1, y-1, z)] +
            // z == 1
            g1[OFFSET(x, y, z+1)] + g1[OFFSET(x+1, y, z+1)] +
            g1[OFFSET(x-1, y, z+1)] + g1[OFFSET(x, y+1, z+1)] +
            g1[OFFSET(x+1, y+1, z+1)] + g1[OFFSET(x-1, y+1, z+1)] +
            g1[OFFSET(x, y-1, z+1)] + g1[OFFSET(x+1, y-1, z+1)] +
            g1[OFFSET(x-1, y-1, z+1)];
        g2[OFFSET(x, y, z)] = v;
      }
    }
  }
  return;
}

REAL reduce(REAL *g) {
  REAL v = 0;
  int i;
  for (i = 0; i < N*N*N; ++i) {
    v += g[i];
  }
  return v;
}

int main(int argc, char *argv[]) {
  REAL *g1, *g2;  
  size_t nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  g2 = (REAL *)malloc(sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = i;
  }

  kernel(g1, g2);
  printf("%d\n", reduce(g2));

  kernel(g2, g1);
  printf("%d\n", reduce(g1));
  
  free(g1);
  free(g2);
  return 0;
}

