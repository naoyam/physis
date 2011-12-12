#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)
#define PSGridGet(g, x, y, z) ((g)[OFFSET(x, y, z)])

void kernel(float *g1, float *g2) {
  int x, y, z;
  int halo_width = 1;
  for (z = 0; z < N; ++z) {
    for (y = 0; y < N; ++y) {
      for (x = 0; x < N; ++x) {
        float c, w, e, n, s, b, t;
        c = PSGridGet(g1, x, y, z);
        if (x == 0)    w = c; else w = PSGridGet(g1, x-1, y, z);
        if (x == N-1) e = c ; else e = PSGridGet(g1, x+1, y, z);
        if (y == 0)    n = c ; else n=PSGridGet(g1, x, y-1, z);
        if (y == N-1) s= c ; else s=PSGridGet(g1, x, y+1, z);
        if (z == 0)    b= c ; else b=PSGridGet(g1, x, y, z-1);
        if (z == N-1) t= c ; else t=PSGridGet(g1, x, y, z+1);
        g2[OFFSET(x, y, z)] = c + w + e + s + n + b + t;
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
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = i;
  }

  kernel(g1, g2);
  dump(g2);
  
  free(g1);
  free(g2);
  return 0;
}

