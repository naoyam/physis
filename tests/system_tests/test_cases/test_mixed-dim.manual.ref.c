#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET1D(x) (x)
#define OFFSET3D(x, y, z) ((x) + (y) * N + (z) * N * N)


void kernel(float *g1, float *g2, float *k) {
  int x, y, z;
  for (z = 0; z < N; ++z) {
    for (y = 0; y < N; ++y) {
      for (x = 1; x < N-1; ++x) {
        float v =
            g1[OFFSET3D(x, y, z)] * k[OFFSET1D(x)] +
            g1[OFFSET3D(x-1, y, z)] * k[OFFSET1D(x-1)] +
            g1[OFFSET3D(x+1, y, z)] * k[OFFSET1D(x+1)];
        g2[OFFSET3D(x, y, z)] = v;
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
  REAL *g1, *g2, *k;
  size_t nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  g2 = (REAL *)malloc(sizeof(REAL) * nelms);
  k = (REAL *)malloc(sizeof(REAL) * N);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = i;
  }

  for (i = 0; i < N; ++i) {
    k[i] = 1 + (i%2); // 1 or 2    
  }

  kernel(g1, g2, k);
  dump(g2);
  
  free(g1);
  free(g2);
  free(k);
  return 0;
}

