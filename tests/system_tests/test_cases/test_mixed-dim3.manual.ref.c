#include <stdio.h>
#include <stdlib.h>

#define N 32
#define M (N+2)
#define REAL float

#define OFFSET1D(x) (x)
#define OFFSET2D(x, y) ((x) + (y) * M)
#define OFFSET3D(x, y, z) ((x) + (y) * N + (z) * N * M)

void kernel(float *g1, float *g2,
            float *i, float *j) {
  int x, y, z;
  for (z = 1; z < N-1; ++z) {
    for (y = 1; y < M-1; ++y) {
      for (x = 1; x < N-1; ++x) {
        float v =
            g1[OFFSET3D(x, y, z)] +
            g1[OFFSET3D(x-1, y, z)] * i[OFFSET1D(x-1)] +
            g1[OFFSET3D(x+1, y, z)] * i[OFFSET1D(x+1)] +
            g1[OFFSET3D(x, y-1, z)] * j[OFFSET2D(y-1, z)] +
            g1[OFFSET3D(x, y+1, z)] * j[OFFSET2D(y+1, z)] +
            g1[OFFSET3D(x, y, z-1)] * j[OFFSET2D(y, z-1)] +
            g1[OFFSET3D(x, y, z+1)] * j[OFFSET2D(y, z+1)];
        g2[OFFSET3D(x, y, z)] = v;
      }
    }
  }
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*M*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g2, *ci, *cj;
  size_t nelms = N*M*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  g2 = (REAL *)malloc(sizeof(REAL) * nelms);
  ci = (REAL *)malloc(sizeof(REAL) * N);
  cj = (REAL *)malloc(sizeof(REAL) * M*N);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = i;
  }

  for (i = 0; i < N; ++i) {
    ci[i] = 1 + (i%2); // 1 or 2    
  }

  for (i = 0; i < M*N; ++i) {
    cj[i] = 1 + (i%2); // 1 or 2
  }

  kernel(g1, g2, ci, cj);
  dump(g2);
  
  free(g1);
  free(g2);
  free(ci);
  free(cj);  
  return 0;
}

