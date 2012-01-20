#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(float *g1, float *g2) {
  int x, y, z;
  for (z = 0; z < N; ++z) {
    int zp = ((z - 1) + N) % N;
    int zn = (z + 1) % N;
    for (y = 0; y < N; ++y) {
      int yp = ((y - 1) + N) % N;
      int yn = (y + 1) % N;
      for (x = 0; x < N; ++x) {
        int xp = ((x - 1) + N) % N;
        int xn = (x + 1) % N;
        float v =
            g1[OFFSET(x, y, z)] +
            g1[OFFSET(xn, y, z)] +
            g1[OFFSET(xp, y, z)] +
            g1[OFFSET(x, yn, z)] +
            g1[OFFSET(x, yp, z)] +
            g1[OFFSET(x, y, zn)] +
            g1[OFFSET(x, y, zp)];
        g2[OFFSET(x, y, z)] = v;
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

