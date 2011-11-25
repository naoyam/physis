#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(float *g1, float *g2) {
  int x, y, z;
  for (z = 0; z < N; ++z) {
    for (y = 0; y < N; ++y) {
      for (x = 0; x < N; ++x) {
        float c = g1[OFFSET(x, y, z)];
        float l = 0.0f;
        if (x > 0) {
          l = g1[OFFSET(x-1, y, z)];
        }
        if (x > 0) {
          l += g1[OFFSET(x-1, y, z)];
        } else {
          l += g1[OFFSET(x, y, z)];
        }
        if (x > 0) {
          l += g1[OFFSET(x-1, y, z)];
        } else {
          l += c;
        }
        if (x > 0 && x < N-1) {
          l += g1[OFFSET(x-1, y, z)] + g1[OFFSET(x+1, y, z)];
        } else {
          l += g1[OFFSET(x, y, z)];
        }
        if (x % 2 == 0) {
          l += g1[OFFSET(x, y, z)];
        }
        g2[OFFSET(x, y, z)] = c + l;
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

