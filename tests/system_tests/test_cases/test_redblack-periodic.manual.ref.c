#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(float *g, int rb) {
  int x, y, z;
  for (z = 0; z < N; ++z) {
    int zp = ((z - 1) + N) % N;
    int zn = (z + 1) % N;
    for (y = 0; y < N; ++y) {
      int yp = ((y - 1) + N) % N;
      int yn = (y + 1) % N;
      for (x = (y+z+rb)%2 ; x < N; x+=2) {
        int xp = ((x - 1) + N) % N;
        int xn = (x + 1) % N;
        float v =
            g[OFFSET(x, y, z)] +
            g[OFFSET(xn, y, z)] +
            g[OFFSET(xp, y, z)] +
            g[OFFSET(x, yn, z)] +
            g[OFFSET(x, yp, z)] +
            g[OFFSET(x, y, zn)] +
            g[OFFSET(x, y, zp)];
        g[OFFSET(x, y, z)] = v;
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
  REAL *g;
  size_t nelms = N*N*N;
  g = (REAL *)malloc(sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g[i] = i;
  }

  kernel(g, 0);
  kernel(g, 1);  
  dump(g);
  
  free(g);
  return 0;
}

