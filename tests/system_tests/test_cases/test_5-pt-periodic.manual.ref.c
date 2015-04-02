#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET(x, y) ((x) + (y) * N)

void kernel(float *g1, float *g2) {
  int x, y;
  for (y = 0; y < N; ++y) {
    int yp = ((y - 1) + N) % N;
    int yn = (y + 1) % N;
    for (x = 0; x < N; ++x) {
      int xp = ((x - 1) + N) % N;
      int xn = (x + 1) % N;
      float v =
          g1[OFFSET(x, y)] +
          g1[OFFSET(xn, y)] +
          g1[OFFSET(xp, y)] +
          g1[OFFSET(x, yn)] +
          g1[OFFSET(x, yp)];
      g2[OFFSET(x, y)] = v;
    }
  }
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g2;  
  size_t nelms = N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  g2 = (REAL *)malloc(sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = i;
  }

  kernel(g1, g2);
  kernel(g2, g1);
  
  dump(g1);
  
  free(g1);
  free(g2);
  return 0;
}

