#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET(x) (x)

void kernel(float *g1, float *g2) {
  int x;
  for (x = 1; x < N-1; ++x) {
    float v = g1[x-1] + g1[x] + g1[x+1];
    g2[x] = v;
  }
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g1, *g2;  
  size_t nelms = N;
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

