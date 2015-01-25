#include <stdio.h>
#include <stdlib.h>

#define N 16
#define REAL float

REAL reduce(float *input) {
  int i;
  REAL v = 0;
  for (i = 0; i < N*N*N; ++i) {
    v += input[i];
  }
  return v;
}

int main(int argc, char *argv[]) {
  REAL *g1;
  size_t nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }

  REAL v = reduce(g1);
  printf("%f\n", v);
  
  free(g1);
  return 0;
}

