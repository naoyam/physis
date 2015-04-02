#include <stdio.h>
#include <stdlib.h>

#define N 32
#define TYPE int

#define OFFSET(x, y) ((x) + (y) * N)

void kernel(TYPE *g1, TYPE *g2) {
  int x, y;
  for (y = 1; y < N-1; ++y) {  
    for (x = 1; x < N-1; ++x) {
      float v = g1[OFFSET(x, y)] +
          g1[OFFSET(x-1, y)] +
          g1[OFFSET(x+1, y)] +
          g1[OFFSET(x, y-1)] +
          g1[OFFSET(x, y+1)] +
          g1[OFFSET(x-1, y-1)] +
          g1[OFFSET(x+1, y-1)] +
          g1[OFFSET(x-1, y+1)] +
          g1[OFFSET(x+1, y+1)];
      g2[OFFSET(x, y)] = v;
    }
  }
  return;
}

TYPE reduce(TYPE *g) {
  TYPE v = 0;
  int i;
  for (i = 0; i < N*N; ++i) {
    v += g[i];
  }
  return v;
}

int main(int argc, char *argv[]) {
  TYPE *g1, *g2;  
  size_t nelms = N*N;
  g1 = (TYPE *)malloc(sizeof(TYPE) * nelms);
  g2 = (TYPE *)malloc(sizeof(TYPE) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = i;
  }

  kernel(g1, g2);
  kernel(g2, g1);
  printf("%d\n", reduce(g1));
  
  free(g1);
  free(g2);
  return 0;
}

