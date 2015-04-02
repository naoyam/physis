#include <stdio.h>
#include <stdlib.h>

#define N 32
#define TYPE int

#define OFFSET(x, y) ((x) + (y) * N)

void kernel(TYPE *g1, TYPE *g2) {
  int x, y;
  for (y = 0; y < N; ++y) {
    int yp = ((y - 1) + N) % N;
    int yn = ((y + 1) + N) % N;    
    for (x = 0; x < N; ++x) {
      int xp = ((x - 1) + N) % N;
      int xn = ((x + 1) + N) % N;
      float v = g1[OFFSET(x, y)] +
          g1[OFFSET(xp, y)] +
          g1[OFFSET(xn, y)] +
          g1[OFFSET(x, yp)] +
          g1[OFFSET(x, yn)] +
          g1[OFFSET(xp, yp)] +
          g1[OFFSET(xn, yp)] +
          g1[OFFSET(xp, yn)] +
          g1[OFFSET(xn, yn)];
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

