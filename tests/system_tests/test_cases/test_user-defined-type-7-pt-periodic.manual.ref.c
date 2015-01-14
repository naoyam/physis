#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32

typedef struct {
  float p;
  float q;
} Point;


#define OFFSET(x, y, z) ((((x)+N)%N) + (((y)+N)%N) * N + (((z)+N)%N) * N * N)

void kernel1(Point *g) {
  int x, y, z;
  for (z = 0; z < N; ++z) {
    for (y = 0; y < N; ++y) {
      for (x = 0; x < N; ++x) {
        float v = g[OFFSET(x, y, z)].p +
                   g[OFFSET(x+1, y, z)].p +
                   g[OFFSET(x-1, y, z)].p +
                   g[OFFSET(x, y+1, z)].p +
                   g[OFFSET(x, y-1, z)].p +
                   g[OFFSET(x, y, z+1)].p +
                   g[OFFSET(x, y, z-1)].p;
        g[OFFSET(x, y, z)].q = v;
      }
    }
  }
  return;
}

void dump(Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f\n", input[i].p, input[i].q);
  }
}

int main(int argc, char *argv[]) {
  Point *g;
  size_t nelms = N*N*N;
  g = (Point *)malloc(sizeof(Point) * nelms);

  int i;
  for (i = 0; i < nelms; i++) {
    g[i].p = i;
    g[i].q = 0;
  }

  kernel1(g);
  dump(g);
  free(g);
  return 0;
}

