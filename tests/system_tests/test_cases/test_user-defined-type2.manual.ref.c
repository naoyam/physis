#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32
#define ITER 10

typedef struct {
  float p;
  float q;
} Point;


#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel1(Point *g) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        float v = g[OFFSET(x, y, z)].p;
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

