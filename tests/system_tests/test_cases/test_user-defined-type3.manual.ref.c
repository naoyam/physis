#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32
#define ITER 10

typedef struct {
  float p;
  float q;
  float r;
} Point;


#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel1(Point *g) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        float v = (g[OFFSET(x, y, z)].p +
                   g[OFFSET(x+1, y, z)].p +
                   g[OFFSET(x-1, y, z)].p +
                   g[OFFSET(x, y+1, z)].p +
                   g[OFFSET(x, y-1, z)].p +
                   g[OFFSET(x, y, z+1)].p +
                   g[OFFSET(x, y, z-1)].p +
                   g[OFFSET(x, y, z)].q +
                   g[OFFSET(x+1, y, z)].q +
                   g[OFFSET(x-1, y, z)].q +
                   g[OFFSET(x, y+1, z)].q +
                   g[OFFSET(x, y-1, z)].q +
                   g[OFFSET(x, y, z+1)].q +
                   g[OFFSET(x, y, z-1)].q) / 14.3;
        g[OFFSET(x, y, z)].r = v;
      }
    }
  }
  return;
}

void kernel2(Point *g) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        g[OFFSET(x, y, z)].p =
            (g[OFFSET(x, y, z)].p +
             g[OFFSET(x, y, z)].r) * 0.5;
        g[OFFSET(x, y, z)].q =
            (g[OFFSET(x, y, z)].q +
             g[OFFSET(x, y, z)].r) * 0.5;
      }
    }
  }
  return;
}

void dump(Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f %f\n", input[i].p, input[i].q, input[i].r);    
  }
}

int main(int argc, char *argv[]) {
  Point *g;
  size_t nelms = N*N*N;
  g = (Point *)malloc(sizeof(Point) * nelms);

  int i;
  for (i = 0; i < nelms; i++) {
    g[i].p = i;
    g[i].q = i+1;
    g[i].r = 0;
  }

  for (i = 0; i < ITER; ++i) {
    kernel1(g);
    kernel2(g);    
  }
  dump(g);
  free(g);
  return 0;
}

