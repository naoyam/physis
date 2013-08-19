#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32
#define ITER 10

typedef struct {
  float x;
  float y;
  float z;
} Point;

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(Point *g1, Point *g2) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        float p = (g1[OFFSET(x, y, z)].x +
                   g1[OFFSET(x+1, y, z)].x +
                   g1[OFFSET(x-1, y, z)].x) / 3.3f;
        float q = (g1[OFFSET(x, y, z)].y +
                   g1[OFFSET(x, y+1, z)].y +
                   g1[OFFSET(x, y-1, z)].y +
                   g1[OFFSET(x, y, z+1)].y +
                   g1[OFFSET(x, y, z-1)].y) / 5.5f;
        float r = (g1[OFFSET(x, y, z)].z +
                   g1[OFFSET(x, y, z+1)].z +
                   g1[OFFSET(x, y, z-1)].z) / 3.3f;
        Point v = {p, q, r};
        g2[OFFSET(x, y, z)] = v;
      }
    }
  }
  return;
}

void dump(Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f %f\n", input[i].x, input[i].y, input[i].z);
  }
}

int main(int argc, char *argv[]) {
  Point *g1, *g2;  
  size_t nelms = N*N*N;
  g1 = (Point *)malloc(sizeof(Point) * nelms);
  g2 = (Point *)malloc(sizeof(Point) * nelms);

  int i;
  for (i = 0; i < nelms; i++) {
    g1[i].x = i;
    g1[i].y = i+1;
    g1[i].z = i+2;
  }

  memcpy(g2, g1, sizeof(Point) * nelms);

  for (i = 0; i < ITER; ++i) {
    kernel(g1, g2);
    Point *t = g1;
    g1 = g2;
    g2 = t;
  }
  dump(g1);
  
  free(g1);
  free(g2);
  return 0;
}

