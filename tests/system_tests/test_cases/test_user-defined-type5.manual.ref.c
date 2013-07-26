#include <stdio.h>
#include <stdlib.h>

#define N 32

#define T float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(T *g1, T *g2) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        T v = g1[OFFSET(x, y, z)] +
            g1[OFFSET(x+1, y, z)] + g1[OFFSET(x-1, y, z)] +
            g1[OFFSET(x, y+1, z)] + g1[OFFSET(x, y-1, z)] +
            g1[OFFSET(x, y, z-1)] + g1[OFFSET(x, y, z+1)];
        g2[OFFSET(x, y, z)] = v;
      }
    }
  }
  return;
}

void dump(T *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  T *g1, *g2;  
  size_t nelms = N*N*N;
  g1 = (T *)malloc(sizeof(T) * nelms);
  g2 = (T *)malloc(sizeof(T) * nelms);

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

