#include <stdio.h>
#include <stdlib.h>

#define N 32
#define REAL float

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(float *g, int rb) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width + ((halo_width & 1) ^ (y + z + rb)%2);
           x < N-halo_width; x+=2) {
        float v = g[OFFSET(x, y, z)] +
            g[OFFSET(x+1, y, z)] + g[OFFSET(x-1, y, z)] +
            g[OFFSET(x, y+1, z)] + g[OFFSET(x, y-1, z)] +
            g[OFFSET(x, y, z-1)] + g[OFFSET(x, y, z+1)];
        g[OFFSET(x, y, z)] = v;
      }
    }
  }
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  REAL *g;  
  size_t nelms = N*N*N;
  g = (REAL *)malloc(sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g[i] = i;
  }

  kernel(g, 0);
  kernel(g, 1);  
  dump(g);
  
  free(g);
  return 0;
}

