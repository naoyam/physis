#include <stdio.h>
#include <stdlib.h>

#define N 32

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel(float *g1, double *g2, int *c) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        double v = (double)g1[OFFSET(x, y, z)] +
            (double)g1[OFFSET(x+1, y, z)] +
            (double)g1[OFFSET(x-1, y, z)] +
            (double)g1[OFFSET(x, y+1, z)] +
            (double)g1[OFFSET(x, y-1, z)] +
            (double)g1[OFFSET(x, y, z-1)] +
            (double)g1[OFFSET(x, y, z+1)];
        g2[OFFSET(x, y, z)] = v * c[OFFSET(x, y, z)];
      }
    }
  }
  return;
}

void dump(double *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  float *g1;
  double *g2;
  int *c;
  size_t nelms = N*N*N;
  g1 = (float *)malloc(sizeof(float) * nelms);
  g2 = (double *)malloc(sizeof(double) * nelms);
  c = (int *)malloc(sizeof(int) * nelms);  

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
    g2[i] = 0;
    c[i] = i % 10;
  }

  kernel(g1, g2, c);
  dump(g2);
  
  free(g1);
  free(g2);
  free(c);
  return 0;
}

