#include <stdio.h>
#include <stdlib.h>

#define N 32

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

struct Point {
  float p[3][2];
};

#define T float

void kernel(struct Point *g1) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        T v1 = g1[OFFSET(x, y, z)].p[0][0] +
            g1[OFFSET(x+1, y, z)].p[0][0] + g1[OFFSET(x-1, y, z)].p[0][0] +
            g1[OFFSET(x, y+1, z)].p[0][0] + g1[OFFSET(x, y-1, z)].p[0][0] +
            g1[OFFSET(x, y, z-1)].p[0][0] + g1[OFFSET(x, y, z+1)].p[0][0];
        T v2 = g1[OFFSET(x, y, z)].p[2][1] +
            g1[OFFSET(x+1, y, z)].p[2][1] + g1[OFFSET(x-1, y, z)].p[2][1] +
            g1[OFFSET(x, y+1, z)].p[2][1] + g1[OFFSET(x, y-1, z)].p[2][1] +
            g1[OFFSET(x, y, z-1)].p[2][1] + g1[OFFSET(x, y, z+1)].p[2][1];
        g1[OFFSET(x, y, z)].p[1][0] = v1+v2;
      }
    }
  }
  return;
}

void dump(struct Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i].p[1][0]);
  }
}

int main(int argc, char *argv[]) {
  int nelms = N*N*N;
  struct Point *indata = (struct Point *)malloc(sizeof(struct Point) * nelms);
  struct Point *outdata = (struct Point *)malloc(sizeof(struct Point) * nelms);

  int i;
  for (i = 0; i < nelms; i++) {
    int j;
    for (j = 0; j < 3; ++j) {
      int k;
      for (k = 0; k < 2; ++k) {
        indata[i].p[j][k] = i+j+k;
      }
    }
  }

  kernel(indata);
  dump(indata);
  
  free(indata);
  return 0;
}

