#include <stdio.h>
#include <stdlib.h>

#define N 32

#define OFFSET(x, y, z) ((x) + (y) * N + (z) * N * N)

struct Point {
  float p[2];
  float q;
  float r;
};

#define T float

void kernel(struct Point *g1, struct Point *g2) {
  int x, y, z;
  int halo_width = 1;
  for (z = halo_width; z < N-halo_width; ++z) {
    for (y = halo_width; y < N-halo_width; ++y) {
      for (x = halo_width; x < N-halo_width; ++x) {
        T v1 = g1[OFFSET(x, y, z)].p[0] +
            g1[OFFSET(x+1, y, z)].p[0] + g1[OFFSET(x-1, y, z)].p[0] +
            g1[OFFSET(x, y+1, z)].p[0] + g1[OFFSET(x, y-1, z)].p[0] +
            g1[OFFSET(x, y, z-1)].p[0] + g1[OFFSET(x, y, z+1)].p[0];
        T v2 = g1[OFFSET(x, y, z)].p[1] +
            g1[OFFSET(x+1, y, z)].p[1] + g1[OFFSET(x-1, y, z)].p[1] +
            g1[OFFSET(x, y+1, z)].p[1] + g1[OFFSET(x, y-1, z)].p[1] +
            g1[OFFSET(x, y, z-1)].p[1] + g1[OFFSET(x, y, z+1)].p[1];
        T v3 = g1[OFFSET(x, y, z)].q +
            g1[OFFSET(x+1, y, z)].q + g1[OFFSET(x-1, y, z)].q +
            g1[OFFSET(x, y+1, z)].q + g1[OFFSET(x, y-1, z)].q +
            g1[OFFSET(x, y, z-1)].q + g1[OFFSET(x, y, z+1)].q;
        g2[OFFSET(x, y, z)].r = v1+v2+v3;
      }
    }
  }
  return;
}

void dump(struct Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i].r);
  }
}

int main(int argc, char *argv[]) {
  size_t nelms = N*N*N;
  struct Point *indata = (struct Point *)malloc(sizeof(struct Point) * nelms);
  struct Point *outdata = (struct Point *)malloc(sizeof(struct Point) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    indata[i].p[0] = i;
    indata[i].p[1] = i+1;
    indata[i].q = i+2;
    indata[i].r = 0;
    outdata[i].p[0] = 0;
    outdata[i].p[1] = 0;
    outdata[i].q = 0;
    outdata[i].r = 0;
  }

  kernel(indata, outdata);
  dump(outdata);
  
  free(indata);
  free(outdata);
  return 0;
}

