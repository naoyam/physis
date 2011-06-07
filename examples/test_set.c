#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 2

#define IDX3(x, y, z) ((x) + (y) * N + (z) * N * N)

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g = PSGrid3DFloatNew(N, N, N);
  size_t nelms = N*N*N;
  
  int i, j, k;
  float v = 0;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      for (k = 0; k < N; ++k) {
        PSGridSet(g, i, j, k, v);
        ++v;
      }
    }
  }
  
  float *outdata = (float *)malloc(sizeof(float) * nelms);
  PSGridCopyout(g, outdata);

  v = 0;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      for (k = 0; k < N; ++k) {
        if (outdata[IDX3(i, j, k)] != v) {
          fprintf(stderr, "Error: mismatch at %d:%d:%d, in: %f, out: %f\n",
                  i, j, k, outdata[IDX3(i,j,k)], v);
        }
        ++v;
      }
    }
  }

  PSGridFree(g);
  PSFinalize();
  free(outdata);
  return 0;
}

