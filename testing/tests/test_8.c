/*
 * TEST: Accessing a 2D plane in a 3D grid
 * DIM: 3
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 4

void kernel(const int x, const int y, const int z,
            PSGrid3DFloat g1, PSGrid3DFloat g2) {
  float v = PSGridGet(g2, x, y, 0);
  PSGridEmit(g1, v);
  return;
}

#define IDX3(x, y, z) ((x) + (y) * N + (z) * N * N)
#define IDX2(x, y) ((x) + (y) * N)

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, 1);
  
  size_t nelms = N*N*N;
  int i, j, k;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);

  for (i = 0; i < N*N; i++) {
    indata[i] = i;
  }
  PSGridCopyin(g2, indata);

  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  PSStencilRun(PSStencilMap(kernel, d, g1, g2), 1);
  
  float *outdata = (float *)malloc(sizeof(float) * nelms);
  PSGridCopyout(g1, outdata);

  for (k = 0; k < N; ++k) {
    for (j = 0; j < N; ++j) {    
      for (i = 0; i < N; ++i) {
        if (indata[IDX2(i, j)] != outdata[IDX3(i, j, k)]) {
          printf("Error: mismatch at %d,%d,%d, in: %f, out: %f\n",
                 i, j, k, indata[IDX2(i, j)], outdata[IDX3(i, j, k)]);
          exit(1);
        }
      }
    }
  }
  
  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();

  free(indata);
  free(outdata);
  return 0;
}

