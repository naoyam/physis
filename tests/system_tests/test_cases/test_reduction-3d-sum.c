/*
 * TEST: Grid reduction OP=PS_SUM
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 16
#define REAL float
#define PSGrid3D PSGrid3DFloat
#define PSGrid3DNew PSGrid3DFloatNew

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3D g1 = PSGrid3DNew(N, N, N);
  size_t nelms = N*N*N;
  float *indata = (float *)malloc(sizeof(REAL) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  PSGridCopyin(g1, indata);
  float v;
  PSReduce(&v, PS_SUM, g1);
  printf("%f\n", v);
  PSGridFree(g1);
  PSFinalize();
  free(indata);
  return 0;
}

