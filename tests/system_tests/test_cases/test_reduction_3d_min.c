/*
 * TEST: Grid reduction OP=PS_MIN
 * DIM: 3
 * PRIORITY: 2
 */

#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 8
#define REAL float
#define PSGrid3D PSGrid3DFloat
#define PSGrid3DNew PSGrid3DFloatNew

REAL reduce(REAL *g) {
  REAL v = g[0];
  int i;
  for (i = 1; i < N*N*N; ++i) {
    v = (v > g[i]) ? g[i] : v;
  }
  return v;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3D g1 = PSGrid3DNew(N, N, N);
  size_t nelms = N*N*N;
  float *indata = (float *)malloc(sizeof(REAL) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i+10;
  }
  PSGridCopyin(g1, indata);
  float v;
  PSReduce(&v, PS_MIN, g1);
  float v_ref = reduce(indata);
  fprintf(stderr, "Reduction result: %f, reference: %f\n", v, v_ref);
  if (v != v_ref) {
    fprintf(stderr, "Error: Non matching result\n");
    exit(1);
  }
  PSGridFree(g1);
  PSFinalize();
  free(indata);
  return 0;
}

