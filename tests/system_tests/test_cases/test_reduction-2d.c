/*
 * TEST: Grid reduction OP=PS_SUM
 * DIM: 2
 * PRIORITY: 1
 */

#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 4
#define REAL double
#define PSGrid2D PSGrid2DDouble
#define PSGrid2DNew PSGrid2DDoubleNew

REAL reduce(REAL *g) {
  REAL v = 0.0;
  int i;
  for (i = 0; i < N*N; ++i) {
    v += g[i];
  }
  return v;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 2, N, N);
  PSGrid2D g1 = PSGrid2DNew(N, N);
  size_t nelms = N*N;
  REAL *indata = (REAL *)malloc(sizeof(REAL) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  PSGridCopyin(g1, indata);
  REAL v;
  PSReduce(&v, PS_SUM, g1);
  REAL v_ref = reduce(indata);
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

