/*
 * TEST: Long grid reduction OP=PS_SUM
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 8
#define NN (N*N*N)
#define GTYPE long
#define FMT "%d"
#define PSGrid3D PSGrid3DLong
#define PSGrid3DNew PSGrid3DLongNew

GTYPE reduce(GTYPE *g) {
  GTYPE v = 0.0;
  int i;
  for (i = 0; i < NN; ++i) {
    v += g[i];
  }
  return v;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3D g1 = PSGrid3DNew(N, N, N);
  GTYPE *indata = (GTYPE *)malloc(sizeof(GTYPE) * NN);
  int i;
  for (i = 0; i < NN; i++) {
    indata[i] = i;
  }
  PSGridCopyin(g1, indata);
  GTYPE v;
  PSReduce(&v, PS_SUM, g1);
  GTYPE v_ref = reduce(indata);
  fprintf(stderr, "Reduction result: " FMT ", reference: " FMT "\n", v, v_ref);
  if (v != v_ref) {
    fprintf(stderr, "Error: No matching result\n");
    exit(1);
  }
  PSGridFree(g1);
  PSFinalize();
  free(indata);
  return 0;
}

