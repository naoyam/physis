/*
 * TEST: copyin and copyout
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 8

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g3 = PSGrid3DFloatNew(N, N, N);
    
  float *indata = (float *)malloc(sizeof(float) * N * N * N);
  int i;
  for (i = 0; i < N*N*N; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * N * N * N);
    
  PSGridCopyin(g3, indata);
  PSGridCopyout(g3, outdata);
    
  for (i = 0; i < N*N*N; i++) {
    if (indata[i] != outdata[i]) {
      fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
              i, indata[i], outdata[i]);
      exit(1);
    }
  }

  PSGridFree(g3);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

