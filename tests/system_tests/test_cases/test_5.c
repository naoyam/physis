/*
 * TEST: Combining two kernels
 * DIM: 3
 */ 

#include <stdio.h>
#include "physis/physis.h"

#define N 8

void kernel1(const int x, const int y, const int z,
             PSGrid3DFloat g1, PSGrid3DFloat g2) {
  float v = PSGridGet(g1, x, y, z) * 2;
  PSGridEmit(g2, v);
  return;
}

void kernel2(const int x, const int y, const int z,
             PSGrid3DFloat g2, PSGrid3DFloat g1) {
  float v = PSGridGet(g2, x, y, z) / 2;
  PSGridEmit(g1, v);
  return;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);  
  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g1, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g1, g2),
               PSStencilMap(kernel2, d, g2, g1));
    
  PSGridCopyout(g1, outdata);
    
  for (i = 0; i < nelms; i++) {
    if (indata[i] != outdata[i]) {
      fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
              i, indata[i], outdata[i]);
      exit(1);
    }
  }

  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

