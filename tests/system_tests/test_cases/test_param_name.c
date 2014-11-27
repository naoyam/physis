/*
 * TEST: Parameter name being the same as an existing function
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include <math.h> // gamma is declared in math.h
#include "physis/physis.h"

#define N 8

// Use "gamma" as a parameter. This causes a CUDA translation error in
// commit 61d14b3e6362e7154d7da3dbdd7887a2106240f4.
void kernel(const int x, const int y, const int z, PSGrid3DFloat g,
             PSGrid3DFloat g2, float gamma) {
  float v = PSGridGet(g, x, y, z) * gamma;
  PSGridEmit(g2, v);
  return;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);  
  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g, indata);

  PSStencilRun(PSStencilMap(kernel, d, g, g2, 1.0f));
    
  PSGridCopyout(g2, outdata);
    
  for (i = 0; i < nelms; i++) {
    if (indata[i] != outdata[i]) {
      fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
              i, indata[i], outdata[i]);
      exit(1);
    }
  }

  PSGridFree(g);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

