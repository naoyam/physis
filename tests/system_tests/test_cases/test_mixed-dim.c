/*
 * TEST: Grids with different dimensions
 * DIM: 3
 * PRIORITY: 2
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

void kernel(const int x, const int y, const int z, PSGrid3DFloat g,
            PSGrid3DFloat g2, PSGrid1DFloat k) {
  float v = PSGridGet(g,x,y,z) * PSGridGet(k, x) +
      PSGridGet(g,x-1,y,z) * PSGridGet(k, x-1) +
      PSGridGet(g,x+1,y,z) * PSGridGet(k, x+1);
  PSGridEmit(g2, v);
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);
  PSGrid1DFloat k = PSGrid1DFloatNew(N);

  PSDomain3D d = PSDomain3DNew(1, N-1, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);

  for (i = 0; i < N; ++i) {
    indata[i] = 1 + (i%2); // 1 or 2
  }

  PSGridCopyin(k, indata);

  PSStencilRun(PSStencilMap(kernel, d, g1, g2, k));
    
  PSGridCopyout(g2, outdata);
  dump(outdata);  

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

