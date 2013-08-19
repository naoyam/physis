/*
 * TEST: Grids with different dimensions
 * DIM: 3
 * PRIORITY: 2
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

void kernel(const int x, const int y, const int z, PSGrid3DFloat g,
            PSGrid3DFloat g2, PSGrid1DFloat i, PSGrid1DFloat j, PSGrid1DFloat k) {
  float v = PSGridGet(g,x,y,z) +
      PSGridGet(g,x-1,y,z) * PSGridGet(i, x-1) +
      PSGridGet(g,x+1,y,z) * PSGridGet(i, x+1) +
      PSGridGet(g,x,y-1,z) * PSGridGet(j, y-1) +
      PSGridGet(g,x,y+1,z) * PSGridGet(j, y+1) +
      PSGridGet(g,x,y,z-1) * PSGridGet(k, z-1) +
      PSGridGet(g,x,y,z+1) * PSGridGet(k, z+1);
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
  PSGrid1DFloat cx = PSGrid1DFloatNew(N);
  PSGrid1DFloat cy = PSGrid1DFloatNew(N);
  PSGrid1DFloat cz = PSGrid1DFloatNew(N);  

  PSDomain3D d = PSDomain3DNew(1, N-1, 1, N-1, 1, N-1);
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
  PSGridCopyin(cx, indata);
  PSGridCopyin(cy, indata);
  PSGridCopyin(cz, indata);  

  PSStencilRun(PSStencilMap(kernel, d, g1, g2, cx, cy, cz));
    
  PSGridCopyout(g2, outdata);
  dump(outdata);  

  PSGridFree(g1);
  PSGridFree(g2);
  PSGridFree(cx);
  PSGridFree(cy);
  PSGridFree(cz);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

