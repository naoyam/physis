/*
 * TEST: Conditional get optimization
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

void kernel(const int x, const int y, const int z, PSGrid3DFloat g,
            PSGrid3DFloat g2) {
  float c = PSGridGet(g, x, y, z);
  float l = 0.0f;
  if (x > 0) {
    l = PSGridGet(g, x-1, y, z);
  }
  if (x > 0) {
    l += PSGridGet(g, x-1, y, z);
  } else {
    l += PSGridGet(g, x, y, z);
  }
  if (x > 0) {
    l += PSGridGet(g, x-1, y, z);
  } else {
    l += c;
  }
  if (x > 0 && x < N-1) {
    l += PSGridGet(g, x-1, y, z) + PSGridGet(g, x+1, y, z);
  } else {
    l += PSGridGet(g, x, y, z);
  }
  if (x % 2 == 0) {
    l += PSGridGet(g, x, y, z);
  }
  PSGridEmit(g2, c+l);
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

#define halo_width (1)

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0, N,
                               0, N,
                               0, N);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);  

  PSStencilRun(PSStencilMap(kernel, d, g1, g2));
    
  PSGridCopyout(g2, outdata);
  dump(outdata);  

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

