/*
 * TEST: 3-point stencil with 1-D grids
 * DIM: 1
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

static void kernel(const int x, PSGrid1DFloat g, PSGrid1DFloat g2) {
  float v =
      PSGridGet(g,x-1) +
      PSGridGet(g,x) + 
      PSGridGet(g,x+1);
  PSGridEmit(g2, v);
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 1, N);
  PSGrid1DFloat g1 = PSGrid1DFloatNew(N);
  PSGrid1DFloat g2 = PSGrid1DFloatNew(N);

  PSDomain1D d = PSDomain1DNew(1, N-1);
  size_t nelms = N;
  
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

