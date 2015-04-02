/*
 * TEST: 7-point periodic-boundary stencil
 * DIM: 2
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

void kernel(const int x, const int y, PSGrid2DFloat g,
            PSGrid2DFloat g2) {
  float v =
      PSGridGetPeriodic(g,x,y) +
      PSGridGetPeriodic(g,x+1,y) +
      PSGridGetPeriodic(g,x-1,y) +
      PSGridGetPeriodic(g,x,y+1) +
      PSGridGetPeriodic(g,x,y-1);
  PSGridEmit(g2, v);
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 2, N, N);
  PSGrid2DFloat g1 = PSGrid2DFloatNew(N, N);
  PSGrid2DFloat g2 = PSGrid2DFloatNew(N, N);

  PSDomain2D d = PSDomain2DNew(0, N, 0, N);
  size_t nelms = N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);  

  PSStencilRun(PSStencilMap(kernel, d, g1, g2),
               PSStencilMap(kernel, d, g2, g1));
    
  PSGridCopyout(g1, outdata);
  dump(outdata);  

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

