/*
 * TEST: 9-point stencil with reduction
 * DIM: 2
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32
#define TYPE int
#define PSGridType PSGrid2DInt

static void kernel(const int x, const int y, PSGridType g, PSGridType g2) {
  float v =
      PSGridGet(g,x, y) + 
      PSGridGet(g,x-1, y) +
      PSGridGet(g,x+1, y) +
      PSGridGet(g,x, y-1) +
      PSGridGet(g,x, y+1) +
      PSGridGet(g,x-1, y-1) +
      PSGridGet(g,x+1, y-1) +
      PSGridGet(g,x-1, y+1) +
      PSGridGet(g,x+1, y+1);
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
  PSGridType g1 = PSGrid2DIntNew(N, N);
  PSGridType g2 = PSGrid2DIntNew(N, N);

  PSDomain2D d = PSDomain2DNew(1, N-1, 1, N-1);
  size_t nelms = N * N;
  
  TYPE *indata = (TYPE *)malloc(sizeof(TYPE) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  TYPE *outdata = (TYPE *)malloc(sizeof(TYPE) * nelms);
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);  

  PSStencilRun(PSStencilMap(kernel, d, g1, g2),
               PSStencilMap(kernel, d, g2, g1));
  int v;
  PSReduce(&v, PS_SUM, g1);
  printf("%d\n", v);

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

