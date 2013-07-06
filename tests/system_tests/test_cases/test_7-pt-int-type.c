/*
 * TEST: Integer 7-point stencil
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 32

void kernel1(const int x, const int y, const int z,
             PSGrid3DInt g1, PSGrid3DInt g2) {
  int v = PSGridGet(g1, x, y, z) +
      PSGridGet(g1, x+1, y, z) + PSGridGet(g1, x-1, y, z) +
      PSGridGet(g1, x, y+1, z) + PSGridGet(g1, x, y-1, z) +
      PSGridGet(g1, x, y, z-1) + PSGridGet(g1, x, y, z+1);
  PSGridEmit(g2, v);
  return;
}

#define halo_width (1)

void dump(int *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%d\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DInt g1 = PSGrid3DIntNew(N, N, N);
  PSGrid3DInt g2 = PSGrid3DIntNew(N, N, N);  

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  int *indata = (int *)malloc(sizeof(int) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  int *outdata = (int *)malloc(sizeof(int) * nelms);  
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g1, g2));
  PSGridCopyout(g2, outdata);

  dump(outdata);

  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

