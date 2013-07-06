/*
 * TEST: 7-point stencil with multiple types
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include <stdlib.h>
#include "physis/physis.h"

#define N 32

void kernel1(const int x, const int y, const int z,
             PSGrid3DFloat g1, PSGrid3DDouble g2,
             PSGrid3DInt coeff) {
  double v = (double)PSGridGet(g1, x, y, z)
      + (double)PSGridGet(g1, x+1, y, z)
      + (double)PSGridGet(g1, x-1, y, z)
      + (double)PSGridGet(g1, x, y+1, z)
      + (double)PSGridGet(g1, x, y-1, z)
      + (double)PSGridGet(g1, x, y, z-1)
      + (double)PSGridGet(g1, x, y, z+1);
  PSGridEmit(g2, v * PSGridGet(coeff, x, y, z));
  return;
}

#define halo_width (1)

void dump(double *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DDouble g2 = PSGrid3DDoubleNew(N, N, N);
  PSGrid3DInt coeffg = PSGrid3DIntNew(N, N, N);  

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int *coeff = (int *)malloc(sizeof(int)*nelms);  
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
    coeff[i] = i % 10;
  }
  double *outdata = (double *)malloc(sizeof(double) * nelms);

  PSGridCopyin(g1, indata);
  PSGridCopyin(coeffg, coeff);

  PSStencilRun(PSStencilMap(kernel1, d, g1, g2, coeffg));
  PSGridCopyout(g2, outdata);

  dump(outdata);

  PSGridFree(g1);
  PSGridFree(g2);
  PSGridFree(coeffg);  
  PSFinalize();
  free(indata);
  free(outdata);
  free(coeff);
  return 0;
}

