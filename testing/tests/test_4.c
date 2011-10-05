/*
 * TEST: 7-point stencil
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 8

void kernel1(const int x, const int y, const int z, PSGrid3DFloat g,
             PSGrid3DFloat g2) {
  float v = PSGridGet(g, x, y, z) +
      PSGridGet(g, x+1, y+1, z) + PSGridGet(g, x-1, y, z) +
      PSGridGet(g, x, y+1, z) + PSGridGet(g, x, y-1, z) +
      PSGridGet(g, x, y, z-1) + PSGridGet(g, x, y, z+1);
  PSGridEmit(g2, v);
  return;
}

#define GET(g, x, y, z) g[(x) + (y) * N + (z) * N * N]

void check(float *input, float *output, int halo_width) {
  int i, j, k;
  for (i = halo_width; i < N-halo_width; ++i) {
    for (j = halo_width; j < N-halo_width; ++j) {
      for (k = halo_width; k < N-halo_width; ++k) {
        float v = GET(input, i, j, k) +
            GET(input, i-1, j, k) + GET(input, i+1, j+1, k) +
            GET(input, i, j-1, k) + GET(input, i, j+1, k) +
            GET(input, i, j, k-1) + GET(input, i, j, k+1);
        if (v != GET(output, i, j, k)) {
          printf("ERROR at (%d, %d, %d): %f != %f\n",
                 i, j, k, v, GET(output, i, j, k));
          exit(1);
        }
      }
    }
  }
}
#define halo_width (1)

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g, g2));
    
  PSGridCopyout(g2, outdata);

  check(indata, outdata, halo_width);

  PSGridFree(g);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

