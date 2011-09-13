#include <stdio.h>
#include "physis/physis.h"

#define N 8

void kernel1(const int x, const int y, const int z, PSGrid3DFloat g) {
  float v = PSGridGet(g, x, y, z) +
      PSGridGet(g, x+1, y, z) + PSGridGet(g, x-1, y, z) +
      PSGridGet(g, x, y+1, z) + PSGridGet(g, x, y-1, z) +
      PSGridGet(g, x, y, z-1) + PSGridGet(g, x, y, z+1);
  PSGridEmit(g, v);
  return;
}

#define GET(g, x, y, z) g[(x) + (y) * N + (z) * N * N]

void kernel_ref(float *input, float *output, int halo_width) {
  int i, j, k;
  for (i = halo_width; i < N-halo_width; ++i) {
    for (j = halo_width; j < N-halo_width; ++j) {
      for (k = halo_width; k < N-halo_width; ++k) {
        float v = GET(input, i, j, k) +
            GET(input, i-1, j, k) + GET(input, i+1, j, k) +
            GET(input, i, j-1, k) + GET(input, i, j+1, k) +
            GET(input, i, j, k-1) + GET(input, i, j, k+1);
        GET(output, i, j, k) = v;
      }
    }
  }
  return;
}

void check(float *input, float *output) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    if (input[i] != output[i]) {
      printf("Mismatch at %d: %f != %f\n", i,
             input[i], output[i]);
    }
  }
}

#define halo_width (1)

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g = PSGrid3DFloatNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);                          size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata_ps = (float *)malloc(sizeof(float) * nelms);
  float *outdata = (float *)malloc(sizeof(float) * nelms);  
    
  PSGridCopyin(g, indata);
  
  int iter = 1 * 2;
  
  PSStencilRun(PSStencilMap(kernel1, d, g), iter);

  for (i = 0 ; i < iter; i+=2) {
    kernel_ref(indata, outdata, halo_width);
    kernel_ref(outdata, indata, halo_width);
  }
    
  PSGridCopyout(g, outdata_ps);

  check(indata, outdata_ps);

  PSGridFree(g);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

