#include <stdio.h>
#include "physis/physis.h"

#define N 8

PSGrid3DFloat g;

void kernel(const int x, const int y, const int z, PSGrid3DFloat g) {
  float v = PSGridGet(g, x, y, z) * 2;
  PSGridEmit(g, v);
  return;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  g = PSGrid3DFloatNew(N, N, N);
  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g, indata);

  PSStencilRun(PSStencilMap(kernel, d, g));
    
  PSGridCopyout(g, outdata);
    
  for (i = 0; i < nelms; i++) {
    if (indata[i] * 2 != outdata[i]) {
      fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
              i, indata[i], outdata[i]);
    }
  }

  PSGridFree(g);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

