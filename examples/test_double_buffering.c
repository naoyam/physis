#include <stdio.h>
#include "physis/physis.h"

#define N 8

void kernel1(const int x, const int y, const int z,
            PSGrid3DFloat g1, PSGrid3DFloat g2) {
  float v = PSGridGet(g1, x, y, z) * 2;
  PSGridEmit(g2, v);
  return;
}

void kernel2(const int x, const int y, const int z,
            PSGrid3DFloat g1, PSGrid3DFloat g2) {
  float v = PSGridGet(g1, x, y, z) * 2;
  PSGridEmit(g2, v);
  return;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);  
  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;

#if 0  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g, indata);
#endif  

#if 1
  PSStencilRun(PSStencilMap(kernel1, d, g1, g2),
               PSStencilMap(kernel1, d, g2, g1));
#else
  PSStencilRun(PSStencilMap(kernel1, d, g1, g2),
               PSStencilMap(kernel2, d, g2, g1));
#endif  
  
#if 0  
  PSGridCopyout(g, outdata);
    
  for (i = 0; i < nelms; i++) {
    if (indata[i] * 2 != outdata[i]) {
      fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
              i, indata[i], outdata[i]);
    }
  }

  PSGridFree(g);
#endif
  
  PSFinalize();
  return 0;
}

