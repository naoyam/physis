#include <stdio.h>
#include "physis/physis.h"

#define N 2

void kernel1(const int x, const int y, const int z,
             PSGrid3DFloat g1, PSGrid3DFloat g2) {
  float v = PSGridGet(g2, x, y, z) + PSGridGet(g2, x+1, y, z)
            + PSGridGet(g2, x, y+1, z) + PSGridGet(g2, x, y, z+1)
            + PSGridGet(g2, x+1, y+1, z) + PSGridGet(g2, x+1, y, z+1)
            + PSGridGet(g2, x, y+1, z+1) + PSGridGet(g2, x+1, y+1, z+1);
  PSGridEmit(g1, v);
  return;
}

void dump(float *buf, size_t len, FILE *out) {
  int i;
  for (i = 0; i < len; ++i) {
    fprintf(out, "%f\n", buf[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N+1, N+1, N+1);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N+1, N+1, N+1);  
  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms1 = N*N*N;
  size_t nelms2 = (N+1)*(N+1)*(N+1);  
  
  float *indata = (float *)malloc(sizeof(float) * nelms2);
  int i;
  for (i = 0; i < nelms2; i++) {
    indata[i] = i;
  }

  float *outdata = (float *)malloc(sizeof(float) * nelms1);
    
  PSGridCopyin(g2, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g1, g2));
    
  PSGridCopyout(g1, outdata);

  dump(outdata, nelms1, stdout);
#if 0    
  for (i = 0; i < nelms; i++) {
    if (indata[i] != outdata[i]) {
      fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
              i, indata[i], outdata[i]);
    }
  }
#endif

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

