#include <stdio.h>
#include "physis/physis.h"

#define N 4

#define IDX3(x, y, z) ((x) + (y) * N + (z) * N * N)

void kernel1(const int x, const int y, const int z, PSGrid3DFloat g,
             PSGrid3DFloat g2) {
  float v = PSGridGet(g, x+1, y+1, z+1);
  PSGridEmit(g2, v);
  return;
}

void dump(float *buf, size_t len, FILE *out) {
  int i;
  for (i = 0; i < len; ++i) {
    fprintf(out, "%f\n", buf[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g = PSGrid3DFloatNew(N, N, N, PS_GRID_CIRCULAR);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N, PS_GRID_CIRCULAR);  
  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i, j, k;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g, g2));
    
  PSGridCopyout(g2, outdata);

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      for (k = 0; k < N; ++k) {
        if (outdata[IDX3(i, j, k)] !=
            indata[IDX3((i+1)%N, (j+1)%N, (k+1)%N)]) {
          fprintf(stderr,
                  "Error: mismatch at %d,%d,%d, in: %f, out: %f\n",
                  i, j, k, indata[IDX3(i, j, k)],
                  outdata[IDX3((i+1)%N, (j+1)%N, (k+1)%N)]);
        }
      }
    }
  }

  PSGridFree(g);
  PSGridFree(g2);
  PSFinalize();

  dump(outdata, N*N*N, stdout);
  
  free(indata);
  free(outdata);

  
  return 0;
}

