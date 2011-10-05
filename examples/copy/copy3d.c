#include <stdio.h>
#include "physis/physis.h"
#include "physis/stopwatch.h"

#define IDX3(x, y, z) ((x) + (y) * N + (z) * N * N)
#define N (256)

void copy(const int x, const int y, const int z, PSGrid3DFloat g1,
          PSGrid3DFloat g2) {
  PSGridEmit(g2, PSGridGet(g1, x, y, z));
  return;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  float *indata = (float *)malloc(sizeof(float) * nelms);
  float *outdata = (float *)malloc(sizeof(float) * nelms);  
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
    
  PSGridCopyin(g1, indata);

  __PSStopwatch st1;
  __PSStopwatchQuery(&st1);
  PSStencilRun(PSStencilMap(copy, d, g1, g2),
               PSStencilMap(copy, d, g2, g1),
               100);
  float elapsed_time = __PSStopwatchStop(&st1);
    
  PSGridCopyout(g2, outdata);

  int j, k;
  for (k = 0; k < N; ++k) {
    for (j = 0; j < N; ++j) {
      for (i = 0; i < N; ++i) {
        if (indata[IDX3(i, j, k)] != outdata[IDX3(i, j, k)]) {
          fprintf(stderr, "ERROR: mismatch found at (%d, %d, %d)\n",
                  i, j, k);
          fprintf(stderr, "Copy failed.\n");
          exit(1);
        }
      }
    }
  }

  printf("Copy bandwidth: %.3f (GB/s)\n",
         sizeof(float) * nelms * 2 / (elapsed_time * 0.001)
         * 0.000000001);
  
  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}
