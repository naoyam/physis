/*
 * TEST: Run stencil on a user-defined type
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

struct Point {
  float p[2];
};

DeclareGrid3D(Point, struct Point);

void kernel1(const int x, const int y, const int z,
             PSGrid3DPoint g) {
  float v = PSGridGet(g, x, y, z).p[0] +
      PSGridGet(g, x-1, y, z).p[0] +
      PSGridGet(g, x+1, y, z).p[0] +
      PSGridGet(g, x+1, y-1, z).p[0] +
      PSGridGet(g, x+1, y+1, z).p[0] +
      PSGridGet(g, x+1, y, z-1).p[0] +
      PSGridGet(g, x+1, y, z+1).p[0];
  PSGridEmitUtype(g.p[1], v);
  return;
}

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DPoint g = PSGrid3DPointNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(1, N-1, 1, N-1, 1, N-1);
  size_t nelms = N*N*N;
  
  struct Point *indata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  struct Point *outdata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i].p[0] = i;
    indata[i].p[1] = i;
    outdata[i].p[0] = 0;
    outdata[i].p[1] = 0;
  }
    
  PSGridCopyin(g, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g));
    
  PSGridCopyout(g, outdata);

  dump(outdata);

  PSGridFree(g);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

