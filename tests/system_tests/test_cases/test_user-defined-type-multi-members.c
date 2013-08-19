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
  float q;
  float r;
};

DeclareGrid3D(Point, struct Point);

static void kernel1(const int x, const int y, const int z,
             PSGrid3DPoint g) {
  float v1 = PSGridGet(g, x, y, z).p[0] +
      PSGridGet(g, x-1, y, z).p[0] +
      PSGridGet(g, x+1, y, z).p[0] +
      PSGridGet(g, x, y-1, z).p[0] +
      PSGridGet(g, x, y+1, z).p[0] +
      PSGridGet(g, x, y, z-1).p[0] +
      PSGridGet(g, x, y, z+1).p[0];
  float v2 = PSGridGet(g, x, y, z).p[1] +
      PSGridGet(g, x-1, y, z).p[1] +
      PSGridGet(g, x+1, y, z).p[1] +
      PSGridGet(g, x, y-1, z).p[1] +
      PSGridGet(g, x, y+1, z).p[1] +
      PSGridGet(g, x, y, z-1).p[1] +
      PSGridGet(g, x, y, z+1).p[1];
  float v3 = PSGridGet(g, x, y, z).q +
      PSGridGet(g, x-1, y, z).q +
      PSGridGet(g, x+1, y, z).q +
      PSGridGet(g, x, y-1, z).q +
      PSGridGet(g, x, y+1, z).q +
      PSGridGet(g, x, y, z-1).q +
      PSGridGet(g, x, y, z+1).q;
  PSGridEmitUtype(g.r, v1+v2+v3);
  return;
}

void dump(struct Point *output) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", output[i].r);
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
    indata[i].p[1] = i+1;
    indata[i].q = i+2;
    indata[i].r = 0;
    outdata[i].p[0] = 0;
    outdata[i].p[1] = 0;
    outdata[i].q = 0;
    outdata[i].r = 0;
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

