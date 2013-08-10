/*
 * TEST: Run stencil on a user-defined type with a multi-dimensional array
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

struct Point {
  float p[3][2];
};

DeclareGrid3D(Point, struct Point);

static void kernel1(const int x, const int y, const int z,
             PSGrid3DPoint g) {
  float v1 = PSGridGet(g, x, y, z).p[0][0] +
      PSGridGet(g, x-1, y, z).p[0][0] +
      PSGridGet(g, x+1, y, z).p[0][0] +
      PSGridGet(g, x, y-1, z).p[0][0] +
      PSGridGet(g, x, y+1, z).p[0][0] +
      PSGridGet(g, x, y, z-1).p[0][0] +
      PSGridGet(g, x, y, z+1).p[0][0];
  float v2 = PSGridGet(g, x, y, z).p[2][1] +
      PSGridGet(g, x-1, y, z).p[2][1] +
      PSGridGet(g, x+1, y, z).p[2][1] +
      PSGridGet(g, x, y-1, z).p[2][1] +
      PSGridGet(g, x, y+1, z).p[2][1] +
      PSGridGet(g, x, y, z-1).p[2][1] +
      PSGridGet(g, x, y, z+1).p[2][1];
  PSGridEmitUtype(g.p[1][0], v1+v2);
  return;
}

void dump(struct Point *output) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", output[i].p[1][0]);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DPoint g = PSGrid3DPointNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(1, N-1, 1, N-1, 1, N-1);
  int nelms = N*N*N;
  
  struct Point *indata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  struct Point *outdata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    int j;
    for (j = 0; j < 3; ++j) {
      int k;
      for (k = 0; k < 2; ++k) {
        indata[i].p[j][k] = i+j+k;
        outdata[i].p[j][k] = 0;
      }
    }
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

