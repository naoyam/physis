/*
 * TEST: Copyin and copyout grids of user-defined types with multi-dimensional arrays
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32
#define ITER 10

#define DIM1 2
#define DIM2 3

struct Point {
  float p[DIM1][DIM2];
};

DeclareGrid3D(Point, struct Point);

static void copy(const int x, const int y, const int z,
                 PSGrid3DPoint g1, PSGrid3DPoint g2) {
  int i, j;
  for (i = 0; i < DIM1; ++i) {
    for (j = 0; j < DIM2; ++j) {
      float v = PSGridGet(g1, x, y, z).p[i][j];
      PSGridEmitUtype(g2.p[i][j], v);
    }
  }
}

void check(struct Point *in, struct Point *out) {
  int i, j;
  int x = 0;
  size_t nelms = N*N*N;  
  for (x = 0; x < nelms; ++x) {
    for (i = 0; i < DIM1; ++i) {
      for (j = 0; j < DIM2; ++j) {
        if (in[x].p[i][j] != out[x].p[i][j]) {
          fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
                  x, in[x].p[i][j], out[x].p[i][j]);
          exit(1);
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DPoint g1 = PSGrid3DPointNew(N, N, N);
  PSGrid3DPoint g2 = PSGrid3DPointNew(N, N, N);  
  size_t nelms = N*N*N;
  struct Point *indata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  struct Point *outdata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  int i, j;
  int x = 0;
  float c = 1.0f;
  for (x = 0; x < nelms; ++x) {
    for (i = 0; i < DIM1; ++i) {
      for (j = 0; j < DIM2; ++j) {
        indata[x].p[i][j] = c++;
        outdata[x].p[i][j] = -1;
      }
    }
  }
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, outdata);
  PSDomain3D dom = PSDomain3DNew(0, N, 0, N, 0, N);
  PSStencilRun(PSStencilMap(copy, dom, g1, g2));
  PSGridCopyout(g2, outdata);
  
  check(indata, outdata);

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

