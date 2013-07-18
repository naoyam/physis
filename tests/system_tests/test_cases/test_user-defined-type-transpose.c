/*
 * TEST: Transpose each 2D array in a 3D grid
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32
#define ITER 10

#define DIM1 2
#define DIM2 3

struct Point1 {
  float p[DIM1][DIM2];
};

struct Point2 {
  float p[DIM2][DIM1];
};

DeclareGrid3D(Point1, struct Point1);
DeclareGrid3D(Point2, struct Point2);

void transpose(const int x, const int y, const int z,
               PSGrid3DPoint1 g1, PSGrid3DPoint2 g2) {
  int i, j;
  for (i = 0; i < DIM1; ++i) {
    for (j = 0; j < DIM2; ++j) {
      PSGridEmitUtype(g2.p[j][i], PSGridGet(g1, x, y, z).p[i][j]);
    }
  }
  return;
}

void check(struct Point2 *p) {
  int i, j;
  int x = 0;
  float c = 1.0f;
  size_t nelms = N*N*N;  
  for (x = 0; x < nelms; ++x) {
    for (i = 0; i < DIM1; ++i) {
      for (j = 0; j < DIM2; ++j) {
        if (p[x].p[j][i] != c) {
          fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
                  x, c, p[x].p[j][i]);
          exit(1);
        }
        ++c;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DPoint1 g1 = PSGrid3DPoint1New(N, N, N);
  PSGrid3DPoint2 g2 = PSGrid3DPoint2New(N, N, N);  

  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  struct Point1 *indata = (struct Point1 *)malloc(
      sizeof(struct Point1) * nelms);
  struct Point2 *outdata = (struct Point2 *)malloc(
      sizeof(struct Point2) * nelms);
  int i, j;
  int x = 0;
  float c = 1.0f;
  for (x = 0; x < nelms; ++x) {
    for (i = 0; i < DIM1; ++i) {
      for (j = 0; j < DIM2; ++j) {
        indata[x].p[i][j] = c++;
      }
    }
  }
    
  PSGridCopyin(g1, indata);

  memset(indata, 0, nelms * sizeof(struct Point1));
  PSGridCopyin(g2, indata);

  PSStencilRun(PSStencilMap(transpose, d, g1, g2));
    
  PSGridCopyout(g2, outdata);

  check(outdata);

  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

