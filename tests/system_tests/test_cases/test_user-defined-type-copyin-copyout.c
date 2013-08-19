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

struct Point1 {
  float p[DIM1][DIM2];
};

DeclareGrid3D(Point1, struct Point1);

void check(struct Point1 *in, struct Point1 *out) {
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
  PSGrid3DPoint1 g1 = PSGrid3DPoint1New(N, N, N);
  size_t nelms = N*N*N;
  struct Point1 *indata = (struct Point1 *)malloc(
      sizeof(struct Point1) * nelms);
  struct Point1 *outdata = (struct Point1 *)malloc(
      sizeof(struct Point1) * nelms);
  int i, j;
  int x = 0;
  float c = 1.0f;
  for (x = 0; x < nelms; ++x) {
    for (i = 0; i < DIM1; ++i) {
      for (j = 0; j < DIM2; ++j) {
        indata[x].p[i][j] = c++;
        outdata[x].p[i][j] = 0;
      }
    }
  }
    
  PSGridCopyin(g1, indata);
  PSGridCopyout(g1, outdata);

  check(indata, outdata);

  PSGridFree(g1);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

