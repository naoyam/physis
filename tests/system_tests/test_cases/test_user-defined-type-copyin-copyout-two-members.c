/*
 * TEST: Run copyin and copyout on a user-defined type with two members
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32
#define ITER 10

struct Point {
  float x;
  float y;
};

DeclareGrid3D(Point, struct Point);

void check(struct Point *in, struct Point *out) {
  int x = 0;
  size_t nelms = N*N*N;  
  for (x = 0; x < nelms; ++x) {
    if (in[x].x != out[x].x) {
      fprintf(stderr, "Error: x mismatch at %d, in: %f, out: %f\n",
              x, in[x].x, out[x].x);
      exit(1);
    }
    if (in[x].y != out[x].y) {
      fprintf(stderr, "Error: y mismatch at %d, in: %f, out: %f\n",
              x, in[x].y, out[x].y);
      exit(1);
    }
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DPoint g1 = PSGrid3DPointNew(N, N, N);
  size_t nelms = N*N*N;
  struct Point *indata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i].x = i;
    indata[i].y = i+1;
  }
  struct Point *outdata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
    
  PSGridCopyin(g1, indata);
  PSGridCopyout(g1, outdata);

  check(indata, outdata);

  PSGridFree(g1);
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

