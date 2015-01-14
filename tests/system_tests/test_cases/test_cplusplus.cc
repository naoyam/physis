/*
 * TEST: C++ test
 * DIM: 3
 * PRIORITY: 10
 * TARGETS: ref cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#include "physis/physis.h"

#define N 8

class Foo {
 public:
  Foo(int x) {
    x_ = x;
  }
  int operator()() {
    return x_;
  }
 private:
    int x_;
};

void kernel(const int x, const int y, const int z, PSGrid3DFloat g,
            PSGrid3DFloat g2) {
  float v = PSGridGet(g, x, y, z);
  PSGridEmit(g2, v);
  return;
}

void physis_func(Foo &x) {
  PSGrid3DFloat g1 = PSGrid3DFloatNew(N, N, N);
  PSGrid3DFloat g2 = PSGrid3DFloatNew(N, N, N);
  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  int nelms = N*N*N;
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i * x();
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);
    
  PSGridCopyin(g1, indata);

  PSStencilRun(PSStencilMap(kernel, d, g1, g2));
    
  PSGridCopyout(g2, outdata);

  for (i = 0; i < nelms; i++) {
    if (indata[i] != outdata[i]) {
      fprintf(stderr, "Error: mismatch at %d, in: %f, out: %f\n",
              i, indata[i], outdata[i]);
      exit(1);
    }
  }

  PSGridFree(g1);
  PSGridFree(g2);
  PSFinalize();
  free(indata);
  free(outdata);
  return;
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  Foo f(2);
  physis_func(f);
  return 0;
}

