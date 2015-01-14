/*
 * TEST: Run 7-pt periodic stencil on a user-defined type
 * DIM: 3
 * PRIORITY: 1
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32

struct Complex {
  float r;
  float i;
};

DeclareGrid3D(Complex, struct Complex);

void kernel1(const int x, const int y, const int z,
             PSGrid3DComplex g1, PSGrid3DComplex g2) {
  struct Complex t = PSGridGetPeriodic(g1, x, y, z);
  struct Complex t1 = PSGridGetPeriodic(g1, x+1, y, z);
  struct Complex t2 = PSGridGetPeriodic(g1, x-1, y, z);    
  struct Complex t3 = PSGridGetPeriodic(g1, x, y+1, z);
  struct Complex t4 = PSGridGetPeriodic(g1, x, y-1, z);    
  struct Complex t5 = PSGridGetPeriodic(g1, x, y, z+1);
  struct Complex t6 = PSGridGetPeriodic(g1, x, y, z-1);    

  float r = t.r + t1.r + t2.r + t3.r + t4.r + t5.r + t6.r;
  float i = t.i + t1.i + t2.i + t3.i + t4.i + t5.i + t6.i;  
  struct Complex v = {r, i};
  PSGridEmit(g2, v);
  return;
}

#define halo_width (1)

void dump(struct Complex *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f\n", input[i].r, input[i].i);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DComplex g1 = PSGrid3DComplexNew(N, N, N);
  PSGrid3DComplex g2 = PSGrid3DComplexNew(N, N, N);  

  PSDomain3D d = PSDomain3DNew(0, N, 0, N, 0, N);
  size_t nelms = N*N*N;
  
  struct Complex *indata = (struct Complex *)malloc(
      sizeof(struct Complex) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i].r = i;
    indata[i].i = i+1;
  }
  struct Complex *outdata = (struct Complex *)malloc(
      sizeof(struct Complex) * nelms);
    
  PSGridCopyin(g1, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g1, g2));
    
  PSGridCopyout(g2, outdata);

  dump(outdata);

  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();
  free(indata);
  free(outdata);
  return 0;
}

