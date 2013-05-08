/*
 * TEST: Run stencil on a user-defined type
 * DIM: 3
 * PRIORITY: 2
 */

#include <stdio.h>
#include "physis/physis.h"

#define N 32
#define ITER 10

struct Point {
  float p;
  float q;
  float r;
};

DeclareGrid3D(Point, struct Point);

void kernel1(const int x, const int y, const int z,
             PSGrid3DPoint g) {
  float v = (PSGridGet(g, x, y, z).p +
             PSGridGet(g, x+1, y, z).p +
             PSGridGet(g, x-1, y, z).p +
             PSGridGet(g, x, y+1, z).p +
             PSGridGet(g, x, y-1, z).p +
             PSGridGet(g, x, y, z+1).p +
             PSGridGet(g, x, y, z-1).p +
             PSGridGet(g, x, y, z).q +
             PSGridGet(g, x+1, y, z).q +
             PSGridGet(g, x-1, y, z).q +
             PSGridGet(g, x, y+1, z).q +
             PSGridGet(g, x, y-1, z).q +
             PSGridGet(g, x, y, z+1).q +
             PSGridGet(g, x, y, z-1).q) / 14.0;
  PSGridEmitUtype(g.r, v);
  return;
}

void kernel2(const int x, const int y, const int z,
             PSGrid3DPoint g) {
  PSGridEmitUtype(g.p,
              (PSGridGet(g, x, y, z).p + 
               PSGridGet(g, x, y, z).r) * 0.5);
  PSGridEmitUtype(g.q,
              (PSGridGet(g, x, y, z).q + 
               PSGridGet(g, x, y, z).r) * 0.5);
  return;
}

#define halo_width (1)

void dump(struct Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f %f\n", input[i].p, input[i].q, input[i].r);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DPoint g = PSGrid3DPointNew(N, N, N);

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  struct Point *indata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i].p = i;
    indata[i].q = i+1;
    indata[i].r = 0;
  }
  struct Point *outdata_ps = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
    
  PSGridCopyin(g, indata);

  PSStencilRun(PSStencilMap(kernel1, d, g),
               PSStencilMap(kernel2, d, g),
               ITER);
    
  PSGridCopyout(g, outdata_ps);

  dump(outdata_ps);

  PSGridFree(g);
  PSFinalize();
  free(indata);
  free(outdata_ps);
  return 0;
}

