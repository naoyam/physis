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
  float x;
  float y;
  float z;
};

DeclareGrid3D(Point, struct Point);

void kernel1(const int x, const int y, const int z,
             PSGrid3DPoint g1, PSGrid3DPoint g2) {
  float p = (PSGridGet(g1, x, y, z).x +
             PSGridGet(g1, x+1, y, z).x +
             PSGridGet(g1, x-1, y, z).x) / 3.0f;
  float q = (PSGridGet(g1, x, y, z).y +
             PSGridGet(g1, x, y+1, z).y +
             PSGridGet(g1, x, y-1, z).y +
             PSGridGet(g1, x, y, z+1).y +
             PSGridGet(g1, x, y, z-1).y) / 5.0f;
  float r = (PSGridGet(g1, x, y, z).z +
             PSGridGet(g1, x, y, z+1).z +
             PSGridGet(g1, x, y, z-1).z) / 3.0f;
  struct Point v = {p, q, r};
  PSGridEmit(g2, v);
  //PSGridEmit2(g2.r, r);
  return;
}

#define halo_width (1)

void dump(struct Point *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f %f %f\n", input[i].x, input[i].y, input[i].z);
  }
}

int main(int argc, char *argv[]) {
  PSInit(&argc, &argv, 3, N, N, N);
  PSGrid3DPoint g1 = PSGrid3DPointNew(N, N, N);
  PSGrid3DPoint g2 = PSGrid3DPointNew(N, N, N);  

  PSDomain3D d = PSDomain3DNew(0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width,
                               0+halo_width, N-halo_width);
  size_t nelms = N*N*N;
  
  struct Point *indata = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i].x = i;
    indata[i].y = i+1;
    indata[i].z = i+2;
  }
  struct Point *outdata_ps = (struct Point *)malloc(
      sizeof(struct Point) * nelms);
    
  PSGridCopyin(g1, indata);
  PSGridCopyin(g2, indata);

  assert(ITER % 2 == 0);
  
  PSStencilRun(PSStencilMap(kernel1, d, g1, g2),
               PSStencilMap(kernel1, d, g2, g1),
               ITER/2);
    
  PSGridCopyout(g1, outdata_ps);

  dump(outdata_ps);

  PSGridFree(g1);
  PSGridFree(g2);  
  PSFinalize();
  free(indata);
  free(outdata_ps);
  return 0;
}

