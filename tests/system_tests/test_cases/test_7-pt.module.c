/*
 * TEST: Module test
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include "physis/physis.h"

static void kernel(const int x, const int y, const int z, PSGrid3DFloat g1,
                   PSGrid3DFloat g2) {
  float v = PSGridGet(g1, x, y, z) +
      PSGridGet(g1, x+1, y, z) + PSGridGet(g1, x-1, y, z) +
      PSGridGet(g1, x, y+1, z) + PSGridGet(g1, x, y-1, z) +
      PSGridGet(g1, x, y, z-1) + PSGridGet(g1, x, y, z+1);
  PSGridEmit(g2, v);
  return;
}

#define halo_width (1)

#ifdef __cplusplus
extern "C" {
#endif

int run(PSGrid3DFloat g1, PSGrid3DFloat g2, int n) {
  PSDomain3D d = PSDomain3DNew(0+halo_width, n-halo_width,
                               0+halo_width, n-halo_width,
                               0+halo_width, n-halo_width);
  PSStencilRun(PSStencilMap(kernel, d, g1, g2));
  return 0;
}

PSGrid3DFloat create_grid(int n) {
  PSGrid3DFloat g = PSGrid3DFloatNew(n, n, n);
  return g;
}

void copyin(PSGrid3DFloat g, float *d) {
  PSGridCopyin(g, d);
}

void copyout(PSGrid3DFloat g, float *d) {
  PSGridCopyout(g, d);
}

int test_module_init(int argc, char *argv[], int n) {
  PSInit(&argc, &argv, 3, n, n, n);
  return 0;
}

#ifdef __cplusplus
}
#endif

