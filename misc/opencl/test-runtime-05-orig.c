#include "physis/physis.h"
#include <math.h>

#define SIZEX 64
#define SIZEY 64
#define SIZEZ 64
#define NDIM  3

void kernel1(const long int x, const long int y, const long int z, PSGrid3DFloat g){
  float v = PSGridGet(g, x, y, z) * 2;
  PSGridEmit(g, v);
}

float plus10(float v){
  return v + 10.0;
}

void kernel2(const long int x, const long int y, const long int z, PSGrid3DFloat g){
  float v = PSGridGet(g, x, y, z);
  v = plus10(v);
  PSGridEmit(g, v);
}

void init (float *buff, const int dx, const int dy, const int dz){
    int jx, jy, jz;
    for (jz = 0; jz < dz; jz++) {
      for (jy = 0; jy < dy; jy++) {
        for (jx = 0; jx < dx; jx++) {
          int j = ((((jz * dx) * dy) + (jy * dx)) + jx);
          buff[j] = jx * jy * jz + 10;
        }
      }
    }
}

int main(int argc, char **argv){
  PSInit(&argc, &argv, NDIM, SIZEX, SIZEY, SIZEZ);

  PSGrid3DFloat g1 = PSGrid3DFloatNew(SIZEX, SIZEY, SIZEZ);
  PSDomain3D d1 = PSDomain3DNew(0L, SIZEX, 0L, SIZEY, 0L, SIZEZ);

  int nx = SIZEX;
  int ny = SIZEY;
  int nz = SIZEZ;

  float *buff1 = (float *)(malloc((sizeof(float)) * nx * ny * nz));
  float *buff2 = (float *)(malloc((sizeof(float)) * nx * ny * nz));
  init(buff1, nx, ny, nz);

  PSGridCopyin(g1, buff1);
  PSGridCopyout(g1, buff2);

  size_t nelms = nx * ny * nz;
  unsigned int i;
  for (i = 0; i < nelms; i++) {
    if (buff1[i] == buff2[i]) continue;
    fprintf(stderr, "Error: buff 1 and 2 differ at %i: %10.3f and %10.3f\n", i, buff1[i], buff2[i]);
  }

  PSGridCopyin(g1, buff1);
  PSStencilRun(PSStencilMap(kernel1, d1, g1));
  PSGridCopyout(g1, buff2);
  for (i = 0; i < nelms; i++) {
    if (buff1[i] * 2 == buff2[i]) {
      continue;
    } else {
    fprintf(stderr, "Error: buff 1 and 2 differ at %i: %10.3f and %10.3f\n", i, buff1[i], buff2[i]);
    }
  }

  PSGridCopyin(g1, buff1);
  PSStencilRun(PSStencilMap(kernel2, d1, g1));
  PSGridCopyout(g1, buff2);
  for (i = 0; i < nelms; i++) {
    if (buff1[i] + 10 == buff2[i]) {
      continue;
    } else {
    fprintf(stderr, "Error: buff 1 and 2 differ at %i: %10.3f and %10.3f\n", i, buff1[i], buff2[i]);
    }
  }

  free(buff1);
  free(buff2);
  PSGridFree(g1);
  PSFinalize();
}
