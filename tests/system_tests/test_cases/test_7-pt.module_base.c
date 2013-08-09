/*
 * TEST: 7-point stencil
 * DIM: 3
 * PRIORITY: 1 
 */

#include <stdio.h>
#include <stdlib.h>
//#include "physis/physis.h"

#define N 32

typedef void * PSGrid3DFloat;
extern int test_module_init(int, char *[]);
extern PSGrid3DFloat create_grid(int);
extern void copyin(PSGrid3DFloat, float*);
extern void copyout(PSGrid3DFloat, float*);
extern void run(PSGrid3DFloat, PSGrid3DFloat, int);

void dump(float *input) {
  int i;
  for (i = 0; i < N*N*N; ++i) {
    printf("%f\n", input[i]);
  }
}

int main(int argc, char *argv[]) {
  test_module_init(argc, argv);
  size_t nelms = N*N*N;
  
  float *indata = (float *)malloc(sizeof(float) * nelms);
  int i;
  for (i = 0; i < nelms; i++) {
    indata[i] = i;
  }
  float *outdata = (float *)malloc(sizeof(float) * nelms);

  PSGrid3DFloat g1 = create_grid(N);
  PSGrid3DFloat g2 = create_grid(N);
    
  copyin(g1, indata);
  copyin(g2, indata);
  run(g1, g2, N);
  copyout(g2, outdata);

  dump(outdata);  

  free(indata);
  free(outdata);
  return 0;
}

