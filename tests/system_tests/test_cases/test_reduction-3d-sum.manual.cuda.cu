#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define N 16
#define REAL float

int main(int argc, char *argv[]) {
  REAL *g1, *g1d;
  size_t nelms = N*N*N;
  g1 = (REAL *)malloc(sizeof(REAL) * nelms);
  cudaMalloc((void**)&g1d, sizeof(REAL) * nelms);

  int i;
  for (i = 0; i < (int)nelms; i++) {
    g1[i] = i;
  }
    
  cudaMemcpy(g1d, g1, sizeof(REAL) * nelms, cudaMemcpyHostToDevice);
  
  thrust::device_ptr<REAL> dev_ptr((REAL*)g1d);
  REAL v = thrust::reduce(dev_ptr, dev_ptr + nelms,
                          0.0f, thrust::plus<REAL>());

  printf("%f\n", v);
  
  cudaDeviceReset();
  return 0;
}

