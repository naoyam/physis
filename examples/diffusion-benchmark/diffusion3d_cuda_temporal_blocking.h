#ifndef BENCHMARKS_DIFFUSION3D_DIFFUSION3D_CUDA_TEMPORAL_BLOCKING_H_
#define BENCHMARKS_DIFFUSION3D_DIFFUSION3D_CUDA_TEMPORAL_BLOCKING_H_

#include "diffusion3d.h"
#include "baseline.h"
#include "diffusion3d_cuda.h"

#include <cuda_runtime.h>

namespace diffusion3d {

class Diffusion3DCUDATemporalBlocking: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDATemporalBlocking(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    block_x_ = 32;
    block_y_ = 16;
  }
  virtual std::string GetName() const {
    return std::string("cuda_temporal_blocking");
  }
  //virtual void InitializeBenchmark();
  virtual void RunKernel(int count);
};

}

#endif /* BENCHMARKS_DIFFUSION3D_DIFFUSION3D_CUDA_TEMPORAL_BLOCKING_H_ */
