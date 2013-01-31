#ifndef BENCHMARKS_DIFFUSION3D_DIFFUSION3D_OPENMP_TEMPORAL_BLOCKING_H_
#define BENCHMARKS_DIFFUSION3D_DIFFUSION3D_OPENMP_TEMPORAL_BLOCKING_H_

#include "diffusion3d_openmp.h"

namespace diffusion3d {

class Diffusion3DOpenMPTemporalBlocking: public Diffusion3DOpenMP {
 public:
  Diffusion3DOpenMPTemporalBlocking(int nx, int ny, int nz):
      Diffusion3DOpenMP(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("openmp_temporal_blocking");
  }
  virtual void RunKernel(int count);
};

}

#endif /* BENCHMARKS_DIFFUSION3D_DIFFUSION3D_OPENMP_H_ */
