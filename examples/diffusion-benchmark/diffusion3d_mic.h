#ifndef BENCHMARKS_DIFFUSION3D_DIFFUSION3D_MIC_H_
#define BENCHMARKS_DIFFUSION3D_DIFFUSION3D_MIC_H_

#include "diffusion3d.h"
#include "baseline.h"

namespace diffusion3d {

class Diffusion3DMIC: public Baseline {
 public:
  Diffusion3DMIC(int nx, int ny, int nz):
      Baseline(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("mic");
  }
  virtual void InitializeBenchmark();
  virtual void RunKernel(int count);
};

}

#endif /* BENCHMARKS_DIFFUSION3D_DIFFUSION3D_OPENMP_H_ */
