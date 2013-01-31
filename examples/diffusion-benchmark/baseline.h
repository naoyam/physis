#ifndef BENCHMARKS_DIFFUSION3D_BASELINE_H_
#define BENCHMARKS_DIFFUSION3D_BASELINE_H_

#include "diffusion3d.h"

namespace diffusion3d {

class Baseline: public Diffusion3D {
 protected:
  REAL *f1_, *f2_;
 public:
  Baseline(int nx, int ny, int nz):
      Diffusion3D(nx, ny, nz), f1_(NULL), f2_(NULL) {}
  virtual std::string GetName() const {
    return std::string("baseline");
  }
  virtual void InitializeBenchmark();
  virtual void FinalizeBenchmark();
  virtual void RunKernel(int count);
  virtual REAL GetAccuracy(int count);  
  virtual void Dump() const;
};

}

#endif /* DIFFUSION3D_DIFFUSION3D_H_ */
