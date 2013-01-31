#ifndef BENCHMARKS_DIFFUSION3D_DIFFUSION3D_PHYSIS_H_
#define BENCHMARKS_DIFFUSION3D_DIFFUSION3D_PHYSIS_H_

#include "diffusion3d.h"
#include "baseline.h"

extern "C" {
  extern void initialize_physis(int argc, char **argv,
                                int nx, int ny, int nz);
  extern void initialize_benchmark_physis(int nx, int ny, int nz);
  extern void finalize_benchmark_physis();
  extern void run_kernel_physis(int count, REAL *f1_host,
                                int nx, int ny, int nz,
                                REAL ce, REAL cw, REAL cn, REAL cs,
                                REAL ct, REAL cb, REAL cc);
  
}

namespace diffusion3d {

class Diffusion3DPhysis: public Baseline {
 public:
  Diffusion3DPhysis(int nx, int ny, int nz,
                    int argc, char **argv):
      Baseline(nx, ny, nz) {
    initialize_physis(argc, argv, nx, ny, nz);
  }
  virtual std::string GetName() const {
    return std::string("physis");
  }
  virtual void InitializeBenchmark() {
    Baseline::InitializeBenchmark();
    initialize_benchmark_physis(nx_, ny_, nz_);
  }
  virtual void FinalizeBenchmark() {
    finalize_benchmark_physis();
  }
  virtual void RunKernel(int count) {
    run_kernel_physis(count, f1_, nx_, ny_, nz_,
                      ce_, cw_, cn_, cs_, ct_, cb_, cc_);
  }
  
};

}

#endif /* BENCHMARKS_DIFFUSION3D_DIFFUSION3D_PHYSIS_H_ */

