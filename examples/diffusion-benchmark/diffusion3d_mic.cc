#include "benchmarks/diffusion3d/diffusion3d_mic.h"

namespace diffusion3d {

void Diffusion3DMIC::InitializeBenchmark() {
  f1_ = (REAL*)_mm_malloc(sizeof(REAL) * nx_ * ny_ * nz_, 4096);
  assert(f1_);    
  f2_ = (REAL*)_mm_malloc(sizeof(REAL) * nx_ * ny_ * nz_, 4096);
  assert(f2_);
  Initialize(f1_, nx_, ny_, nz_,
             kx_, ky_, kz_, dx_, dy_, dz_,
             kappa_, 0.0);
}

void Diffusion3DMIC::RunKernel(int count) {
  int i;
#pragma offload target(mic) \
  inout(f1_:length(nx_*ny_*nz_) align(2*1024*1024))     \
  inout(f2_:length(nx_*ny_*nz_) align(2*1024*1024))
  {
    for (i = 0; i < count; ++i) {
      int y, z;
#pragma omp parallel for collapse(2) private(y, z)
      for (z = 0; z < nz_; z++) {
        for (y = 0; y < ny_; y++) {
          int x;
#pragma ivdep          
          for (x = 0; x < nx_; x++) {
            int c, w, e, n, s, b, t;
            c =  x + y * nx_ + z * nx_ * ny_;
            w = (x == 0)    ? c : c - 1;
            e = (x == nx_-1) ? c : c + 1;
            n = (y == 0)    ? c : c - nx_;
            s = (y == ny_-1) ? c : c + nx_;
            b = (z == 0)    ? c : c - nx_ * ny_;
            t = (z == nz_-1) ? c : c + nx_ * ny_;
            f2_[c] = cc_ * f1_[c] + cw_ * f1_[w] + ce_ * f1_[e]
                + cs_ * f1_[s] + cn_ * f1_[n] + cb_ * f1_[b] + ct_ * f1_[t];
          }
        }
      }
      REAL *t = f1_;
      f1_ = f2_;
      f2_ = t;
    }
  }
  return;
}

}
