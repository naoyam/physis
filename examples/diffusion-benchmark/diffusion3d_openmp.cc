#include "benchmarks/diffusion3d/diffusion3d_openmp.h"

namespace diffusion3d {

void Diffusion3DOpenMP::InitializeOMP(REAL *buff, const int nx, const int ny, const int nz,
                                      const REAL kx, const REAL ky, const REAL kz,
                                      const REAL dx, const REAL dy, const REAL dz,
                                      const REAL kappa, const REAL time) {
  REAL ax = exp(-kappa*time*(kx*kx));
  REAL ay = exp(-kappa*time*(ky*ky));
  REAL az = exp(-kappa*time*(kz*kz));
  int jz;
#pragma omp parallel for    
  for (jz = 0; jz < nz; jz++) {
    int jy;
    for (jy = 0; jy < ny; jy++) {
      int jx;
      for (jx = 0; jx < nx; jx++) {
        int j = jz*nx*ny + jy*nx + jx;
        REAL x = dx*((REAL)(jx + 0.5));
        REAL y = dy*((REAL)(jy + 0.5));
        REAL z = dz*((REAL)(jz + 0.5));
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx*x))
          *(1.0 - ay*cos(ky*y))
          *(1.0 - az*cos(kz*z));
        buff[j] = f0;
      }
    }
  }
}

void Diffusion3DOpenMP::InitializeBenchmark() {
  f1_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
  assert(f1_);    
  f2_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
  assert(f2_);
  InitializeOMP(f1_, nx_, ny_, nz_,
                kx_, ky_, kz_, dx_, dy_, dz_,
                kappa_, 0.0);
}

void Diffusion3DOpenMP::RunKernel(int count) {
  int i;
  for (i = 0; i < count; ++i) {
    int z;
#pragma omp parallel for        
    for (z = 0; z < nz_; z++) {
      int y;
      for (y = 0; y < ny_; y++) {
        int x;
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
  return;
}

}
