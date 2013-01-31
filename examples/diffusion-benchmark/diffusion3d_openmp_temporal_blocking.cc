#include "benchmarks/diffusion3d/diffusion3d_openmp_temporal_blocking.h"
#include <omp.h>

#define BLOCKING_FACTOR (2)

namespace diffusion3d {

void Diffusion3DOpenMPTemporalBlocking::RunKernel(int count) {
  int i;
  for (i = 0; i < count; i += BLOCKING_FACTOR) {

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      int nzp = nz_ / nthreads;
      assert(nz_ % nthreads == 0);
      assert((nz_ / nthreads) %  BLOCKING_FACTOR == 0);
      float *f2_work = (float*)malloc(sizeof(float)*nx_*ny_*(nzp+2));
      int z;
      for (z = nzp * tid - 1; z < nzp * (tid + 1) + 1; ++z) {
        if (z < 0 || z >= nz_) continue;
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
            int c_work = x + y * nx_ + (z - nzp * tid + 1) * nx_ * ny_;
            f2_work[c_work] = cc_ * f1_[c] + cw_ * f1_[w] + ce_ * f1_[e]
                + cs_ * f1_[s] + cn_ * f1_[n] + cb_ * f1_[b] + ct_ * f1_[t];
          }
        }
      }

      for (z = nzp * tid; z < nzp * (tid + 1); ++z) {
        int y;
        for (y = 0; y < ny_; y++) {
          int x;
          for (x = 0; x < nx_; x++) {
            int c, w, e, n, s, b, t, c_work;
            c_work =  x + y * nx_ + (z - nzp * tid + 1) * nx_ * ny_;            
            c =  x + y * nx_ + z * nx_ * ny_;
            w = (x == 0)    ? c_work : c_work - 1;
            e = (x == nx_-1) ? c_work : c_work + 1;
            n = (y == 0)    ? c_work : c_work - nx_;
            s = (y == ny_-1) ? c_work : c_work + nx_;
            b = (z == 0)    ? c_work : c_work - nx_ * ny_;
            t = (z == nz_-1) ? c_work : c_work + nx_ * ny_;
            f2_[c] = cc_ * f2_work[c_work] + cw_ * f2_work[w] + ce_ * f2_work[e]
                + cs_ * f2_work[s] + cn_ * f2_work[n] + cb_ * f2_work[b] + ct_ * f2_work[t];
          }
        }
      }
      free(f2_work);
    }
    REAL *t = f1_;
    f1_ = f2_;
    f2_ = t;
  }
  return;
}

}
