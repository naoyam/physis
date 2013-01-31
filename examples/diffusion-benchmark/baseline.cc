#include "baseline.h"

namespace diffusion3d {

void Baseline::InitializeBenchmark() {
  f1_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
  assert(f1_);    
  f2_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
  assert(f2_);
  Initialize(f1_, nx_, ny_, nz_,
             kx_, ky_, kz_, dx_, dy_, dz_,
             kappa_, 0.0);
}

void Baseline::FinalizeBenchmark() {
  assert(f1_);
  free(f1_);
  assert(f2_);
  free(f2_);
}

void Baseline::RunKernel(int count) {
  int i;
  for (i = 0; i < count; ++i) {
    int z;
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

REAL Baseline::GetAccuracy(int count) {
  REAL *ref = GetCorrectAnswer(count);
  REAL err = 0.0;
  long len = nx_*ny_*nz_;
  for (long i = 0; i < len; i++) {
    REAL diff = ref[i] - f1_[i];
    err +=  diff * diff;
  }
  return (REAL)sqrt(err/len);
}

void Baseline::Dump() const {
  FILE *out = fopen(GetDumpPath().c_str(), "w");
  assert(out);
  long nitems = nx_ * ny_ * nz_;
  for (long i = 0; i < nitems; ++i) {
    fprintf(out, "%f\n", f1_[i]);
  }
  fclose(out);
}


}
