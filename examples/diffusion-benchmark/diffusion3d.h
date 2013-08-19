#ifndef BENCHMARKS_DIFFUSION3D_DIFFUSION3D_H_
#define BENCHMARKS_DIFFUSION3D_DIFFUSION3D_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <string>

#define REAL float
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#include "stopwatch.h"

#define GET_INITIAL_VAL(x, y, z, nx, ny, nz,            \
                        kx, ky, kz,                     \
                        dx, dy, dz, kappa, time,        \
                        ax, ay, az)                     \
  do {                                                  \
    REAL i = dx*((REAL)((x) + 0.5));                    \
    REAL j = dy*((REAL)((y) + 0.5));                    \
    REAL k = dz*((REAL)((z) + 0.5));                    \
    return (REAL)0.125 *(1.0 - (ax)*cos((kx)*(i)))      \
        * (1.0 - (ay)*cos((ky)*(j)))                    \
        * (1.0 - (az)*cos((k)z*(k)));                   \
  } while (0)

#define OFFSET3D(i, j, k, nx, ny) \
  ((i) + (j) * (nx) + (k) * (nx) * (ny))

namespace diffusion3d {

void Initialize(REAL *buff, const int nx, const int ny, const int nz,
                const REAL kx, const REAL ky, const REAL kz,
                const REAL dx, const REAL dy, const REAL dz,
                const REAL kappa, const REAL time);

class Diffusion3D {
 protected:
  int nx_;
  int ny_;
  int nz_;
  REAL kappa_;
  REAL dx_, dy_, dz_;
  REAL kx_, ky_, kz_;
  REAL dt_;
  REAL ce_, cw_, cn_, cs_, ct_, cb_, cc_;
 public:
  Diffusion3D(int nx, int ny, int nz):
      nx_(nx), ny_(ny), nz_(nz), kappa_(0.1) {
    REAL l = 1.0;
    dx_ = l / nx;
    dy_ = l / ny;
    dz_ = l / nz;
    kx_ = ky_ = kz_ = 2.0 * M_PI;
    dt_ = 0.1 * dx_ * dx_ / kappa_;
    ce_ = cw_ = kappa_*dt_/(dx_*dx_);
    cn_ = cs_ = kappa_*dt_/(dy_*dy_);
    ct_ = cb_ = kappa_*dt_/(dz_*dz_);
    cc_ = 1.0 - (ce_ + cw_ + cn_ + cs_ + ct_ + cb_);
  }
  virtual ~Diffusion3D() {}
  virtual std::string GetName() const = 0;
  void RunBenchmark(int count, bool dump) {
    std::cout << "Initializing benchmark input...\n";
    InitializeBenchmark();
    std::cout << "Running diffusion3d/" << GetName() << "\n";
    std::cout << "Iteration count: " << count << "\n";
    std::cout << "Grid size: " << nx_ << "x" << ny_
              << "x" << nz_ << "\n";
    Stopwatch st;
    StopwatchStart(&st);
    RunKernel(count);
    float elapsed_time = StopwatchStop(&st);
    std::cout << "Benchmarking finished.\n";
    DisplayResult(count, elapsed_time);
    if (dump) Dump();
    FinalizeBenchmark();
  }

 protected:
  std::string GetDumpPath() const {
    return std::string("diffusion3d_result.")
        + GetName() + std::string(".out");
  }
  virtual void InitializeBenchmark() = 0;  
  virtual void RunKernel(int count) = 0;
  virtual void Dump() const = 0;
  virtual REAL GetAccuracy(int count) = 0;
  virtual void FinalizeBenchmark() = 0;    
  
  float GetThroughput(int count, float time) {
    return (nx_ * ny_ * nz_) * sizeof(REAL) * 2.0 * ((float)count)
        / time * 1.0e-09;    
  }
  float GetGFLOPS(int count, float time) {
    float f = (nx_*ny_*nz_)*13.0*(float)(count)/time * 1.0e-09;
    return f;
  }
  virtual void DisplayResult(int count, float time) {
    printf("Elapsed time : %.3f (s)\n", time);
    printf("FLOPS        : %.3f (GFLOPS)\n",
           GetGFLOPS(count, time));
    printf("Throughput   : %.3f (GB/s)\n",
           GetThroughput(count ,time));
    printf("Accuracy     : %e\n", GetAccuracy(count));
  }
  REAL *GetCorrectAnswer(int count) const {
    REAL *f = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f);
    Initialize(f, nx_, ny_, nz_,
               kx_, ky_, kz_, dx_, dy_, dz_,
               kappa_, count * dt_);
    return f;
  }
};

}

#endif /* DIFFUSION3D_DIFFUSION3D_H_ */
