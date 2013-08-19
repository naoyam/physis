#ifndef BENCHMARKS_DIFFUSION3D_DIFFUSION3D_CUDA_H_
#define BENCHMARKS_DIFFUSION3D_DIFFUSION3D_CUDA_H_

#include "diffusion3d.h"
#include "baseline.h"

#include <cuda_runtime.h>

namespace diffusion3d {

class Diffusion3DCUDA: public Baseline {
 public:
  Diffusion3DCUDA(int nx, int ny, int nz):
      Baseline(nx, ny, nz), f1_d_(NULL), f2_d_(NULL),
      block_x_(64), block_y_(4), block_z_(1)
  {
    //assert(nx_ % block_x_ == 0);
    //assert(ny_ % block_y_ == 0);
    //assert(nz_ % block_z_ == 0);
  }
  virtual std::string GetName() const {
    return std::string("cuda");
  }
  virtual void InitializeBenchmark();
  virtual void RunKernel(int count);
  virtual void FinalizeBenchmark();
  virtual void DisplayResult(int count, float time);
 protected:
  REAL *f1_d_, *f2_d_;
  int block_x_, block_y_, block_z_;
  cudaEvent_t ev1_, ev2_;

};

class Diffusion3DCUDAOpt1: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAOpt1(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("cuda_opt1");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAOpt2: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAOpt2(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    block_x_ = 128;    
    block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_opt2");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAXY: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAXY(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    block_x_ = 32;    
    block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_xy");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class Diffusion3DCUDAShared: public Diffusion3DCUDA {
 public:
  Diffusion3DCUDAShared(int nx, int ny, int nz):
      Diffusion3DCUDA(nx, ny, nz) {
    block_x_ = 128;
    block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};


}

#endif /* BENCHMARKS_DIFFUSION3D_DIFFUSION3D_CUDA_H_ */
