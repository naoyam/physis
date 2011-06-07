// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_BUFFER_CUDA_H_
#define PHYSIS_RUNTIME_BUFFER_CUDA_H_

#include "runtime/buffer.h"
#include <cuda_runtime.h>

namespace physis {
namespace runtime {

class BufferCUDADev;

// Pinned buffer on *host* memory
class BufferCUDAHost: public Buffer {
 public:
  BufferCUDAHost(size_t elm_size);
  BufferCUDAHost(int num_dims, size_t elm_size);  
  virtual ~BufferCUDAHost();
  
  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size);
  virtual void Copyin(const void *buf, const IntArray &size) {
    Copyin(buf, IntArray((index_t)0), size);
  }
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IntArray &offset,
                      const IntArray &size);                       
 
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout(void *buf,  const IntArray &size) {
    Copyout(buf, IntArray((index_t)0), size);
  }
  virtual void Copyout(BufferHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void Copyout(BufferHost &buf,  const IntArray &size) {
    Copyout(buf, IntArray(), size);
  }
  
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &size) {
    MPIRecv(src, comm, IntArray((index_t)0), size);
  }
  
  virtual void MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);

  template <class T>
  std::ostream& print(std::ostream &os) const {
    StringJoin sj;
    T *p = (T*)buf_;
    int idx = 0;
    for (int d = 0; d < num_dims_; ++d) {
      StringJoin sjd;
      for (int i = 0; i < size_[d]; ++i) {
        sjd << p[idx];
        ++idx;
      }
      sj << "{" << sjd.str() << "}";
    }
    os << sj.str();  
    return os;
  }

 protected:
  virtual void *GetChunk(const IntArray &size);

  BufferHost *mpi_buf_;
 public:
  static void DeleteChunk(void *ptr);  
};

class BufferCUDAHostMapped: public Buffer {
 public:
  BufferCUDAHostMapped(size_t elm_size);
  BufferCUDAHostMapped(int num_dims, size_t elm_size);  
  virtual ~BufferCUDAHostMapped();
  
  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size);
  virtual void Copyin(const void *buf, const IntArray &size) {
    Copyin(buf, IntArray((index_t)0), size);
  }
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IntArray &offset,
                      const IntArray &size);                       
 
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout(void *buf,  const IntArray &size) {
    Copyout(buf, IntArray((index_t)0), size);
  }
  virtual void Copyout(BufferHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void Copyout(BufferHost &buf,  const IntArray &size) {
    Copyout(buf, IntArray(), size);
  }
  
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &size) {
    MPIRecv(src, comm, IntArray((index_t)0), size);
  }
  
  virtual void MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void Allocate(int num_dims, size_t elm_size, const IntArray &size);
 protected:
  virtual void *GetChunk(const IntArray &size);

  BufferHost *mpi_buf_;
  void *dev_ptr_;
  
 public:
  static void DeleteChunk(void *ptr);
  void *GetDevPointer() { return dev_ptr_;}
};

class BufferCUDADev: public Buffer {
 public:
  BufferCUDADev(size_t elm_size);
  BufferCUDADev(int num_dims, size_t elm_size);  
  virtual ~BufferCUDADev();
  
  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size);
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IntArray &offset,
                      const IntArray &size);                      
  virtual void Copyin(const BufferCUDAHost &buf, const IntArray &offset,
                      const IntArray &size);                      
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout(BufferHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void Copyout(BufferCUDAHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  cudaStream_t &strm() { return strm_; }

 protected:
  virtual void *GetChunk(const IntArray &size);
  BufferCUDAHost *pinned_buf_;
  cudaStream_t strm_;  
 public:
  static void DeleteChunk(void *ptr);
};

class BufferCUDADev3D: public Buffer {
 public:
  BufferCUDADev3D(int num_dims, size_t elm_size);  
  virtual ~BufferCUDADev3D();

  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size);
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IntArray &offset,
                      const IntArray &size);                      
  virtual void Copyin(const BufferCUDAHost &buf, const IntArray &offset,
                      const IntArray &size);                      
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout(BufferHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void Copyout(BufferCUDAHost &buf, const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout(BufferCUDAHostMapped &buf,
                       const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout_Opt(BufferCUDAHostMapped &buf,
                           const IntArray &offset,
                           const IntArray &size);                       
  
  
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  size_t GetPitch() const {
    return pp_.pitch;
  }
  cudaStream_t &strm() { return strm_; }
 protected:
  virtual cudaPitchedPtr GetChunk3D(const IntArray &size);
  virtual void Allocate(const IntArray &size);
  virtual void Allocate(int num_dims, size_t elm_size, const IntArray &size);  
  BufferCUDAHost *pinned_buf_;
  BufferCUDAHostMapped *mapped_buf_;
  cudaPitchedPtr pp_;
  cudaStream_t strm_;
 public:
  static void DeleteChunk(void *ptr);
};



} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_BUFFER_CUDA_H_ */
