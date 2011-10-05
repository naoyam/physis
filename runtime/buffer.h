// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_BUFFER_H_
#define PHYSIS_RUNTIME_BUFFER_H_

#include "runtime/runtime_common.h"
#include "runtime/mpi_wrapper.h"


namespace physis {
namespace runtime {

class Buffer {
 protected:
  Buffer(size_t elm_size);  
 public:
  Buffer(int num_dims, size_t elm_size);  
  virtual ~Buffer();
  virtual void Allocate(int num_dims, size_t elm_size,
                        const IntArray &size);
  void Allocate(const IntArray &size);  
  void *&Get() { return buf_; }
  const void *Get() const { return buf_; }
  virtual const IntArray &size() const { return size_; }
  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size) = 0;
  void Copyin(const void *buf, size_t size);
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size) = 0;
  void *Copyout();
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size) = 0;
  void MPIRecv(int src, MPI_Comm comm, const IntArray &size) {
    MPIRecv(src, comm, IntArray((index_t)0), size);
  }
  virtual void MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size) = 0;
  void MPISend(int dst, MPI_Comm comm, const IntArray &size) {
    MPISend(dst, comm, IntArray((index_t)0), size);
  }
  virtual void Delete();
  virtual void EnsureCapacity(size_t size) {
    EnsureCapacity(elm_size_, size);
  }
  virtual void EnsureCapacity(const IntArray &size) {
    EnsureCapacity(num_dims_, elm_size_, size);
  }
  virtual void EnsureCapacity(size_t elm_size, size_t size);
  virtual void EnsureCapacity(int num_dims, size_t elm_size,
                              const IntArray &size);
  int num_dims() const { return num_dims_; }
  virtual size_t GetLinearSize(const IntArray &s) {
    return s.accumulate(num_dims_) * elm_size_; }
  virtual size_t GetLinearSize() { return GetLinearSize(size_); }

 protected:
  virtual void Shrink(size_t size);
  virtual void Shrink(const IntArray &size);
  virtual void *GetChunk(const IntArray &size);

  int num_dims_;
  size_t elm_size_;
  IntArray size_;
  void *buf_;
 public:
  void (*deleter_)(void*);
  static size_t GetLinearSize(int num_dims, size_t elm_size, const IntArray &s) {
    return s.accumulate(num_dims) * elm_size; }
  
 private:
  void DeleteChunk(void *ptr);
};

class BufferHost: public Buffer {
 public:
  BufferHost(size_t elm_size);
  BufferHost(int num_dims, size_t elm_size);  
  virtual ~BufferHost();
  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size);
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IntArray &offset,
                      const IntArray &size);                      
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size);
  // Assumes 1-D buffer as buf
  virtual void Copyout(BufferHost &buf, const IntArray &offset,
                       const IntArray &size);                       

  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void MPIIrecv(int src, MPI_Comm comm, MPI_Request *req,
                        const IntArray &offset, const IntArray &size);  
  virtual void MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void MPIIsend(int dst, MPI_Comm comm, MPI_Request *req,
                        const IntArray &offset, const IntArray &size);
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
  //void DeleteChunk(void *ptr);

};

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_BUFFER_H_ */
