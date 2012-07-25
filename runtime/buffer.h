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
  //! Create a buffer
  /*!
    Actual memory chunk is not allocated. Number of dimensions and the
    buffer size are determined with their default values. 
    
    \param elm_size Size of each element in bytes
  */
  Buffer(size_t elm_size);  
 public:
  //! Create a buffer
  /*!
    Actual memory chunk is not allocated.
    
    \param num_dims Number of dimensions
    \param elm_size Size of each element in bytes
  */
  Buffer(int num_dims, size_t elm_size);  
  virtual ~Buffer();
  //! Allocate a chunk of memory for the buffer
  /*!
    \param num_dims Number of dimensions
    \param elm_size Size of each element in bytes
    \param size Number of elements
  */
  virtual void Allocate(int num_dims, size_t elm_size,
                        const IndexArray &size);
  //! Allocate a chunk of memory for the buffer
  /*!
    The dimensionality and element size are determined with the class
    member variables.
    
    \param size Number of elements
  */
  void Allocate(const IndexArray &size) {
    Allocate(num_dims_, elm_size_, size);
  }
  //! Get the pointer to the data memory chunk
  void *&Get() { return buf_; }
  //! Get the pointer to the data memory chunk
  const void *Get() const { return buf_; }
  //! Returns the size of buffer
  virtual const IndexArray &size() const { return size_; }
  //! Copy data to this buffer.
  /*! 
    \param buf Source memory chunk
    \param offset Offset to the destination address
    \param size Number of elements to copy
  */
  virtual void Copyin(const void *buf, const IndexArray &offset,
                      const IndexArray &size) = 0;
  //! Copy data to this buffer.
  /*!
    This is equivalent to Copyin(buf, SizeArray(0), size).
    
    \param buf Source memory chunk
    \param size Number of elements to copy
  */
  void Copyin(const void *buf, PSIndex size);
  //! Copy data from this buffer.
  /*!
    \param buf Destination memory chunk    
    \param offset Offset to the source address
    \param size Number of elements to copy
  */
  virtual void Copyout(void *buf, const IndexArray &offset,
                       const IndexArray &size) = 0;
  //! Copy data from this buffer.
  /*!
    This is equivalent to Copyout(buf, SizeArray(0), size).
    
    \param buf Destination memory chunk    
    \param size Number of elements to copy
  */
  void *Copyout();
  virtual void MPIRecv(int src, MPI_Comm comm, const IndexArray &offset,
                       const IndexArray &size) = 0;
  void MPIRecv(int src, MPI_Comm comm, const IndexArray &size) {
    MPIRecv(src, comm, IndexArray(), size);
  }
  virtual void MPISend(int dst, MPI_Comm comm, const IndexArray &offset,
                       const IndexArray &size) = 0;
  void MPISend(int dst, MPI_Comm comm, const IndexArray &size) {
    MPISend(dst, comm, IndexArray(), size);
  }
  virtual void Delete();
  virtual void EnsureCapacity(PSIndex size) {
    EnsureCapacity(elm_size_, size);
  }
  virtual void EnsureCapacity(const IndexArray &size) {
    EnsureCapacity(num_dims_, elm_size_, size);
  }
  virtual void EnsureCapacity(size_t elm_size, PSIndex size);
  virtual void EnsureCapacity(int num_dims, size_t elm_size,
                              const IndexArray &size);
  //! Return the number of dimensions
  int num_dims() const { return num_dims_; }
  virtual size_t GetLinearSize(const IndexArray &s) const {
    return s.accumulate(num_dims_) * elm_size_; }
  virtual size_t GetLinearSize() const {
    return GetLinearSize(size_); }

 protected:
  virtual void Shrink(PSIndex size);
  virtual void Shrink(const IndexArray &size);
  virtual void *GetChunk(const IndexArray &size);

  //! Number of dimensions
  int num_dims_;
  //! Size of each element in bytes
  size_t elm_size_;
  //! Number of elements
  IndexArray size_;
  //! Memory chunk for the buffer data
  void *buf_;
 public:
  void (*deleter_)(void*);
  static size_t GetLinearSize(int num_dims, size_t elm_size, const IndexArray &s) {
    return s.accumulate(num_dims) * elm_size; }
  
 private:
  void DeleteChunk(void *ptr);
};

class BufferHost: public Buffer {
 public:
  BufferHost(size_t elm_size);
  BufferHost(int num_dims, size_t elm_size);  
  virtual ~BufferHost();
  virtual void Copyin(const void *buf, const IndexArray &offset,
                      const IndexArray &size);
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IndexArray &offset,
                      const IndexArray &size);                      
  virtual void Copyout(void *buf, const IndexArray &offset,
                       const IndexArray &size);
  // Assumes 1-D buffer as buf
  virtual void Copyout(BufferHost &buf, const IndexArray &offset,
                       const IndexArray &size);                       

  virtual void MPIRecv(int src, MPI_Comm comm, const IndexArray &offset,
                       const IndexArray &size);
  virtual void MPIIrecv(int src, MPI_Comm comm, MPI_Request *req,
                        const IndexArray &offset, const IndexArray &size);  
  virtual void MPISend(int dst, MPI_Comm comm, const IndexArray &offset,
                       const IndexArray &size);
  virtual void MPIIsend(int dst, MPI_Comm comm, MPI_Request *req,
                        const IndexArray &offset, const IndexArray &size);
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
  virtual void *GetChunk(const IndexArray &size);
  //void DeleteChunk(void *ptr);

};

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_BUFFER_H_ */
