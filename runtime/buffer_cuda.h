// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_BUFFER_CUDA_H_
#define PHYSIS_RUNTIME_BUFFER_CUDA_H_

#include "runtime/buffer.h"
#include <cuda_runtime.h>

namespace physis {
namespace runtime {

class BufferCUDADev;

// Pinned buffer on *host* memory
class BufferCUDAHost: public BufferHost {
 public:
  BufferCUDAHost();
  virtual ~BufferCUDAHost();
  template <class T>
  std::ostream& print(std::ostream &os) const {
    StringJoin sj;
    T *p = static_cast<T*>(buf_);
    for (int i = 0; i < size_ / sizeof(T); ++i) {
      sj << p[i];
    }
    os << "{" << sj.str() << "}";
    return os;
  }

 protected:
  virtual void *GetChunk(size_t size);
 private:
  static void FreeChunk(void *ptr);  
};

class BufferCUDADev: public Buffer {
 public:
  BufferCUDADev();
  virtual ~BufferCUDADev();
  virtual void Copyout(void *dst, size_t size);  
  virtual void Copyout(size_t elm_size, int rank,
                       const IndexArray  &grid_size,
                       void *subgrid,
                       const IndexArray &subgrid_offset,
                       const IndexArray &subgrid_size);
  virtual void Copyin(const void *src, size_t size);  
  virtual void Copyin(size_t elm_size, int rank,
                      const IndexArray  &grid_size,
                      const void *subgrid,
                      const IndexArray &subgrid_offset,
                      const IndexArray &subgrid_size);
  
 protected:
  virtual void *GetChunk(size_t size);
  BufferCUDADev *temp_buffer_;
 private:
  static void FreeChunk(void *ptr);
};


} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_BUFFER_CUDA_H_ */
