// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_BUFFER_OPENCL_H_
#define PHYSIS_RUNTIME_BUFFER_OPENCL_H_

#include "runtime/buffer.h"
#include "runtime/rpc_opencl_common.h"

namespace physis {
namespace runtime {

class BufferOpenCLDev;

// Pinned buffer on *host* memory
class BufferOpenCLHost: public Buffer {
 public:
  BufferOpenCLHost(size_t elm_size);
  BufferOpenCLHost(int num_dims, size_t elm_size);  
  virtual ~BufferOpenCLHost();
  
  virtual void Copyin(const void *buf, const IndexArray &offset,
                      const IndexArray &size);
  virtual void Copyin(const void *buf, const IndexArray &size) {
    Copyin(buf, IndexArray(), size);
  }
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IntArray &offset,
                      const IntArray &size);                       
 
  virtual void Copyout(void *buf, const IndexArray &offset,
                       const IndexArray &size);
  virtual void Copyout(void *buf,  const IndexArray &size) {
    Copyout(buf, IndexArray(), size);
  }
  virtual void Copyout(BufferHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void Copyout(BufferHost &buf,  const IntArray &size) {
    Copyout(buf, IntArray(), size);
  }
  
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);

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


class BufferOpenCLDev: public Buffer {
 public:
  BufferOpenCLDev(size_t elm_size, CLbaseinfo *clinfo_in);
  BufferOpenCLDev(int num_dims, size_t elm_size, CLbaseinfo *clinfo_in);  
  virtual ~BufferOpenCLDev();
  
  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size);
  // Assumes 1-D buffer as buf  
  virtual void Copyin(const BufferHost &buf, const IntArray &offset,
                      const IntArray &size);                      
  virtual void Copyin(const BufferOpenCLHost &buf, const IntArray &offset,
                      const IntArray &size);                      
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout(BufferHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void Copyout(BufferOpenCLHost &buf, const IntArray &offset,
                       const IntArray &size);                       
  virtual void Copyout(BufferOpenCLHost &buf, const IntArray &offset,
                       const IntArray &size, const IntArray &total_size);                       
  virtual void MPIRecv(int src, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);
  virtual void MPISend(int dst, MPI_Comm comm, const IntArray &offset,
                       const IntArray &size);

  size_t GetPitch() const {
    return pitch;
  }

  virtual cl_mem Get_buf_mem() { return buf_mem; };
  virtual const void *Get() const;
  virtual void *&Get();
  virtual void GetChunk_CL(
      const IntArray &size, cl_mem *ret_p_mem, size_t *ret_pitch);

 protected:
  virtual void *GetChunk(const IntArray &size);
  virtual void Allocate(int num_dims, size_t elm_size, const IntArray &size);  
  virtual void Delete();
  BufferOpenCLHost *pinned_buf_;
  size_t pitch;

  CLbaseinfo *base_clinfo_;
  CLbaseinfo *stream_clinfo_;
  cl_mem buf_mem;

 public:
  void DeleteChunk_CL(cl_mem p_mem);
  static void DeleteChunk(void *ptr);

 public:
  CLbaseinfo *&stream_clinfo() { return stream_clinfo_; }
  CLbaseinfo *&base_clinfo() { return base_clinfo_; }

 protected:
  void *tmp_buf_;
  size_t tmp_buf_size_;

 public:
  void *&tmpbuf() { return tmp_buf_; }
  size_t &tmpbuf_size() { return tmp_buf_size_; }

};

#if 0
#else
#define BufferOpenCLDev3D BufferOpenCLDev
#endif


} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_BUFFER_OPENCL_H_ */
