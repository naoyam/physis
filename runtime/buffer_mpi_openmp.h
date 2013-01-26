#ifndef PHYSIS_RUNTIME_BUFFER_MPI_OPENMP_H_
#define PHYSIS_RUNTIME_BUFFER_MPI_OPENMP_H_

#include "runtime/runtime_common.h"
#include "runtime/mpi_wrapper.h"
#include "buffer.h"

namespace physis {
namespace runtime {

class BufferHostOpenMP: public BufferHost {
 protected:

 public:
  BufferHostOpenMP(size_t elm_size);  
  BufferHostOpenMP(int num_dims, size_t elm_size);
  BufferHostOpenMP(int num_dims, size_t elm_size, IntArray &division_in);

  virtual ~BufferHostOpenMP();
  virtual void Allocate(int num_dims, size_t elm_size,
                        const IntArray &size);
  void Allocate(const IntArray &size);

 private:
  void DeleteOpenMP();

 public:

  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size);
  virtual void Copyin(const void *buf, const IntArray &offset,
                      const IntArray &size, const size_t linear_offset);

  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size);
  virtual void Copyout(void *buf, const IntArray &offset,
                       const IntArray &size, const size_t linear_offset);

 public:
  virtual void MPISendRecvInoI(
      int recv_p,
      int blocking_p,
      int srcdstNUM,
      MPI_Comm comm, MPI_Request *req,
      const IntArray &offset, const IntArray &size,
      const size_t *cpu_memsize
                               );
  
 protected:
  virtual int CreateAlignedMultiBuffer(
      const IntArray &requested_size, IntArray &division,
      const int elmsize,
      const size_t alignment,
      void ***ret_buf,
      size_t **&ret_offset, size_t **&ref_width,
      size_t *&ret_cpu_memsize,
      size_t *&ret_cpu_allocbytes,
      size_t &linesize
                                       );

  virtual void DestroyMultiBuffer(
      void ***src_buf, const IntArray &division
                                  );
  virtual void DestroyMP3Dinfo(
      size_t **&mp_3dinfo, const IntArray &division
                               );

 protected:
  void **buf_mp_; // release with free

 protected:
  IntArray division_;
  size_t **mp_offset_; // release with new
  size_t **mp_width_; // release with new
  size_t mp_linesize_;
  size_t *mp_cpu_memsize_; // release with new
  size_t *mp_cpu_allocBytes_; // release with new, in bytes!!
  // Note that mp_offset_, mp_width_, mp_linesize_, mp_cpu_memsize_
  // are not multiplied by elm_size 

 public:
  void **&Get_MP() { return buf_mp_; }
  const IntArray &MPdivision() { return division_; }
  size_t **MPoffset() { return mp_offset_; }
  size_t **MPwidth() { return mp_width_; }
  size_t *MPcpumemsize() { return mp_cpu_memsize_; }
  size_t *MPcpuallocBytes() { return mp_cpu_allocBytes_; }

 public:
  void MoveMultiBuffer(unsigned int maxcpunodes);
 private:
  bool mbind_done_p;

 public:
  const size_t elm_size() const { return elm_size_; }

};

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_BUFFER_MPI_OPENMP_H_ */
