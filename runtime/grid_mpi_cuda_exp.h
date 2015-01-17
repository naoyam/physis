// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_GRID_MPI_CUDA_EXP_H_
#define PHYSIS_RUNTIME_GRID_MPI_CUDA_EXP_H_

#include "runtime/grid_mpi.h"
#include "runtime/buffer.h"
#include "runtime/buffer_cuda.h"
#include "runtime/timing.h"
#include "runtime/grid_space_mpi.h"
#include "physis/physis_mpi_cuda.h"

namespace physis {
namespace runtime {

template <class GridType>
class GridSpaceMPICUDA;

//! Encapsulates each grid instance for the MPI-CUDA backend.
/*!
  Internal multi-dimensional buffer spans both the subgrid and its
  halo regions, and no separate buffer is allocated for neighbor halo 
  regions. This is the same as GridMPI, and different from
  GridMPICUDA3D, which holds halo region data in separate
  buffers. Thus, the "Exp" in the class name. The width of the
  expansion for halo is determined by the argument to the GridNew
  call for this instance. The actual value is currently automatically
  determined by the translator by taking the maximum of the halo
  widths of all stencil functions for this object. However, it
  assumes that grid instantiations and stencil calls are all visible
  in a single compilation unit, which in turn disallows separate
  compilations. Explicit setting of halo width will be possible with
  the future version.
  
  For each stencil call, neighbor exchanges transfer the whole halo
  regions irrespective of the minimum halo width for the particular
  stencil. This simplifies implementation and also avoids
  non-continuous data copies. So, while the amount of data transfered
  may be larger than necessary, actual performance may be
  better. Also, most grids won't have hugely vayring halo widths, so
  this will have little impact in practie.
 */
class GridMPICUDAExp: public GridMPI {
  friend class GridSpaceMPI<GridMPICUDAExp>;  
  friend class GridSpaceMPICUDA<GridMPICUDAExp>;

 protected:
  
  GridMPICUDAExp(const __PSGridTypeInfo *type_info, int num_dims,
                 const IndexArray &size, const IndexArray &global_offset,
                 const IndexArray &local_offset, const IndexArray &local_size,
                 const Width2 &halo,
                 const Width2 *halo_member,
                 int attr);
 public:
  
  static GridMPICUDAExp *Create(
      PSType type, int elm_size, int num_dims,  const IndexArray &size,
      const IndexArray &global_offset, const IndexArray &local_offset,
      const IndexArray &local_size, const Width2 &halo,
      int attr);
  
  static GridMPICUDAExp *Create(
      const __PSGridTypeInfo *type_info,
      int num_dims,  const IndexArray &size,
      const IndexArray &global_offset, const IndexArray &local_offset,
      const IndexArray &local_size, const Width2 &halo,
      const Width2 *halo_member,      
      int attr);
  
  
  virtual ~GridMPICUDAExp();

  // Inherited from parent classes
  size_t CalcHaloSize(int dim, unsigned width, bool diagonal);  
  virtual void InitBuffers();  
  virtual void DeleteBuffers();
  BufferCUDADev *buffer() {
    //return static_cast<BufferCUDADev*>(data_buffer_);
    LOG_ERROR() << "Not supported; member ID needs to be specified\n";
    PSAbort(1);
    return NULL;
  }
  BufferCUDADev *buffer(int member_id) {
    return data_buffer_m_[member_id];
  }
  void *data(int member_id=0) {
    return buffer(member_id)->Get();
  }
  intptr_t idata(int member_id=0) {
    return (intptr_t)data(member_id);
  }
  IndexArray local_real_size(int member) const {
    return local_size_ + halo(member).fw + halo(member).bw;
  }
  size_t local_real_num_elms() const {
    size_t n = 0;
    for (int i = 0; i < num_members(); ++i) {
      n += local_real_size(i).accumulate(num_dims_);
    }
    return n;
  }  
  virtual size_t GetLocalBufferSize() const {
    size_t n = 0;
    for (int i = 0; i < num_members(); ++i) {
      n += GridMPI::GetLocalBufferSize(i);
    }
    return n;
  }
  size_t GetLocalBufferSize(int member) const {
    return GridMPI::GetLocalBufferSize(member);
  }
  virtual size_t GetLocalBufferRealSize() const {
    size_t n = 0;
    for (int i = 0; i < num_members(); ++i) {
      n += GetLocalBufferRealSize(i);
    }
    return n;
  }
  size_t GetLocalBufferRealSize(int member) const {
    return local_real_size(member).accumulate(num_dims_) * elm_size(member);
  }
  IndexArray local_real_offset(int member) const {
    return local_offset_ - halo(member).bw;
  }

  virtual void CopyoutHalo(int dim, const Width2 &width,
                           bool fw, bool diagonal);
  virtual void CopyoutHalo(int dim, const Width2 &width,
                           bool fw, bool diagonal, int member);
  virtual void CopyinHalo(int dim, const Width2 &width,
                          bool fw, bool diagonal);
  virtual void CopyinHalo(int dim, const Width2 &width,
                          bool fw, bool diagonal, int member);
  virtual int Reduce(PSReduceOp op, void *out);
  virtual void Copyout(void *dst);
  virtual void Copyout(void *dst, int member);  
  virtual void Copyin(const void *src);
  virtual void Copyin(const void *src, int member);
  
  // Not inherited
  void *GetDev() { return dev_; }
  //void *GetDev(int member_id) { return dev_; }
  
  void *GetAddress(int member, const IndexArray &indices) {
    return (void*)(idata(member) +
                   GridCalcOffset(indices, local_real_size(member),
                                  local_real_offset(member), num_dims_)
                   * elm_size(member));
  }
  
  void CopyDiag(int dim, bool fw, int member);
  void CopyDiag3D1(bool fw, int member);  
  void CopyDiag3D2(bool fw, int member);
  //void SetCUDAStream(cudaStream_t strm);

 protected:
  BufferCUDADev **data_buffer_m_; // different buffers for different members
  BufferCUDAHost *(*halo_self_host_)[2];
  BufferCUDAHost *(*halo_peer_host_)[2];
  void *dev_;
  
  virtual void FixupBufferPointers();

  BufferCUDAHost *&GetHaloBuf(int dim, bool fw,
                              BufferCUDAHost *(*const & buf_array)[2],
                              int member) const {
    int offset = dim + member * num_dims_;
    return fw ? buf_array[offset][1] : buf_array[offset][0];
  }

  BufferCUDAHost *&GetHaloBuf(int dim, bool fw, 
                              BufferCUDAHost *(*&buf_array)[2],
                              int member) {
    int offset = dim + member * num_dims_;
    return fw ? buf_array[offset][1] : buf_array[offset][0];
  }
  
 public:
#if 0                                          
  virtual BufferCUDAHost *GetHaloSelfHost(int member, int dim, bool fw) const {
    int offset = dim + member * num_dims_;
    return fw ? halo_self_host_[offset][1] : halo_self_host_[offset][0];
  }
  virtual BufferCUDAHost *&GetHaloSelfHost(int member, int dim, bool fw) {
    int offset = dim + member * num_dims_;
    return fw ? halo_self_host_[offset][1] : halo_self_host_[offset][0];
  }
  virtual BufferCUDAHost *GetHaloPeerHost(int member, int dim, bool fw) const {
    int offset = dim + member * num_dims_;    
    return fw ? halo_peer_host_[offset][1] : halo_peer_host_[offset][0];
  }
  virtual BufferCUDAHost *&GetHaloPeerHost(int member, int dim, bool fw) {
    int offset = dim + member * num_dims_;    
    return fw ? halo_peer_host_[offset][1] : halo_peer_host_[offset][0];
  }
#else  
  virtual BufferCUDAHost *&GetHaloSelfHost(
      int dim, bool fw, int member) const {
    return GetHaloBuf(dim, fw, halo_self_host_, member);
  }
  virtual BufferCUDAHost *&GetHaloSelfHost(
      int dim, bool fw, int member) {
    return GetHaloBuf(dim, fw, halo_self_host_, member);
  }
  virtual BufferCUDAHost *&GetHaloPeerHost(
      int dim, bool fw, int member) const {
    return GetHaloBuf(dim, fw, halo_peer_host_, member);
  }
  virtual BufferCUDAHost *&GetHaloPeerHost(
      int dim, bool fw, int member) {
    return GetHaloBuf(dim, fw, halo_peer_host_, member);    
  }
#endif  
};

  
} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_CUDA_H_ */
