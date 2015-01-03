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
  GridMPICUDAExp(PSType type, int elm_size, int num_dims, const IndexArray &size,
                 const IndexArray &global_offset,
                 const IndexArray &local_offset, const IndexArray &local_size,
                 const Width2 &halo,                 
                 int attr);
 public:
  
  static GridMPICUDAExp *Create(
      PSType type, int elm_size, int num_dims,  const IndexArray &size,
      const IndexArray &global_offset,
      const IndexArray &local_offset, const IndexArray &local_size,
      const Width2 &halo,
      int attr);
  
  virtual ~GridMPICUDAExp();

  // Inherited from parent classes
  size_t CalcHaloSize(int dim, unsigned width, bool diagonal);  
  virtual void InitBuffers();  
  virtual void DeleteBuffers();
  BufferCUDADev *buffer() { return static_cast<BufferCUDADev*>(data_buffer_); }
  virtual void CopyoutHalo(int dim, const Width2 &width, bool fw, bool diagonal);
  virtual void CopyinHalo(int dim, const Width2 &width, bool fw, bool diagonal);
  virtual int Reduce(PSReduceOp op, void *out);
  virtual void Copyout(void *dst);
  virtual void Copyin(const void *src);
  
  // Not inherited
  void *GetDev() { return dev_; }
  void CopyDiag(int dim, const Width2 &width, bool fw);
  void CopyDiag3D1(const Width2 &width, bool fw);  
  void CopyDiag3D2(const Width2 &width, bool fw);
  //void SetCUDAStream(cudaStream_t strm);
  
 protected:
  BufferCUDAHost *(*halo_self_host_)[2];
  BufferCUDAHost *(*halo_peer_host_)[2];
  BufferCUDADev *(*halo_peer_dev_)[2];
  void *dev_;
  
  virtual void FixupBufferPointers();
  
 public:
  virtual BufferCUDAHost *GetHaloSelfHost(int dim, bool fw) const {
    return fw ? halo_self_host_[dim][1] : halo_self_host_[dim][0];
  }
  virtual BufferCUDAHost *&GetHaloSelfHost(int dim, bool fw) {
    return fw ? halo_self_host_[dim][1] : halo_self_host_[dim][0];
  }
  virtual BufferCUDAHost *GetHaloPeerHost(int dim, bool fw) const {
    return fw ? halo_peer_host_[dim][1] : halo_peer_host_[dim][0];
  }
  virtual BufferCUDAHost *&GetHaloPeerHost(int dim, bool fw) {
    return fw ? halo_peer_host_[dim][1] : halo_peer_host_[dim][0];
  }
  
};

  
} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_CUDA_H_ */
