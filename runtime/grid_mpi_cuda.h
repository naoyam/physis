// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_GRID_MPI_CUDA_H_
#define PHYSIS_RUNTIME_GRID_MPI_CUDA_H_

#include "physis/physis_common.h"
#include "runtime/grid_mpi.h"
#include "runtime/buffer.h"
#include "runtime/buffer_cuda.h"
#include "runtime/timing.h"
#include "runtime/grid_space_mpi.h"
#include "physis/physis_mpi_cuda.h"


#define USE_MAPPED 0

namespace physis {
namespace runtime {

class GridSpaceMPICUDA;

class GridMPICUDA3D: public GridMPI {
  friend class GridSpaceMPICUDA;
 protected:  
  GridMPICUDA3D(PSType type, int elm_size, int num_dims, const IndexArray &size,
                const IndexArray &global_offset,
                const IndexArray &local_offset, const IndexArray &local_size,
                int attr);
  __PSGridDev3D dev_;
#if 0  
  // the source is address of ordinary memory region
  virtual void Copyin(void *dst, const void *src, size_t size);
  // the dstination is address of ordinary memory region  
  virtual void Copyout(void *dst, const void *src, size_t size);
#endif  
 public:
  
  static GridMPICUDA3D *Create(
      PSType type, int elm_size, int num_dims,  const IndexArray &size,
      const IndexArray &global_offset,
      const IndexArray &local_offset, const IndexArray &local_size,
      int attr);
  
  virtual ~GridMPICUDA3D();
  
  // Inherited from GridMPI
  virtual void DeleteBuffers();
  virtual void InitBuffers();
  virtual std::ostream &Print(std::ostream &os) const;
  virtual void CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal);
  virtual void *GetAddress(const IndexArray &indices);
  
  // Not inherited
  __PSGridDev3D *GetDev() { return &dev_; }
  virtual void CopyoutHalo3D0(unsigned width, bool fw);
  virtual void CopyoutHalo3D1(unsigned width, bool fw);
  void SetCUDAStream(cudaStream_t strm);
  virtual int Reduce(PSReduceOp op, void *out);
  
  // Unused?
#ifdef DEPRECATED  
  virtual void EnsureRemoteGrid(const IndexArray &loal_offset,
                                const IndexArray &local_size);
#endif  

#ifdef CHECKPOINT_ENABLED
  virtual void Save();
  virtual void Restore();
#endif
  
  // REFACTORING: this is an ugly fix to make things work...
  //protected:
 public:
#if USE_MAPPED  
  BufferCUDAHostMapped *(*halo_self_cuda_)[2];
#else  
  BufferCUDAHost *(*halo_self_cuda_)[2];
#endif
  BufferHost *(*halo_self_mpi_)[2];
  BufferCUDAHost *(*halo_peer_cuda_)[2];
  BufferCUDADev *(*halo_peer_dev_)[2];
  virtual void FixupBufferPointers();
};

  
} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI_CUDA_H_ */
