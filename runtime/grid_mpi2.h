// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_GRID_MPI2_H_
#define PHYSIS_RUNTIME_GRID_MPI2_H_

#define __STDC_LIMIT_MACROS
#include "mpi.h"

#include "runtime/runtime_common.h"
#include "runtime/grid.h"
#include "runtime/grid_mpi.h"
#include "runtime/mpi_util.h"

namespace physis {
namespace runtime {

class GridSpaceMPI2;

class GridMPI2: public GridMPI {
  friend class GridSpaceMPI2;

 protected:
  GridMPI2(PSType type, int elm_size, int num_dims,
           const IndexArray &size,
           const IndexArray &global_offset, const IndexArray &local_offset,
           const IndexArray &local_size, 
           const Width2 &halo,
           int attr);
 public:
  static GridMPI2 *Create(
      PSType type, int elm_size,
      int num_dims, const IndexArray &size,
      const IndexArray &global_offset,
      const IndexArray &local_offset,
      const IndexArray &local_size,
      const Width2 &halo,
      int attr);
  virtual ~GridMPI2();
  virtual std::ostream &Print(std::ostream &os) const;

  const IndexArray& local_real_size() const { return local_real_size_; }
  size_t GetLocalBufferRealSize() const {
    return local_real_size_.accumulate(num_dims_) * elm_size_;
  };
  const IndexArray& local_offset() const { return local_offset_; }
  bool empty() const { return empty_; }
  const Width2 &halo() const { return halo_; }
  bool HasHalo() const { return ! (halo_.fw == 0 && halo_.bw == 0); }
  
  // Allocates buffers
  virtual void InitBuffers();
  //! Allocates buffers for halo communications
  virtual void InitHaloBuffers();
  //! Deletes buffers
  virtual void DeleteBuffers();
  //! Deletes halo buffers
  virtual void DeleteHaloBuffers();
  // Returns buffer for remote halo
  /*
    \param dim Access dimension.
    \param fw Flag to indicate access direction.
    \param width Halo width.
    \return Buffer pointer.
  */
  char *GetHaloPeerBuf(int dim, bool fw, unsigned width);

  // These are not virtual because they are used inside kernel inner
  // loops 
  void *GetAddress(const IndexArray &indices_param);

  template <int dim>
  PSIndex CalcOffset(const IndexArray &indices_param) {
#if 0
    IndexArray indices = indices_param;
    indices -= local_real_offset_;
    return GridCalcOffset3D(indices, local_real_size_);
#elif 0    
    return (indices_param[0] - local_real_offset_[0]) +
        (indices_param[1] - local_real_offset_[1]) * local_real_size_[0] +
        (indices_param[2] - local_real_offset_[2]) *
        local_real_size_[0] *local_real_size_[1];
#else
    PSIndex off = indices_param[0] - local_real_offset_[0];
    if (dim > 1)
      off += (indices_param[1] - local_real_offset_[1]) * local_real_size_[0];
    if (dim > 2)
      off +=
          (indices_param[2] - local_real_offset_[2]) *
          local_real_size_[0] *local_real_size_[1];
#endif
    return off;
  }

  template <int dim>
  PSIndex CalcOffsetPeriodic(const IndexArray &indices_param) {
#if 0    
    IndexArray indices = indices_param;
    for (int i = 0; i < num_dims_; ++i) {
      // No halo if no domain decomposition is done for a
      // dimension. Periodic access must be done by wrap around the offset
      if (local_size_[i] == size_[i]) {
        indices[i] = (indices[i] + size_[i]) % size_[i];
      } else {
        indices[i] -= local_real_offset_[i];
      }
    }
    return GridCalcOffset3D(indices, local_real_size_);
#else
    PSIndex off =
        local_size_[0] == size_[0] ? 
        (indices_param[0] + size_[0]) % size_[0] :
        indices_param[0] - local_real_offset_[0];
    if (dim > 1)
      off +=
          (local_size_[1] == size_[1] ? 
           (indices_param[1] + size_[1]) % size_[1] :
           indices_param[1] - local_real_offset_[1])
          * local_real_size_[0];
    if (dim > 2)
      off +=
          (local_size_[2] == size_[2] ? 
           (indices_param[2] + size_[2]) % size_[2] :
           indices_param[2] - local_real_offset_[2])
          * local_real_size_[0] * local_real_size_[1];
    return off;
#endif
  }


  // Reduction
  virtual int Reduce(PSReduceOp op, void *out);


  virtual void Copyout(void *dst) const;
  virtual void Copyin(const void *src);
  
  virtual void CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal);
  virtual void CopyinHalo(int dim, unsigned width, bool fw, bool diagonal);
  
 protected:
  IndexArray local_real_offset_;  
  IndexArray local_real_size_;  
  Width2 halo_;

  size_t CalcHaloSize(int dim, unsigned width);
  void SetHaloSize(int dim, bool fw, unsigned width, bool diagonal);
};

class GridSpaceMPI2: public GridSpaceMPI {
 public:
  GridSpaceMPI2(int num_dims, const IndexArray &global_size,
                int proc_num_dims, const IntArray &proc_size,
                int my_rank);
  virtual ~GridSpaceMPI2();
  virtual GridMPI2 *CreateGrid(PSType type, int elm_size, int num_dims,
                               const IndexArray &size, 
                               const IndexArray &global_offset,
                               const IndexArray &stencil_offset_min,
                               const IndexArray &stencil_offset_max,
                               int attr);
  virtual GridMPI *LoadNeighbor(GridMPI *g,
                                const IndexArray &offset_min,
                                const IndexArray &offset_max,
                                bool diagonal,
                                bool reuse,
                                bool periodic);

 protected:
  void ExchangeBoundariesAsync(
      GridMPI *grid, int dim,
      unsigned halo_fw_width, unsigned halo_bw_width,
      bool diagonal, bool periodic,
      std::vector<MPI_Request> &requests) const;
  void ExchangeBoundaries(GridMPI *grid,
                          int dim,
                          unsigned halo_fw_width,
                          unsigned halo_bw_width,
                          bool diagonal,
                          bool periodic) const;
  
};


} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI2_H_ */


