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

  const IndexArray& local_size() const { return local_size_; }
  const IndexArray& local_real_size() const { return local_real_size_; }
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

  virtual void *GetAddress(const IndexArray &indices_param);

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


