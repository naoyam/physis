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
  bool HasHalo() const { return halo_.fw == 0 && halo_.bw == 0; }
  
  // Buffer management
  virtual void InitBuffer();
  virtual void DeleteBuffers();

  virtual void *GetAddress(const IndexArray &indices_param);

  // Reduction
  virtual int Reduce(PSReduceOp op, void *out) {
    LOG_ERROR() << "Not implemented.\n";
    PSAbort(1);
    return 0;
  }

  virtual void Copyout(void *dst) const;
  virtual void Copyin(const void *src);
  
 protected:
  bool empty_;
  IndexArray global_offset_;
  IndexArray local_offset_;
  IndexArray local_real_offset_;  
  IndexArray local_size_;
  IndexArray local_real_size_;  
  bool halo_has_diagonal_;
  Width2 halo_;

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
                               const Width2 &stencil_width,
                               int attr);
};


} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_MPI2_H_ */


