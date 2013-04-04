// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_GRID_MPI_H_
#define PHYSIS_RUNTIME_GRID_MPI_H_

#include <iostream>
#include <sstream>

#include "runtime/runtime_common.h"
#include "runtime/grid.h"
#include "runtime/grid_util.h"

namespace physis {
namespace runtime {

class GridSpaceMPI;

// TODO: Replace MPI with class IPC.
class GridMPI: public Grid {
  friend class GridSpaceMPI;
  template <class T> friend std::ostream& print_grid(
      GridMPI *, int, std::ostream&);
 protected:
  GridMPI(PSType type, int elm_size, int num_dims,
          const IndexArray &size,
          const IndexArray &global_offset,
          const IndexArray &local_offset,
          const IndexArray &local_size, 
          const Width2 &halo,
          int attr);
  //! Flag to indicate whether this sub grid is empty
  bool empty_;
  //! Offset of the whole grid within the global domain.
  /*!
    This is usually 0. If non-zero, the grid is located within the
    whole global domain. This is experimentally used to control where
    to put a grid within the global domain. For example, a grid that
    is only used with a partial area of another grid, then it would be
    desirable not to decompose the grid and to be kept within the same
    process as the partial area of the other grid.
   */
  IndexArray global_offset_;
  //! Offset of the sub grid within the whole grid.
  IndexArray local_offset_;
  //! Length of the logical sub grid w/o including halo.
  IndexArray local_size_;

  //! Width of the halo for each dimension.
  /*!
    halo_.bw[i] and halo_.fw[i] are the (unsigned) width of the
    backward and forward halo in the i'th dimension, respectively.
   */
  Width2 halo_; 
  //! Offset of the actual buffer within the whole grid.
  IndexArray local_real_offset_;    
  //! Length of the actual buffer with halo
  IndexArray local_real_size_;

  //! Buffer for sending halo for forward accesses
  char **halo_self_fw_;
  //! Buffer for sending halo for backward accesses
  char **halo_self_bw_;
  //! Buffer for receiving halo for forward accesses
  char **halo_peer_fw_;
  //! Buffer for receiving halo for backward accesses  
  char **halo_peer_bw_;

  size_t CalcHaloSize(int dim, unsigned width);    
  
  //! Allocates buffers, including halo buffers.
  virtual void InitBuffers();
  //! Allocates buffers for halo communications.
  virtual void InitHaloBuffers();
  //! Deletes buffers, including halo buffers.
  virtual void DeleteBuffers();
  //! Deletes halo buffers.
  virtual void DeleteHaloBuffers();

  // Returns buffer for remote halo
  /*
    \param dim Access dimension.
    \param fw Flag to indicate access direction.
    \param width Halo width.
    \return Buffer pointer.
  */
  char *GetHaloPeerBuf(int dim, bool fw, unsigned width);

  //! Copy halo from the grid buffer into the send buffer.
  /*!
    \param dim Dimension to copy.
    \param width Halo length.
    \param fw True if the received halo is for forward accesses. 
    \param diagonal Flag to indicate diagonal point is used.
   */
  virtual void CopyoutHalo(int dim, unsigned width, bool fw, bool diagonal);
  
  //! Copy remote halo from the recv buffer into the grid buffer.
  /*!
    \param dim Dimension to copy.
    \param width Halo length.
    \param fw True if the received halo is for forward accesses. 
    \param diagonal Flag to indicate diagonal point is used.
   */
  virtual void CopyinHalo(int dim, unsigned width, bool fw, bool diagonal);

 public:
  static GridMPI *Create(
      PSType type, int elm_size,
      int num_dims, const IndexArray &size,
      const IndexArray &global_offset,
      const IndexArray &local_offset,
      const IndexArray &local_size,
      const Width2 &halo,
      int attr);
  
  virtual ~GridMPI();
  virtual std::ostream &Print(std::ostream &os) const;
  
  bool empty() const { return empty_; }
  const IndexArray& local_size() const { return local_size_; }  
  const IndexArray& local_offset() const { return local_offset_; }
  const IndexArray& local_real_size() const { return local_real_size_; }
  const Width2 &halo() const { return halo_; }
  bool HasHalo() const { return ! (halo_.fw == 0 && halo_.bw == 0); }  
  
  virtual int Reduce(PSReduceOp op, void *out);

  //! Copy out the grid data (w/o halo).
  /*!
    \param dst Destination buffer.
   */
  virtual void Copyout(void *dst) const;
  //! Copy grid data into this grid.
  /*!
    \param src Source buffer to copy.
   */
  virtual void Copyin(const void *src);

  //! Get the address of an grid element.
  /*!
    Keep definition here to make it inlined.
    
    \param indices Position of the element.
    \return Address of the element.
   */
  void *GetAddress(const IndexArray &indices) {
    IndexArray t = indices;
    t -= local_real_offset_;
    return (void*)(_data() +
                   GridCalcOffset3D(t, local_real_size_)
                   * elm_size());
  }
  
  //! Get the offset of an grid element.
  /*!
    \param indices Position of the element.
    \return Offset of the element in length.
   */
  template <int dim>
  PSIndex CalcOffset(const IndexArray &indices) {
    PSIndex off = indices[0] - local_real_offset_[0];
    if (dim > 1)
      off += (indices[1] - local_real_offset_[1]) * local_real_size_[0];
    if (dim > 2)
      off +=
          (indices[2] - local_real_offset_[2]) *
          local_real_size_[0] *local_real_size_[1];
    return off;
  }

  //! Get the offset of an grid element with periodic access.
  /*!
    The indices may be outside the grid logical domain. In that case,
    the access is wrapped around.
    
    \param indices Position of the element.
    \return Offset of the element in length.
   */
  template <int dim>
  PSIndex CalcOffsetPeriodic(const IndexArray &indices) {
    PSIndex off =
        local_size_[0] == size_[0] ? 
        (indices[0] + size_[0]) % size_[0] :
        indices[0] - local_real_offset_[0];
    if (dim > 1)
      off +=
          (local_size_[1] == size_[1] ? 
           (indices[1] + size_[1]) % size_[1] :
           indices[1] - local_real_offset_[1])
          * local_real_size_[0];
    if (dim > 2)
      off +=
          (local_size_[2] == size_[2] ? 
           (indices[2] + size_[2]) % size_[2] :
           indices[2] - local_real_offset_[2])
          * local_real_size_[0] * local_real_size_[1];
    return off;
  }
  
  //! Returns the size of the logical buffer area in bytes.
  /*!
    Does not count the halo region.
    
    \return Size in bytes.
  */
  size_t GetLocalBufferSize() const {
    return local_size_.accumulate(num_dims_) * elm_size_;
  };

  //! Returns the size of the actual buffer area in bytes.
  /*!
    Does count the halo region.
    
    \return Size in bytes.
  */
  size_t GetLocalBufferRealSize() const {
    return local_real_size_.accumulate(num_dims_) * elm_size_;
  };
};

} // namespace runtime
} // namespace physis

inline std::ostream& operator<<(std::ostream &os,
                                physis::runtime::GridMPI &g) {
  return g.Print(os);
}

#endif /* PHYSIS_RUNTIME_GRID_MPI_H_ */

