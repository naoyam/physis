// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_GRID_UTIL_H_
#define PHYSIS_RUNTIME_GRID_UTIL_H_

#include "physis/physis_common.h"
#include "runtime/runtime_common.h"

namespace physis {
namespace runtime {

//! Copy a multi-dimensional sub grid into a continuous buffer.
/*
  \param elm_size The size of each element.
  \param num_dims The number of dimensions of the grid.
  \param grid The source grid.
  \param grid_size The size of each dimension of the grid.
  \param subgrid The destination buffer.
  \param subgrid_offset The offset of the sub grid to copy.
  \param subgrid_size The offset of the sub grid to copy.
 */
void CopyoutSubgrid(size_t elm_size, int num_dims,
                    const void *grid,
                    const IndexArray  &grid_size,
                    void *subgrid,
                    const IndexArray &subgrid_offset,
                    const IndexArray &subgrid_size);

//! Copy a continuous buffer into a multi-dimensional sub grid.
/*
  \param elm_size The size of each element.
  \param num_dims The number of dimensions of the grid.
  \param grid The destination grid.
  \param grid_size The size of each dimension of the grid.
  \param subgrid The source buffer.
  \param subgrid_offset The offset of the sub grid to copy.
  \param subgrid_size The offset of the sub grid to copy.
 */
void CopyinSubgrid(size_t elm_size, int num_dims,
                   void *grid, const IndexArray &grid_size,
                   const void *subgrid,
                   const IndexArray &subgrid_offset,
                   const IndexArray &subgrid_size);


// TODO (Index range): Create two distinctive types: offset_type and
//index_type.
//#define _OFFSET_TYPE intprt_t
#define _OFFSET_TYPE PSIndex

inline _OFFSET_TYPE GridCalcOffset(PSIndex x, PSIndex y, 
                                   PSIndex xsize) {
  return ((_OFFSET_TYPE)x) + ((_OFFSET_TYPE)y) * ((_OFFSET_TYPE)xsize);
}

inline _OFFSET_TYPE GridCalcOffset(PSIndex x, PSIndex y, 
                                   const IndexArray &size) {
  return GridCalcOffset(x, y, size[0]);
}

inline _OFFSET_TYPE GridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                     PSIndex xsize, PSIndex ysize) {
  return ((_OFFSET_TYPE)x) + ((_OFFSET_TYPE)y) * ((_OFFSET_TYPE)xsize)
      + ((_OFFSET_TYPE)z) * ((_OFFSET_TYPE)xsize) * ((_OFFSET_TYPE)ysize);
}

inline _OFFSET_TYPE GridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                     const IndexArray &size) {
  return GridCalcOffset3D(x, y, z, size[0], size[1]);
}  

inline intptr_t GridCalcOffset3D(const IndexArray &index,
                                 const IndexArray &size) {
  return GridCalcOffset3D(index[0], index[1], index[2], size[0], size[1]);  
}

template <int DIM>
inline intptr_t GridCalcOffset(const IndexArray &index,
                               const IndexArray &size) {
  if (DIM == 1) {
    return index[0];
  } else if (DIM == 2) {
    return GridCalcOffset(index[0], index[1], size[0]);
  } else if (DIM == 3) {
    return GridCalcOffset3D(index[0], index[1], index[2],
                            size[0], size[1]);
  } else {
    LOG_ERROR() << "Unsupported dimensionality: " << DIM << "\n";
    PSAbort(1);
    return 0; // just to suppress compiler warning
  }
}

template <int DIM>
inline intptr_t GridCalcOffset(const IndexArray &index,
                               const IndexArray &size,
                               const IndexArray &offset) {
  if (DIM == 1) {
    return index[0] - offset[0];
  } else if (DIM == 2) {
    return GridCalcOffset(index[0]-offset[0], index[1]-offset[1], size[0]);
  } else if (DIM == 3) {
    return GridCalcOffset3D(index[0]-offset[0], index[1]-offset[1],
                            index[2]-offset[2], size[0], size[1]);
  } else {
    LOG_ERROR() << "Unsupported dimensionality: " << DIM << "\n";
    PSAbort(1);
    return 0; // just to suppress compiler warning
  }
}


inline intptr_t GridCalcOffset(const IndexArray &index,
                               const IndexArray &size,
                               int dim) {
  if (dim == 1) {
    return index[0];
  } else if (dim == 2) {
    return GridCalcOffset(index[0], index[1], size[0]);
  } else if (dim == 3) {
    return GridCalcOffset3D(index[0], index[1], index[2],
                            size[0], size[1]);
  } else {
    LOG_ERROR() << "Unsupported dimensionality: " << dim << "\n";
    PSAbort(1);
    return 0; // to suppress warning about no return
  }
}

inline intptr_t GridCalcOffset(const IndexArray &index,
                               const IndexArray &size,
                               const IndexArray &offset,
                               int dim) {
  if (dim == 1) {
    return index[0]-offset[0];
  } else if (dim == 2) {
    return GridCalcOffset(index[0]-offset[0], index[1]-offset[1], size[0]);
  } else if (dim == 3) {
    return GridCalcOffset3D(index[0]-offset[0], index[1]-offset[1],
                            index[2]-offset[2], size[0], size[1]);
  } else {
    LOG_ERROR() << "Unsupported dimensionality: " << dim << "\n";
    PSAbort(1);
    return 0; // to suppress warning about no return
  }
}

void PartitionSpace(int num_dims, int num_procs,
                    const IndexArray &size, 
                    const IntArray &num_partitions,
                    PSIndex **partitions, PSIndex **offsets,
                    std::vector<IntArray> &proc_indices,
                    IndexArray &min_partition);


} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_UTIL_H_ */
