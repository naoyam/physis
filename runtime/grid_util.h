// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

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



inline intptr_t GridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                 PSIndex xsize, PSIndex ysize) {
  return ((intptr_t)x) + ((intptr_t)y) * ((intptr_t)xsize)
      + ((intptr_t)z) * ((intptr_t)xsize) * ((intptr_t)ysize);
}

inline intptr_t GridCalcOffset3D(PSIndex x, PSIndex y, PSIndex z, 
                                 const IndexArray &size) {
  return GridCalcOffset3D(x, y, z, size[0], size[1]);
}  

inline intptr_t GridCalcOffset3D(const IndexArray &index,
                                 const IndexArray &size) {
  return GridCalcOffset3D(index[0], index[1], index[2], size[0], size[1]);  
}

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_UTIL_H_ */
