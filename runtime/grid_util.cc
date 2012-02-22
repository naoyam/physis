// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/grid_util.h"

using namespace physis::runtime;
using namespace physis::util;

static size_t get1DOffset(const IntArray &md_offset,
                          const IntArray &size,
                          int num_dims) {
  size_t offset_1d = 0;
  size_t ref_offset = 1;
  for (int i = 0; i < num_dims; ++i) {
    offset_1d += (md_offset[i] * ref_offset);
    ref_offset *= size[i];
  }
  return offset_1d;
}

namespace physis {
namespace runtime {

void CopyoutSubgrid(size_t elm_size, int num_dims,
                    const void *grid, const IntArray &grid_size,
                    void *subgrid,
                    const IntArray &subgrid_offset,
                    const IntArray &subgrid_size) {
  intptr_t subgrid_original = (intptr_t)subgrid; // just for sanity checking
  std::list<IntArray> *offsets = new std::list<IntArray>;
  std::list<IntArray> *offsets_new = new std::list<IntArray>;
  
  LOG_DEBUG() << __FUNCTION__ << ": "
              << "subgrid offset: " << subgrid_offset
              << "subgrid size: " << subgrid_size
              << "grid size: " << grid_size
              << "\n";

  // Collect 1-D continuous regions
  offsets->push_back(subgrid_offset);
  for (int i = num_dims - 1; i >= 1; --i) {
    FOREACH (oit, offsets->begin(), offsets->end()) {
      const IntArray &cur_offset = *oit;
      for (int j = 0; j < subgrid_size[i]; ++j) {
        IntArray new_offset = cur_offset;
        new_offset[i] += j;
        offsets_new->push_back(new_offset);
      }
    }
    std::swap(offsets, offsets_new);
    offsets_new->clear();
  }

  // Copy the collected 1-D continuous regions
  size_t line_size = subgrid_size[0] * elm_size;
  FOREACH (oit, offsets->begin(), offsets->end()) {
    const IntArray &offset = *oit;
    const void *src =
        (const void *)((intptr_t)grid +
                       get1DOffset(offset, grid_size, num_dims) * elm_size);
    memcpy(subgrid, src, line_size);
    subgrid = (void*)((intptr_t)subgrid + line_size);
  }

  delete offsets;
  delete offsets_new;

  void *p = (void *)(subgrid_original +
                     subgrid_size.accumulate(num_dims) * elm_size);
  assert(subgrid == p);

  return;
}

void CopyinSubgrid(size_t elm_size, int num_dims,
                   void *grid, const IntArray &grid_size,
                   const void *subgrid,
                   const IntArray &subgrid_offset,
                   const IntArray &subgrid_size) {

  LOG_DEBUG() << __FUNCTION__ << ": "
              << "subgrid offset: " << subgrid_offset
              << "subgrid size: " << subgrid_size
              << "grid size: " << grid_size
              << "\n";

  intptr_t subgrid_original = (intptr_t)subgrid; // just for sanity checking
  std::list<IntArray> *offsets = new std::list<IntArray>;
  std::list<IntArray> *offsets_new = new std::list<IntArray>;
  offsets->push_back(subgrid_offset);
  for (int i = num_dims - 1; i >= 1; --i) {
    FOREACH (oit, offsets->begin(), offsets->end()) {
      const IntArray &cur_offset = *oit;
      for (int j = 0; j < subgrid_size[i]; ++j) {
        IntArray new_offset = cur_offset;
        new_offset[i] += j;
        offsets_new->push_back(new_offset);
      }
    }
    std::swap(offsets, offsets_new);
    offsets_new->clear();    
  }

  size_t line_size = subgrid_size[0] * elm_size;
  FOREACH (oit, offsets->begin(), offsets->end()) {
    const IntArray &offset = *oit;
    void *src = (void *)((intptr_t)grid +
                         get1DOffset(offset, grid_size, num_dims) * elm_size);
    memcpy(src, subgrid, line_size);
    subgrid = (const void*)((intptr_t)subgrid + line_size);
  }

  delete offsets;
  delete offsets_new;

  // Sanity check
  void *p = (void *)(subgrid_original +
                     subgrid_size.accumulate(num_dims) * elm_size);
  assert(subgrid == p);

  return;
}

} // namespace runtime
} // namespace physis
