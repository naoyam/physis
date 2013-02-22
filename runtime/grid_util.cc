// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/grid_util.h"

using namespace physis::runtime;
using physis::IntArray;
using physis::IndexArray;

namespace physis {
namespace runtime {

static size_t get1DOffset(const IndexArray &md_offset,
                          const IndexArray &size,
                          int num_dims) {
  size_t offset_1d = 0;
  size_t ref_offset = 1;
  for (int i = 0; i < num_dims; ++i) {
    offset_1d += (md_offset[i] * ref_offset);
    ref_offset *= size[i];
  }
  return offset_1d;
}

static void CopySubgrid(void *buf, const void *src,
                        size_t elm_size, int num_dims,
                        const IndexArray &grid_size,
                        const IndexArray &subgrid_offset,
                        const IndexArray &subgrid_size,
                        bool is_copyin) {
  // just for sanity checking
  intptr_t subgrid_original = (intptr_t) (is_copyin ? src : buf);
  std::list<IndexArray> *offsets = new std::list<IndexArray>;
  std::list<IndexArray> *offsets_new = new std::list<IndexArray>;
  
  LOG_DEBUG() << __FUNCTION__ << ": "
              << "subgrid offset: " << subgrid_offset
              << "subgrid size: " << subgrid_size
              << "grid size: " << grid_size
              << "\n";

  // Collect 1-D continuous regions
  offsets->push_back(subgrid_offset);
  for (int i = num_dims - 1; i >= 1; --i) {
    FOREACH (oit, offsets->begin(), offsets->end()) {
      const IndexArray &cur_offset = *oit;
      for (PSIndex j = 0; j < subgrid_size[i]; ++j) {
        IndexArray new_offset = cur_offset;
        new_offset[i] += j;
        // support periodic access
        new_offset[i] = (new_offset[i] + grid_size[i]) %  grid_size[i];
        offsets_new->push_back(new_offset);
      }
    }
    std::swap(offsets, offsets_new);
    offsets_new->clear();
  }

  // Copy the collected 1-D continuous regions. The region is allowed
  // to periodically access off grid boundaries
  FOREACH (oit, offsets->begin(), offsets->end()) {
    IndexArray &offset = *oit;
    PSIndex next_offset;
    PSIndex initial_size;
    IndexArray ss = subgrid_size;
    bool done = false;
    while (!done) {
      initial_size = subgrid_size[0];
      if (offset[0] < 0) {
        // Copy sub region with backward periodic access
        offset[0] = (offset[0] + grid_size[0]) % grid_size[0];
        ss[0] = grid_size[0] - offset[0];
        next_offset = 0;
      } else if (offset[0] + ss[0] > grid_size[0]) {
        // Copy data only within the continuous region and leave the
        // remaining, which then results in periodic access
        ss[0] = grid_size[0] - offset[0];
        next_offset = 0;
      } else {
        done = true;
      }
      size_t grid_1d_offset =
          get1DOffset(offset, grid_size, num_dims) * elm_size;
      const void *src_ptr;
      void *dst_ptr;
      if (is_copyin) {
        src_ptr = src;
        dst_ptr = (void *)((intptr_t)buf + grid_1d_offset);
      } else {
        src_ptr = (const void *)((intptr_t)src + grid_1d_offset);
        dst_ptr = buf;
      }
      size_t line_size = ss[0] * elm_size;
      memcpy(dst_ptr, src_ptr, line_size);
      if (is_copyin) {
        src = (const void*)((intptr_t)src + line_size);
      } else {
        buf = (void*)((intptr_t)buf + line_size);
      }
      
      offset[0] = next_offset;
      ss[0] = initial_size - ss[0];
    }
  }

  delete offsets;
  delete offsets_new;

  // sanity checking
  void *p = (void *)(subgrid_original +
                     subgrid_size.accumulate(num_dims) * elm_size);
  if (is_copyin) {
    assert(src == p);
  } else {
    assert(buf == p);
  }

  return;
}

void CopyoutSubgrid(size_t elm_size, int num_dims,
                    const void *grid, const IndexArray &grid_size,
                    void *subgrid,
                    const IndexArray &subgrid_offset,
                    const IndexArray &subgrid_size) {
#if 0  
  intptr_t subgrid_original = (intptr_t)subgrid; // just for sanity checking
  std::list<IndexArray> *offsets = new std::list<IndexArray>;
  std::list<IndexArray> *offsets_new = new std::list<IndexArray>;
  
  LOG_DEBUG() << __FUNCTION__ << ": "
              << "subgrid offset: " << subgrid_offset
              << "subgrid size: " << subgrid_size
              << "grid size: " << grid_size
              << "\n";

  // Collect 1-D continuous regions
  offsets->push_back(subgrid_offset);
  for (int i = num_dims - 1; i >= 1; --i) {
    FOREACH (oit, offsets->begin(), offsets->end()) {
      const IndexArray &cur_offset = *oit;
      for (PSIndex j = 0; j < subgrid_size[i]; ++j) {
        IndexArray new_offset = cur_offset;
        new_offset[i] += j;
        // support periodic access
        new_offset[i] = (new_offset[i] + grid_size[i]) %  grid_size[i];
        offsets_new->push_back(new_offset);
      }
    }
    std::swap(offsets, offsets_new);
    offsets_new->clear();
  }

  // Copy the collected 1-D continuous regions. The region is allowed
  // to periodically access off grid boundaries
  FOREACH (oit, offsets->begin(), offsets->end()) {
    IndexArray &offset = *oit;
    PSIndex next_offset;
    PSIndex initial_size;
    IndexArray ss = subgrid_size;
    bool done = false;
    while (done) {
      initial_size = subgrid_size[0];
      if (offset[0] < 0) {
        // Copy sub region with backward periodic access
        offset[0] = (offset[0] + grid_size[0]) % grid_size[0];
        ss[0] = grid_size[0] - offset[0];
        next_offset = 0;
      } else if (offset[0] + ss[0] > grid_size[0]) {
        // Copy data only within the continuous region and leave the
        // remaining, which then results in periodic access
        ss[0] = grid_size[0] - offset[0];
        next_offset = 0;
      } else {
        done = true;
      }
      const void *src =
          (const void *)((intptr_t)grid +
                         get1DOffset(offset, grid_size, num_dims) * elm_size);
      size_t line_size = ss[0] * elm_size;
      memcpy(subgrid, src, line_size);
      subgrid = (void*)((intptr_t)subgrid + line_size);
      
      offset[0] = next_offset;
      ss[0] = initial_size - ss[0];
    }
  }

  delete offsets;
  delete offsets_new;

  void *p = (void *)(subgrid_original +
                     subgrid_size.accumulate(num_dims) * elm_size);
  assert(subgrid == p);
#else
  CopySubgrid(subgrid, grid, elm_size, num_dims, grid_size,
              subgrid_offset, subgrid_size, false);
#endif
  return;
}

void CopyinSubgrid(size_t elm_size, int num_dims,
                   void *grid, const IndexArray &grid_size,
                   const void *subgrid,
                   const IndexArray &subgrid_offset,
                   const IndexArray &subgrid_size) {
#if 0  
  LOG_DEBUG() << __FUNCTION__ << ": "
              << "subgrid offset: " << subgrid_offset
              << "subgrid size: " << subgrid_size
              << "grid size: " << grid_size
              << "\n";

  intptr_t subgrid_original = (intptr_t)subgrid; // just for sanity checking
  std::list<IndexArray> *offsets = new std::list<IndexArray>;
  std::list<IndexArray> *offsets_new = new std::list<IndexArray>;
  offsets->push_back(subgrid_offset);
  for (int i = num_dims - 1; i >= 1; --i) {
    FOREACH (oit, offsets->begin(), offsets->end()) {
      const IndexArray &cur_offset = *oit;
      for (PSIndex j = 0; j < subgrid_size[i]; ++j) {
        IndexArray new_offset = cur_offset;
        new_offset[i] += j;
        offsets_new->push_back(new_offset);
      }
    }
    std::swap(offsets, offsets_new);
    offsets_new->clear();    
  }

  size_t line_size = subgrid_size[0] * elm_size;
  FOREACH (oit, offsets->begin(), offsets->end()) {
    const IndexArray &offset = *oit;
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
#else
  CopySubgrid(grid, subgrid, elm_size, num_dims,
              grid_size, subgrid_offset, subgrid_size,
              true);
#endif
  return;
}

} // namespace runtime
} // namespace physis
