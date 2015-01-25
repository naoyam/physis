// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/grid_util_mpi_openmp.h"

using namespace physis::runtime;
//using namespace physis::util;

namespace physis {
namespace runtime {
namespace mpiopenmputil {

void getMPOffset(
    const unsigned int num_dims,
    const IntArray &offset,
    const IntArray &grid_size,
    const IntArray &grid_division,
    const size_t * const *grid_mp_offset,
    const size_t * const *grid_mp_width,
    unsigned int &cpuid_OUT,
    size_t &gridid_OUT,
    size_t &width_avail_OUT
                 ){

  unsigned long cpuid[PS_MAX_DIM] = {0};
  IntArray local_offset(0,0,0);

  cpuid_OUT = 0;
  gridid_OUT = 0;
  size_t cpuid_factor = 1;
  size_t gridid_factor = 1;

  for (unsigned int dim = 0; dim < num_dims; dim++) {
    unsigned int divid = 0;
    for (divid = 0; divid + 1 < (unsigned)grid_division[dim]; divid++) {
#if 0
      LOG_DEBUG() << "dim, divid,  offset[dim], grid_mp_offset[dim][divid]" <<
          dim << " " << divid << " " << offset[dim] << " " <<
          grid_mp_offset[dim][divid] << ""
                  << "\n";
#endif
      if ((size_t) offset[dim] < grid_mp_offset[dim][divid + 1]) break;
    }
    cpuid[dim] = divid;
    local_offset[dim] = offset[dim] - grid_mp_offset[dim][divid];
    // PSAssert((size_t) local_offset[dim] <= grid_mp_width[dim][divid]);

    cpuid_OUT += cpuid[dim] * cpuid_factor;
    cpuid_factor *= grid_division[dim];

    gridid_OUT += local_offset[dim] * gridid_factor;
    gridid_factor *= grid_mp_width[dim][cpuid[dim]];

  }
  width_avail_OUT = grid_mp_width[0][cpuid[0]] - local_offset[0];
}

void CopyinoutSubgrid_MP(
    bool copyout_to_subgrid_mode_p,
    size_t elm_size, int num_dims,
    void **grid_mp,
    const IntArray  &grid_size,
    const IntArray &grid_division,
    const size_t * const *grid_mp_offset,
    const size_t * const *grid_mp_width,
    void **subgrid_mp,
    const IntArray &subgrid_offset,
    const IntArray &subgrid_size,
    const IntArray &subgrid_division,
    const size_t * const *subgrid_mp_offset,
    const size_t * const *subgrid_mp_width
                         )
{
#if 0
  intptr_t subgrid_original = (intptr_t)subgrid; // just for sanity checking
#endif
  std::list<IntArray> *offsets = new std::list<IntArray>;
  std::list<IntArray> *offsets_new = new std::list<IntArray>;
  
  LOG_DEBUG() << __FUNCTION__ << ":"
              << " copyout to subgrid: " << copyout_to_subgrid_mode_p
              << " subgrid offset: " << subgrid_offset
              << " subgrid size: " << subgrid_size
              << " subgrid division: " << subgrid_division
              << "\n";
  LOG_DEBUG() << __FUNCTION__ << ":"
              << " grid size: " << grid_size
              << " grid division: " << grid_division
              << "\n";


  void *tmpbuf = 0;

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

  tmpbuf = calloc(1, line_size);
  if (!tmpbuf) {
    LOG_ERROR() << "calloc failed.\n";
    PSAbort(1);
  }

  IntArray subgrid_pos(0,0,0);

  FOREACH (oit, offsets->begin(), offsets->end()) {
    IntArray &offset = *oit;
    size_t size_left = 0;
    unsigned int cpuid = 0;
    size_t gridid = 0;
    size_t width_avail = 0;

    if (copyout_to_subgrid_mode_p) {
      do {
        // First copy grid to tmpbuf
        size_left = subgrid_size[0];
        size_t size_writenow = 0;

        intptr_t tmpbuf_pos = (intptr_t) tmpbuf;

        while (size_left) {
#if 0
          LOG_DEBUG() << "offset[0], size_left, grid_size[0] " <<
              offset[0] << " " << size_left << " " << grid_size[0] <<
              "\n";
#endif
          PSAssert(offset[0] < grid_size[0]);
          getMPOffset(
              num_dims,
              offset, grid_size, grid_division,
              grid_mp_offset, grid_mp_width,
              cpuid, gridid, width_avail
                      );
          size_writenow = width_avail;
          if (size_writenow > size_left)
            size_writenow = size_left;

          intptr_t srcgridpos = 
              (intptr_t) grid_mp[cpuid] + gridid * elm_size;

          memcpy(
              (void *)tmpbuf_pos, (void *) srcgridpos,
              size_writenow * elm_size);

          tmpbuf_pos += size_writenow * elm_size;
          offset[0] += size_writenow;
          size_left -= size_writenow;
        } // while (size_left)

      } while(0);

      do {
        // Then copy tmpbuf to subgrid
        size_left = subgrid_size[0];
        size_t size_writenow = 0;

        intptr_t tmpbuf_pos = (intptr_t) tmpbuf;

        while (size_left) {
          PSAssert(subgrid_pos[0] < subgrid_size[0]);
          getMPOffset(
              num_dims,
              subgrid_pos, subgrid_size, subgrid_division,
              subgrid_mp_offset, subgrid_mp_width,
              cpuid, gridid, width_avail
                      );
          size_writenow = width_avail;
          if (size_writenow > size_left)
            size_writenow = size_left;

          intptr_t dstgridpos = 
              (intptr_t) subgrid_mp[cpuid] + gridid * elm_size;
#if 0
          LOG_DEBUG() << "subgrid_pos, subgrid_size, cpuid, gridid" <<
              subgrid_pos << " " << subgrid_size << " " <<
              cpuid << " " << gridid <<
              "\n";
#endif
#if 1
          memcpy(
              (void *) dstgridpos, (void *)tmpbuf_pos, 
              size_writenow * elm_size);
#endif

          tmpbuf_pos += size_writenow * elm_size;
          subgrid_pos[0] += size_writenow;
          size_left -= size_writenow;
        } // while (size_left)

      } while(0);

    } else { // if (copyout_to_subgrid_mode_p)
      do {
        // First copy subgrid to tmpbuf
        size_left = subgrid_size[0];
        size_t size_writenow = 0;

        intptr_t tmpbuf_pos = (intptr_t) tmpbuf;

        while (size_left) {
          PSAssert(subgrid_pos[0] < subgrid_size[0]);
          getMPOffset(
              num_dims,
              subgrid_pos, subgrid_size, subgrid_division,
              subgrid_mp_offset, subgrid_mp_width,
              cpuid, gridid, width_avail
                      );
          size_writenow = width_avail;
          if (size_writenow > size_left)
            size_writenow = size_left;

          intptr_t srcgridpos = 
              (intptr_t) subgrid_mp[cpuid] + gridid * elm_size;
          memcpy(
              (void *)tmpbuf_pos, (void *) srcgridpos,
              size_writenow * elm_size);

          tmpbuf_pos += size_writenow * elm_size;
          subgrid_pos[0] += size_writenow;
          size_left -= size_writenow;
        } // while (size_left)

      } while(0);

      do {
        // Then copy tmpbuf to grid
        size_left = subgrid_size[0];
        size_t size_writenow = 0;

        intptr_t tmpbuf_pos = (intptr_t) tmpbuf;

        while (size_left) {
          PSAssert(offset[0] < grid_size[0]);
          getMPOffset(
              num_dims,
              offset, grid_size, grid_division,
              grid_mp_offset, grid_mp_width,
              cpuid, gridid, width_avail
                      );
          size_writenow = width_avail;
          if (size_writenow > size_left)
            size_writenow = size_left;

          intptr_t dstgridpos = 
              (intptr_t) grid_mp[cpuid] + gridid * elm_size;
          memcpy(
              (void *) dstgridpos, (void *)tmpbuf_pos, 
              size_writenow * elm_size);

          tmpbuf_pos += size_writenow * elm_size;
          offset[0] += size_writenow;
          size_left -= size_writenow;
        } // while (size_left)

      } while(0);

    } //  if (!copyout_to_subgrid_mode_p)


    // Move subgrid_pos;
    subgrid_pos[0] = 0;
    subgrid_pos[1] ++;
    if (subgrid_pos[1] >= subgrid_size[1]) {
      subgrid_pos[1] = 0;
      subgrid_pos[2] ++;
    }

  } // FOREACH (oit, offsets->begin(), offsets->end())

  delete offsets;
  delete offsets_new;

  free(tmpbuf);

#if 0
  void *p = (void *)(subgrid_original +
                     subgrid_size.accumulate(num_dims) * elm_size);
  assert(subgrid == p);
#endif

  return;
}


} // namespace mpiopenmputil
} // namespace runtime
} // namespace physis
