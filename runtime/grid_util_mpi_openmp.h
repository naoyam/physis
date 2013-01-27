#ifndef PHYSIS_RUNTIME_GRID_UTIL_MPI_OPENMP_H_
#define PHYSIS_RUNTIME_GRID_UTIL_MPI_OPENMP_H_

#include "physis/physis_common.h"
#include "runtime/runtime_common.h"

namespace physis {
namespace runtime {
namespace mpiopenmputil {

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
                         );

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
                 );  

} // namespace mpiopenmputil
} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_GRID_UTIL_MPI_OPENMP_H_ */

