#include "runtime/grid_mpi_openmp.h"
#include "runtime/grid_util_mpi_openmp.h"

namespace physis {
namespace runtime {

void GridMPIOpenMP::CopyinoutSubgrid(
    bool copyout_to_buf_p,
    size_t elm_size, int num_dims,
    BufferHostOpenMP *bufmp,
    const IntArray &grid_size,
    void *buf,
    const IntArray &subgrid_offset,
    const IntArray &subgrid_size
){

    IntArray subgrid_offset_inner(0,0,0);
    IntArray subgrid_size_inner(1,1,1);

    subgrid_offset_inner.SetNoLessThan(subgrid_offset);
    subgrid_size_inner.SetNoLessThan(subgrid_size);

    // Only support the following case!!
    PSAssert(elm_size == bufmp->elm_size());
    PSAssert(grid_size == bufmp->size());

    IntArray subbuf_division(1,1,1);
    size_t **subbuf_offset = new size_t*[PS_MAX_DIM];
    size_t **subbuf_width = new size_t*[PS_MAX_DIM];
    for (unsigned int dim = 0; dim < 3; dim++) {
      subbuf_offset[dim] = new size_t[1];
      subbuf_width[dim] = new size_t[1];
      subbuf_offset[dim][0] = 0;
      subbuf_width[dim][0] = subgrid_size_inner[dim];
    }

    mpiopenmputil::CopyinoutSubgrid_MP(
      copyout_to_buf_p,
      elm_size, 3, // Always 3 here
      bufmp->Get_MP(),
      bufmp->size(),
      bufmp->MPdivision(),
      bufmp->MPoffset(),
      bufmp->MPwidth(),
      (void **) &buf,
      subgrid_offset_inner,
      subgrid_size_inner,
      subbuf_division,
      subbuf_offset,
      subbuf_width
    );

    for (unsigned int dim = 0; dim < PS_MAX_DIM; dim++) {
      delete[] subbuf_offset[dim];
      delete[] subbuf_width[dim];
    }
    delete[] subbuf_offset;
    delete[] subbuf_width;

} // GridMPIOpenMP::CopyinoutSubgrid

void GridMPIOpenMP::CopyinoutSubgrid(
    bool copyIN_FROM_BUF_p,
    size_t elm_size, int num_dims,
    void *global_buf,
    const IntArray &global_size,
    BufferHostOpenMP *subbufmp,
    const IntArray &subgrid_offset,
    const IntArray &subgrid_size
){

    IntArray subgrid_offset_inner(0,0,0);
    IntArray global_size_inner(1,1,1);

    subgrid_offset_inner.SetNoLessThan(subgrid_offset);
    global_size_inner.SetNoLessThan(global_size);

    // Only support the following case!!
    PSAssert(elm_size == subbufmp->elm_size());
    PSAssert(subgrid_size <= subbufmp->size());

    IntArray buf_division(1,1,1);
    size_t **buf_offset = new size_t*[PS_MAX_DIM];
    size_t **buf_width = new size_t*[PS_MAX_DIM];
    for (unsigned int dim = 0; dim < 3; dim++) {
      buf_offset[dim] = new size_t[1];
      buf_width[dim] = new size_t[1];
      buf_offset[dim][0] = 0;
      buf_width[dim][0] =  global_size[dim];
    }

    mpiopenmputil::CopyinoutSubgrid_MP(
      copyIN_FROM_BUF_p,
      elm_size, 3, // Always 3 here
      (void **) &global_buf,
      global_size,
      buf_division,
      buf_offset,
      buf_width,
      subbufmp->Get_MP(),
      subgrid_offset,
      subgrid_size,
      subbufmp->MPdivision(),
      subbufmp->MPoffset(),
      subbufmp->MPwidth()
    );

    for (unsigned int dim = 0; dim < PS_MAX_DIM; dim++) {
      delete[] buf_offset[dim];
      delete[] buf_width[dim];
    }
    delete[] buf_offset;
    delete[] buf_width;

} // GridMPIOpenMP::CopyinoutSubgrid

} // namespace runtime
} // namespace physis
