// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_REDUCE_GRID_MPI_CUDA_EXP_H_
#define PHYSIS_RUNTIME_REDUCE_GRID_MPI_CUDA_EXP_H_

#include "runtime/runtime_common.h"
#include "physis/reduce.h"

namespace physis {
namespace runtime {

extern int ReduceGridMPICUDAExp(void *buf, PSType type, PSReduceOp op,
                                void *dev_grid, int dim, const IndexArray &size,
                                const Width2 &width);

} //namespace runtime
} //namespace runtime

#endif // PHYSIS_RUNTIME_REDUCE_GRID_MPI_CUDA_EXP_H_ 
