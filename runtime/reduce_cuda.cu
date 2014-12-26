// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/runtime_common.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/cuda_util.h"
#include "runtime/reduce.h"
#include "physis/physis_cuda.h"
#include "runtime/reduce_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

  void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                           __PSGrid *g) {
    // Note: Assuming primitive-type grids. p0 is only valid for
    // primitive types, and not valid for user-defined types.
    physis::runtime::ReduceGridCUDA<float>(buf, op, g->p, g->num_elms);
  }

  void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                            __PSGrid *g) {
    // Note: Assuming primitive-type grids. p0 is only valid for
    // primitive types, and not valid for user-defined types.
    physis::runtime::ReduceGridCUDA<double>(buf, op, g->p, g->num_elms);
  }

  void __PSReduceGridInt(void *buf, enum PSReduceOp op,
                         __PSGrid *g) {
    // Note: Assuming primitive-type grids. p0 is only valid for
    // primitive types, and not valid for user-defined types.
    physis::runtime::ReduceGridCUDA<int>(buf, op, g->p, g->num_elms);
  }

  void __PSReduceGridLong(void *buf, enum PSReduceOp op,
                            __PSGrid *g) {
    // Note: Assuming primitive-type grids. p0 is only valid for
    // primitive types, and not valid for user-defined types.
    physis::runtime::ReduceGridCUDA<long>(buf, op, g->p, g->num_elms);
  }
  
#ifdef __cplusplus
}
#endif
