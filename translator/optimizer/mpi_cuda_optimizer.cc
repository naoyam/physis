// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/optimizer/mpi_cuda_optimizer.h"
#include "translator/optimizer/optimization_passes.h"

namespace physis {
namespace translator {
namespace optimizer {

void MPICUDAOptimizer::DoStage1() {
}

void MPICUDAOptimizer::DoStage2() {
  // TODO: support this optimization
#if 0  
  if (config_->LookupFlag("OPT_UNCONDITIONAL_GET")) {
    pass::unconditional_get(proj_, tx_, builder_);
  }
#endif  
}

} // namespace optimizer
} // namespace translator
} // namespace physis

