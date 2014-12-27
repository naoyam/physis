// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/optimizer/cuda_optimizer.h"
#include "translator/optimizer/optimization_passes.h"

namespace physis {
namespace translator {
namespace optimizer {

void CUDAOptimizer::DoStage1() {
}

void CUDAOptimizer::DoStage2() {
  if (config_->LookupFlag("OPT_KERNEL_INLINING")) {
    pass::kernel_inlining(proj_, tx_, builder_);
  }
  if (config_->LookupFlag("OPT_LOOP_PEELING")) {
    pass::loop_peeling(proj_, tx_, builder_);
  }
  if (config_->LookupFlag("OPT_REGISTER_BLOCKING")) {
    pass::register_blocking(proj_, tx_, builder_);
  }
  if (config_->LookupFlag("OPT_UNCONDITIONAL_GET")) {
    pass::unconditional_get(proj_, tx_, builder_);
  }
  if (config_->LookupFlag("OPT_OFFSET_CSE")) {
    pass::offset_cse(proj_, tx_, builder_);
  }
  if (config_->LookupFlag("OPT_OFFSET_SPATIAL_CSE")) {
    pass::offset_spatial_cse(proj_, tx_, builder_);
  }
  if (config_->LookupFlag("OPT_LOOP_OPT")) {
    pass::loop_opt(proj_, tx_, builder_);
    pass::primitive_optimization(proj_, tx_, builder_);
  }
}

} // namespace optimizer
} // namespace translator
} // namespace physis

