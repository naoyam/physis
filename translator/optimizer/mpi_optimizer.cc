// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/optimizer/mpi_optimizer.h"
#include "translator/optimizer/optimization_passes.h"

namespace physis {
namespace translator {
namespace optimizer {

void MPIOptimizer::DoStage1() {
}

void MPIOptimizer::DoStage2() {
  if (config_->LookupFlag("OPT_KERNEL_INLINING")) {
    pass::kernel_inlining(proj_, tx_, builder_);
  }
}

} // namespace optimizer
} // namespace translator
} // namespace physis

