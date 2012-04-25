// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

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

