// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/cuda_optimizer.h"
#include "translator/optimizer/optimization_passes.h"

namespace physis {
namespace translator {
namespace optimizer {

void CUDAOptimizer::DoStage1() {
}

void CUDAOptimizer::DoStage2() {
  if (config_->LookupFlag("OPT_KERNEL_INLINING") ||
      config_->LookupFlag("OPT_LOOP_PEELING") ||
      config_->LookupFlag("OPT_REGISTER_BLOCKING") ||
      config_->LookupFlag("OPT_OFFSET_CSE")) {
    pass::kernel_inlining(proj_, tx_, builder_);
  }
  if (config_->LookupFlag("OPT_LOOP_PEELING") ||
      config_->LookupFlag("OPT_REGISTER_BLOCKING")) {
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
}

} // namespace optimizer
} // namespace translator
} // namespace physis

