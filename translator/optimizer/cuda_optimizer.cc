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

void CUDAOptimizer::Stage1() {
}

void CUDAOptimizer::Stage2() {
  if (config_->LookupFlag("OPT_KERNEL_INLINING")) {
    pass::kernel_inlining(proj_, tx_);
  }
  if (config_->LookupFlag("OPT_UNCONDITIONAL_GET")) {
    pass::unconditional_get(proj_, tx_);
  }
}

} // namespace optimizer
} // namespace translator
} // namespace physis

