// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/reference_optimizer.h"
#include "translator/optimizer/optimization_passes.h"

namespace physis {
namespace translator {
namespace optimizer {

void ReferenceOptimizer::Stage1() {
}

void ReferenceOptimizer::Stage2() {
  if (config_->LookupFlag("OPT_KERNEL_INLINING")) {
    pass::kernel_inlining(proj_, tx_);
  }
#if 0  
  if (config_->LookupFlag("OPT_UNCONDITIONAL_GET")) {
    pass::unconditional_get(proj_, tx_);
  }
#endif
}

} // namespace optimizer
} // namespace translator
} // namespace physis

