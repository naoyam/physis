// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/mpi_optimizer.h"
#include "translator/optimizer/optimization_passes.h"

namespace physis {
namespace translator {
namespace optimizer {

void MPIOptimizer::Stage1() {
}

void MPIOptimizer::Stage2() {
  if (config_->LookupFlag("OPT_KERNEL_INLINING")) {
    pass::kernel_inlining(proj_, tx_, builder_);
  }
}

} // namespace optimizer
} // namespace translator
} // namespace physis

