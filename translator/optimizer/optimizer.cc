// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/optimizer.h"
#include "translator/optimizer/optimization_passes.h"

namespace physis {
namespace translator {
namespace optimizer {

void Optimizer::Stage1() {
  pass::null_optimization(proj_, tx_, builder_);
}

void Optimizer::Stage2() {
  pass::null_optimization(proj_, tx_, builder_);
}

} // namespace optimizer
} // namespace translator
} // namespace physis

