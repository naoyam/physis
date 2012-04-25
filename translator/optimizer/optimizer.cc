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

void Optimizer::DoStage1() {
  pass::null_optimization(proj_, tx_, builder_);
}

void Optimizer::DoStage2() {
  pass::null_optimization(proj_, tx_, builder_);
}
void Optimizer::Stage1() {
  PreProcess();
  LOG_DEBUG() << "Applying Stage 1 optimization passes\n";  
  DoStage1();
  LOG_DEBUG() << "Stage 1 optimization done\n";  
  PostProcess();
}

void Optimizer::Stage2() {
  PreProcess();
  LOG_DEBUG() << "Applying Stage 2 optimization passes\n";
  DoStage2();
  LOG_DEBUG() << "Stage 2 optimization done\n";
  PostProcess();  
}

void Optimizer::PreProcess() {
}

void Optimizer::PostProcess() {
}


} // namespace optimizer
} // namespace translator
} // namespace physis

