// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_OPTIMIZER_REFERENCE_OPTIMIZER_H_
#define PHYSIS_TRANSLATOR_OPTIMIZER_REFERENCE_OPTIMIZER_H_

#include "translator/optimizer/optimizer.h"

namespace physis {
namespace translator {
namespace optimizer {

class ReferenceOptimizer: public Optimizer {
 public:
  ReferenceOptimizer(SgProject *proj,
                     physis::translator::TranslationContext *tx,
                     physis::translator::RuntimeBuilder *builder,
                     physis::translator::Configuration *config)
      : Optimizer(proj, tx, builder, config) {}
  virtual ~ReferenceOptimizer() {}
 protected:
  virtual void DoStage1();
  virtual void DoStage2();
};

} // namespace optimizer
} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_OPTIMIZER_REFERENCE_OPTIMIZER_H_ */
