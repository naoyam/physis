// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_OPTIMIZER_OPTIMIZER_H_
#define PHYSIS_TRANSLATOR_OPTIMIZER_OPTIMIZER_H_

#include "translator/translator_common.h"
#include "translator/translation_context.h"
#include "translator/configuration.h"
#include "translator/runtime_builder.h"

namespace physis {
namespace translator {
namespace optimizer {

class Optimizer {
 public:
  Optimizer(SgProject *proj,
            physis::translator::TranslationContext *tx,
            physis::translator::RuntimeBuilder *builder,
            physis::translator::Configuration *config)
      : proj_(proj), tx_(tx), builder_(builder), config_(config) {}
  virtual ~Optimizer() {}
  //! Pre-translation optimizations
  virtual void Stage1();
  //! Post-translation optimizations
  virtual void Stage2();
 protected:
  SgProject *proj_;  
  physis::translator::TranslationContext *tx_;
  physis::translator::RuntimeBuilder *builder_;
  physis::translator::Configuration *config_;
};

} // namespace optimizer
} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_OPTIMIZER_OPTIMIZER_H_ */
