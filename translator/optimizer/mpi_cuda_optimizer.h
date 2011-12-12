// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_OPTIMIZER_REFERENCE_MPI_CUDA_OPTIMIZER_H_
#define PHYSIS_TRANSLATOR_OPTIMIZER_REFERENCE_MPI_CUDA_OPTIMIZER_H_

#include "translator/optimizer/optimizer.h"

namespace physis {
namespace translator {
namespace optimizer {

class MPICUDAOptimizer: public Optimizer {
 public:
  MPICUDAOptimizer(SgProject *proj,
                   physis::translator::TranslationContext *tx,
                   physis::translator::RuntimeBuilder *builder,
                   physis::translator::Configuration *config)
      : Optimizer(proj, tx, builder, config) {}
  virtual ~MPICUDAOptimizer() {}
  virtual void Stage1();
  virtual void Stage2();
};

} // namespace optimizer
} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_OPTIMIZER_MPI_CUDA_OPTIMIZER_H_ */
