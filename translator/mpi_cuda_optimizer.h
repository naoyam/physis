// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_MPI_CUDA_OPTIMIZER_H_
#define PHYSIS_TRANSLATOR_MPI_CUDA_OPTIMIZER_H_

#include "translator/translator_common.h"
#include "translator/mpi_cuda_translator.h"

namespace physis {
namespace translator {

class MPICUDAOptimizer {
 public:
  MPICUDAOptimizer(MPICUDATranslator &trans);
  virtual ~MPICUDAOptimizer() {}
  virtual void GridPreCalcAddr(SgFunctionDeclaration *func);
 protected:
  MPICUDATranslator &trans_;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_OPTIMIZER_H_ */
