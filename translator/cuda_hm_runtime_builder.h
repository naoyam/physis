// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_TRANSLATOR_CUDA_HM_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_CUDA_HM_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/reference_runtime_builder.h"
#include "translator/cuda_runtime_builder.h"

namespace physis {
namespace translator {

class CUDAHMRuntimeBuilder : public CUDARuntimeBuilder {
 public:
  CUDAHMRuntimeBuilder(SgScopeStatement *global_scope,
                       const Configuration &config);
  virtual ~CUDAHMRuntimeBuilder() {}
};

} // namespace translator
} // namespace physis



#endif /* PHYSIS_TRANSLATOR_CUDA_HM_RUNTIME_BUILDER_H_ */
