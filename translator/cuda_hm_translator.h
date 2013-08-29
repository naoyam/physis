// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_TRANSLATOR_CUDA_HM_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_CUDA_HM_TRANSLATOR_H_

#include "translator/translator.h"
#include "translator/translator_common.h"
#include "translator/cuda_translator.h"
#include "translator/cuda_runtime_builder.h"

namespace physis {
namespace translator {

class CUDAHMTranslator : public CUDATranslator {
 public:
  CUDAHMTranslator(const Configuration &config);
  virtual ~CUDAHMTranslator() {}
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_CUDA_HM_TRANSLATOR_H_ */
