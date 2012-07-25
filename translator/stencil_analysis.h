// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_STENCIL_ANALYSIS_H_
#define PHYSIS_TRANSLATOR_STENCIL_ANALYSIS_H_

#include "translator/translator_common.h"
#include "translator/map.h"

namespace physis {
namespace translator {

bool AnalyzeStencilIndex(SgExpression *arg, StencilIndex &idx,
                         SgFunctionDeclaration *kernel);
void AnalyzeStencilRange(StencilMap &sm, TranslationContext &tx);

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_STENCIL_ANALYSIS_H_ */
