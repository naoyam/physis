// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"

namespace physis {
namespace translator {

SgFunctionCallExp *BuildGridGetDev(SgExpression *grid_var);
SgFunctionCallExp *BuildGetLocalSize(SgExpression *dim);
SgFunctionCallExp *BuildGetLocalOffset(SgExpression *dim);
SgFunctionCallExp *BuildDomainShrink(SgExpression *dom,
                                     SgExpression *width);
SgExpression *BuildStreamBoundaryKernel(int idx);
} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_ */

