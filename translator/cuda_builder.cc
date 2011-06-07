// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/cuda_builder.h"

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

SgFunctionCallExp *BuildCudaThreadSynchronize(void) {
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(
      "cudaThreadSynchronize", sb::buildVoidType());
  return fc;
}

SgFunctionCallExp *BuildCudaDim3(SgExpression *x, SgExpression *y,
                                 SgExpression *z) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("dim3");
  SgExprListExp *args = sb::buildExprListExp(x, y, z);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(fs, args);
  return call;
}

SgFunctionCallExp *BuildCudaStreamSynchronize(SgExpression *strm) {
  SgExprListExp *args = sb::buildExprListExp(strm);
  SgFunctionCallExp *call = sb::buildFunctionCallExp("cudaStreamSynchronize",
                                                     sb::buildVoidType(),
                                                     args);
  return call;
}  


} // namespace translator
} // namespace physis
