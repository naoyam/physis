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

SgType *BuildCudaErrorType() {
  return si::lookupNamedTypeInParentScopes("cudaError_t");
}

SgFunctionCallExp *BuildCudaMalloc(SgExpression *buf, SgExpression *size) {
  SgExprListExp *args = sb::buildExprListExp(
      sb::buildCastExp(sb::buildAddressOfOp(buf),
                       sb::buildPointerType(
                           sb::buildPointerType(sb::buildVoidType()))),
      size);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaMalloc",
      BuildCudaErrorType(),
      args);
  return call;
}

SgFunctionCallExp *BuildCudaFree(SgExpression *p) {
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaFree",
      BuildCudaErrorType(),
      sb::buildExprListExp(p));
  return call;
}

SgFunctionCallExp *BuildCudaMallocHost(SgExpression *buf, SgExpression *size) {
  SgExprListExp *args = sb::buildExprListExp(
      sb::buildCastExp(sb::buildAddressOfOp(buf),
                       sb::buildPointerType(
                           sb::buildPointerType(sb::buildVoidType()))),
      size);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaMallocHost",
      BuildCudaErrorType(),
      args);
  return call;
}

SgFunctionCallExp *BuildCudaFreeHost(SgExpression *p) {
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaFreeHost",
      BuildCudaErrorType(),
      sb::buildExprListExp(p));
  return call;
}

SgFunctionCallExp *BuildCudaMemcpyHostToDevice(
    SgExpression *dst, SgExpression *src, SgExpression *size) {
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaMemcpy",
      BuildCudaErrorType(),
      sb::buildExprListExp(
          dst, src, size,
          sb::buildOpaqueVarRefExp("cudaMemcpyHostToDevice")));
  return call;
}
SgFunctionCallExp *BuildCudaMemcpyDeviceToHost(
    SgExpression *dst, SgExpression *src, SgExpression *size) {
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaMemcpy",
      BuildCudaErrorType(),
      sb::buildExprListExp(
          dst, src, size,
          sb::buildOpaqueVarRefExp("cudaMemcpyDeviceToHost")));
  return call;
}

} // namespace translator
} // namespace physis
