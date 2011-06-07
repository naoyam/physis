// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/mpi_cuda_runtime_builder.h"

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

SgFunctionCallExp *BuildGridGetDev(SgExpression *grid_var) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGridGetDev");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(grid_var));
  return fc;
}

SgFunctionCallExp *BuildGetLocalSize(SgExpression *dim) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGetLocalSize");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(dim));
  return fc;
}  
SgFunctionCallExp *BuildGetLocalOffset(SgExpression *dim) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGetLocalOffset");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(dim));
  return fc;
}

SgFunctionCallExp *BuildDomainShrink(SgExpression *dom,
                                     SgExpression *width) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSDomainShrink");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(
          fs, sb::buildExprListExp(dom, width));
  return fc;
}

SgExpression *BuildStreamBoundaryKernel(int idx) {
  SgVarRefExp *inner_stream = sb::buildVarRefExp("stream_boundary_kernel");
  return sb::buildPntrArrRefExp(inner_stream, sb::buildIntVal(idx));
}
  
} // namespace translator
} // namespace physis
