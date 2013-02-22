// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/runtime_builder.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

SgFunctionCallExp *BuildTraceStencilPre(SgExpression *msg) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSTraceStencilPre");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(msg));
  return fc;
}

SgFunctionCallExp *BuildTraceStencilPost(SgExpression *time) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSTraceStencilPost");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(time));
  return fc;
}

SgVariableDeclaration *BuildStopwatch(const std::string &name,
                                      SgScopeStatement *scope,
                                      SgScopeStatement *global_scope) {
  SgType *t = si::lookupNamedTypeInParentScopes("__PSStopwatch",
                                                global_scope);
  assert(t);
  SgVariableDeclaration *decl
      = sb::buildVariableDeclaration(
          name, t,          
          NULL, scope);
  return decl;
}


SgFunctionCallExp *BuildStopwatchStart(SgExpression *sw) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSStopwatchStart");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(sw));
  return fc;
}

SgFunctionCallExp *BuildStopwatchStop(SgExpression *sw) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSStopwatchStop");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(sw));
  return fc;
}

SgFunctionCallExp *BuildDomainGetBoundary(SgExpression *dom,
                                          int dim, int right,
                                          SgExpression *width,
                                          int factor, int offset) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSDomainGetBoundary");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(
          fs,
          sb::buildExprListExp(dom, sb::buildIntVal(dim),
                               sb::buildIntVal(right), width,
                               sb::buildIntVal(factor), sb::buildIntVal(offset)));
  return fc;
}

SgExprListExp *RuntimeBuilder::BuildStencilWidth(const StencilRange &sr,
                                                 bool is_forward) {
  SgExprListExp *exp_list = sb::buildExprListExp();
  IntVector offset_min, offset_max;
  int nd = sr.num_dims();
  if (sr.IsEmpty()) {
    for (int i = 0; i < nd; ++i) {
      offset_min.push_back(0);
      offset_max.push_back(0);
    }
  } else if (!sr.GetNeighborAccess(offset_min, offset_max)) {
    LOG_DEBUG() << "Stencil access is not regular: "
                << sr << "\n";
    return NULL;
  }
  for (int i = 0; i < (int)offset_min.size(); ++i) {
    PSIndex v = is_forward ? offset_max[i] : -offset_min[i];
    if (v < 0) v = 0;
    si::appendExpression(exp_list, sb::buildIntVal(v));
  }
  return exp_list;
}

SgExprListExp *RuntimeBuilder::BuildStencilWidthFW(const StencilRange &sr) {
  return BuildStencilWidth(sr, true);
}

SgExprListExp *RuntimeBuilder::BuildStencilWidthBW(const StencilRange &sr) {
  return BuildStencilWidth(sr, false);
}


} // namespace translator
} // namespace physis
