// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/builder_interface.h"

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



} // namespace translator
} // namespace physis
