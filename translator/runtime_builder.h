// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_BUILDER_H_
#define PHYSIS_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/translation_context.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

SgFunctionCallExp *BuildTraceStencilPre(SgExpression *msg);
SgFunctionCallExp *BuildTraceStencilPost(SgExpression *time);

SgVariableDeclaration *BuildStopwatch(const std::string &name,
                                      SgScopeStatement *scope,
                                      SgScopeStatement *global_scope);
SgFunctionCallExp *BuildStopwatchStart(SgExpression *sw);
SgFunctionCallExp *BuildStopwatchStop(SgExpression *sw);

SgFunctionCallExp *BuildDomainGetBoundary(SgExpression *dom,
                                          int dim, int right,
                                          SgExpression *width,
                                          int factor, int offset);

class RuntimeBuilder {
 public:
  RuntimeBuilder(SgScopeStatement *global_scope):
      gs_(global_scope) {}
  //!
  /*!
    \param 
    \return
   */
  virtual SgFunctionCallExp *BuildGridDim(
      SgExpression *grid_ref,
      int dim) = 0;
  //!
  /*!
    \param
    \return
   */
  virtual SgExpression *BuildGridRefInRunKernel(
      SgInitializedName *gv,
      SgFunctionDeclaration *run_kernel) = 0;

  virtual SgExpression *BuildOffset(
      SgInitializedName *gv, int num_dim,
      SgExprListExp *offset_exprs, bool is_kernel,
      bool is_periodic, SgScopeStatement *scope) = 0;
  
  virtual SgExpression *BuildGet(  
    SgInitializedName *gv,
    SgExprListExp *offset_exprs,
    SgScopeStatement *scope,
    TranslationContext *tx, bool is_kernel,

  
  virtual SgType *GetIndexType() {
    return sb::buildOpaqueType(PS_INDEX_TYPE_NAME, gs_);
  }
      
 protected:
  SgScopeStatement *gs_;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_RUNTIME_BUILDER_H_ */
