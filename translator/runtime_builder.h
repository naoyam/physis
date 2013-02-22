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

  //!
  /*!
    \param offset_exprs Free AST node of offset expressions
    
    Parameter offset_exprs will be used in the returned offset
    expression without cloning.
   */
  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
      SgExpressionPtrList *offset_exprs, bool is_kernel,
      bool is_periodic,
      const StencilIndexList *sil) = 0;
  //!
#if 0  
  /*!
    \param offset_exprs Free AST node of offset expressions
    
    Parameter offset_exprs will be used in the returned offset
    expression without cloning.
   */
  virtual SgExpression *BuildGet(  
    SgInitializedName *gv,
    SgExprListExp *offset_exprs,
    SgScopeStatement *scope,
    TranslationContext *tx, bool is_kernel,
    bool is_periodic) = 0;
#endif

  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,      
      GridType *gt,
      SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic) = 0;

  virtual SgBasicBlock *BuildGridSet(
      SgExpression *grid_var, int num_dims,
      const SgExpressionPtrList &indices, SgExpression *val) = 0;

  virtual SgFunctionCallExp *BuildGridGetID(SgExpression *grid_var) = 0;
  
  virtual SgType *GetIndexType() {
    return sb::buildOpaqueType(PS_INDEX_TYPE_NAME, gs_);
  }

  virtual SgExprListExp *BuildStencilWidthFW(const StencilRange &sr);
  virtual SgExprListExp *BuildStencilWidthBW(const StencilRange &sr);
      
 protected:
  SgScopeStatement *gs_;
  
  virtual SgExprListExp *BuildStencilWidth(const StencilRange &sr,
                                           bool is_forward);
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_RUNTIME_BUILDER_H_ */
