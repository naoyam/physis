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

  //! Build an offset expression.
  /*!
    Parameter offset_exprs will be used in the returned offset 
    expression without cloning.

    @param num_dim Number of dimensions.
    @param offset_exprs Index argument list.
    @param is_kernel True if the expression is used in a stencil
    kernel. 
    @param is_periodic True if it is a periodic access.
    @param sil The stencil index list of this access.
   */
  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
      const SgExpressionPtrList *offset_exprs, bool is_kernel,
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
      GridVarAttribute *gva,                  
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic) = 0;
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,                  
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name) = 0;
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,                  
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name,
      const SgExpressionVector &array_indices) = 0;

  //! Build code for grid emit.
  /*!
    \param grid_exp Grid expression
    \param attr GridEmit attribute
    \param offset_exprs offset expressions
    \param emit_val Value to emit
    \param scope Scope where this expression is built    
    \return Expression implementing the emit.
   */
  virtual SgExpression *BuildGridEmit(
      SgExpression *grid_exp,            
      GridEmitAttribute *attr,
      const SgExpressionPtrList *offset_exprs,
      SgExpression *emit_val,
      SgScopeStatement *scope=NULL) = 0;
  

  virtual SgBasicBlock *BuildGridSet(
      SgExpression *grid_var, int num_dims,
      const SgExpressionPtrList &indices, SgExpression *val) = 0;

  virtual SgFunctionCallExp *BuildGridGetID(SgExpression *grid_var) = 0;
  
  virtual SgType *GetIndexType() {
    return sb::buildOpaqueType(PS_INDEX_TYPE_NAME, gs_);
  }

  virtual SgExprListExp *BuildStencilOffsetMax(const StencilRange &sr);
  virtual SgExprListExp *BuildStencilOffsetMin(const StencilRange &sr);

  //! Build an ivec array containing the size of a given grid.
  /*!
    \param g Grid object
    \return ivec expression of the grid size
   */
  virtual SgExprListExp *BuildSizeExprList(const Grid *g);
      
 protected:
  SgScopeStatement *gs_;
  
  virtual SgExprListExp *BuildStencilOffset(const StencilRange &sr,
                                            bool is_max);
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_RUNTIME_BUILDER_H_ */
