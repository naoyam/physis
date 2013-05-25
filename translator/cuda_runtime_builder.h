// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_CUDA_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_CUDA_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/reference_runtime_builder.h"
#include "translator/map.h"

namespace physis {
namespace translator {

class CUDARuntimeBuilder: public ReferenceRuntimeBuilder {
 public:
  CUDARuntimeBuilder(SgScopeStatement *global_scope):
      ReferenceRuntimeBuilder(global_scope) {}
  virtual SgExpression *BuildGridRefInRunKernel(
      SgInitializedName *gv,
      SgFunctionDeclaration *run_kernel);
  virtual SgExpression *BuildGridOffset(
      SgExpression *gv,
      int num_dim,
      const SgExpressionPtrList *offset_exprs,
      bool is_kernel,
      bool is_periodic,
      const StencilIndexList *sil);

  virtual SgExpression *BuildGridArrayMemberOffset(
    SgExpression *gvref,
    GridType *gt,
    const string &member_name,
    const SgExpressionVector &array_indices);

  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridType *gt,      
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,      
      bool is_kernel,
      bool is_periodic);
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,      
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name);
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,      
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name,
      const SgExpressionVector &array_indices);

  //! Build code for grid emit.
  /*!
    \param attr GridEmit attribute
    \param gt GridType of the grid
    \param offset_exprs offset expressions
    \param emit_val Value to emit
    \return Expression implementing the emit.
   */
  virtual SgExpression *BuildGridEmit(
      GridEmitAttribute *attr,
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      SgExpression *emit_val);
  
  virtual SgClassDeclaration *BuildGridDevTypeForUserType(
      SgClassDeclaration *grid_decl,
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridNewFuncForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridFreeFuncForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridCopyinFuncForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridCopyoutFuncForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridGetFuncForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridEmitFuncForUserType(
      GridType *gt);
  
 protected:
  virtual SgFunctionDeclaration *BuildGridCopyFuncForUserType(
      GridType *gt, bool is_copyout);
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
