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
  
  virtual SgClassDeclaration *BuildGridDevTypeForUserType(
      SgClassDeclaration *grid_decl,
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridNewForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridFreeForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridCopyinForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridCopyoutForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridGetForUserType(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridEmitForUserType(
      GridType *gt);
  
 protected:
  virtual SgFunctionDeclaration *BuildGridCopyForUserType(
      GridType *gt, bool is_copyout);
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
