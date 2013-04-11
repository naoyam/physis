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
      SgExpressionPtrList *offset_exprs,
      bool is_kernel,
      bool is_periodic,
      const StencilIndexList *sil);
  virtual SgClassDeclaration *BuildGridDevType(
      SgClassDeclaration *grid_decl,
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridNew(
      GridType *gt);
  virtual SgFunctionDeclaration *BuildGridFree(
      GridType *gt);
  

  
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
