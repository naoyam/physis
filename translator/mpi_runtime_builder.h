// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_MPI_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_MPI_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/stencil_range.h"
#include "translator/reference_runtime_builder.h"

namespace physis {
namespace translator {

class MPIRuntimeBuilder: public ReferenceRuntimeBuilder {
 public:
  MPIRuntimeBuilder(SgScopeStatement *global_scope):
      ReferenceRuntimeBuilder(global_scope) {}
  virtual ~MPIRuntimeBuilder() {}
  virtual SgFunctionCallExp *BuildIsRoot();
  virtual SgFunctionCallExp *BuildGetGridByID(SgExpression *id_exp);
  virtual SgFunctionCallExp *BuildDomainSetLocalSize(SgExpression *dom);

  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,                  
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic);
};

SgFunctionCallExp *BuildCallLoadSubgrid(SgExpression *grid_var,
                                        SgVariableDeclaration *grid_range,
                                        SgExpression *reuse);
SgFunctionCallExp *BuildCallLoadSubgridUniqueDim(SgExpression *grid_var,
                                                 StencilRange &sr,
                                                 SgExpression *reuse);

SgFunctionCallExp *BuildLoadNeighbor(SgExpression *grid_var,
                                     StencilRange &sr,
                                     SgScopeStatement *scope,
                                     SgExpression *reuse,
                                     SgExpression *overlap,
                                     bool is_periodic);
SgFunctionCallExp *BuildActivateRemoteGrid(SgExpression *grid_var,
                                           bool active);

                                   


} // namespace translator
} // namespace physis



#endif /* PHYSIS_TRANSLATOR_MPI_RUNTIME_BUILDER_H_ */
