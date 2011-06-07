// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/map.h"

namespace physis {
namespace translator {

class ReferenceRuntimeBuilder {
 public:
  ReferenceRuntimeBuilder(SgScopeStatement *global_scope);
  virtual ~ReferenceRuntimeBuilder() {}
  virtual SgFunctionCallExp *BuildGridGetID(SgExpression *grid_var);
  virtual SgBasicBlock *BuildGridSet(
      SgExpression *grid_var, int num_dims,
      const SgExpressionPtrList &indices, SgExpression *val);
  virtual SgFunctionCallExp *BuildGridGet(
      SgExpression *grid_var, 
      const SgExpressionPtrList &indices, SgType *elm_type);
 protected:
  SgScopeStatement *gs_;
  SgType *index_t_;
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
