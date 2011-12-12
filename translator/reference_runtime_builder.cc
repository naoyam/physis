// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/reference_runtime_builder.h"

#include "translator/translation_util.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

ReferenceRuntimeBuilder::ReferenceRuntimeBuilder(
    SgScopeStatement *global_scope):
    RuntimeBuilder(global_scope) {
  PSAssert(index_t_ = si::lookupNamedTypeInParentScopes("index_t", gs_));
}

SgFunctionCallExp *ReferenceRuntimeBuilder::BuildGridGetID(
    SgExpression *grid_var) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGridGetID");
  PSAssert(fs);
  SgExprListExp *args = sb::buildExprListExp(grid_var);
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(fs, args);  
  return fc;
}

SgBasicBlock *ReferenceRuntimeBuilder::BuildGridSet(
    SgExpression *grid_var, int num_dims, const SgExpressionPtrList &indices,
    SgExpression *val) {
  SgBasicBlock *tb = sb::buildBasicBlock();
  SgVariableDeclaration *decl =
      sb::buildVariableDeclaration("t", val->get_type(),
                                   sb::buildAssignInitializer(val),
                                   tb);
  si::appendStatement(decl, tb);
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGridSet");
  PSAssert(fs);
  SgExprListExp *args = sb::buildExprListExp(
      grid_var, sb::buildAddressOfOp(sb::buildVarRefExp(decl)));
  for (int i = 0; i < num_dims; ++i) {
    args->append_expression(sb::buildCastExp(indices[i],
                                             index_t_));
  }
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(fs, args);
  rose_util::AppendExprStatement(tb, fc);
  return tb;
}

SgFunctionCallExp *ReferenceRuntimeBuilder::BuildGridGet(
    SgExpression *grid_var, const SgExpressionPtrList &indices,
    SgType *elm_type) {
  
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(
          "__PSGridGet" + GetTypeName(elm_type));
  PSAssert(fs);
  SgExprListExp *args = sb::buildExprListExp(
      grid_var);
  FOREACH (it, indices.begin(), indices.end()) {
    args->append_expression(sb::buildCastExp(*it, index_t_));
  }
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(fs, args);
  return fc;
}

SgFunctionCallExp *ReferenceRuntimeBuilder::BuildGridDim(
    SgExpression *grid_ref, int dim) {
  // PSGridDim accepts an integer parameter designating dimension,
  // where zero means the first dimension.
  dim = dim - 1;
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(
          "PSGridDim", gs_);
  PSAssert(fs);
  SgExprListExp *args = sb::buildExprListExp(
      grid_ref, sb::buildIntVal(dim));
  SgFunctionCallExp *grid_dim = sb::buildFunctionCallExp(fs, args);
  return grid_dim;
}

SgExpression *ReferenceRuntimeBuilder::BuildGridRefInRunKernel(
    SgInitializedName *gv,
    SgFunctionDeclaration *run_kernel) {
  SgInitializedName *stencil_param = run_kernel->get_args()[0];
  SgNamedType *type = isSgNamedType(
      isSgPointerType(stencil_param->get_type())->get_base_type());
  PSAssert(type);
  SgClassDeclaration *stencil_class_decl
      = isSgClassDeclaration(type->get_declaration());
  PSAssert(stencil_class_decl);
  SgClassDefinition *stencil_class_def =
      isSgClassDeclaration(
          stencil_class_decl->get_definingDeclaration())->
      get_definition();
  PSAssert(stencil_class_def);
  SgVariableSymbol *grid_field =
      si::lookupVariableSymbolInParentScopes(
          gv->get_name(), stencil_class_def);
  PSAssert(grid_field);
  SgExpression *grid_ref =
      rose_util::BuildFieldRef(sb::buildVarRefExp(stencil_param),
                               sb::buildVarRefExp(grid_field));
  return grid_ref;
}



} // namespace translator
} // namespace physis
