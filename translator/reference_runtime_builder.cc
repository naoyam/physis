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

const std::string
ReferenceRuntimeBuilder::grid_type_name_ = "__PSGrid";

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
    si::appendExpression(args,
                         sb::buildCastExp(
                             si::copyExpression(indices[i]),
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
    si::appendExpression(args,
                         sb::buildCastExp(
                             si::copyExpression(*it), index_t_));
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

SgExpression *ReferenceRuntimeBuilder::BuildOffset(
    SgInitializedName *gv,
    int num_dim,
    SgExprListExp *offset_exprs,
    bool is_kernel,
    SgScopeStatement *scope) {
  /*
    __PSGridGetOffsetND(g, i)
  */
  std::string func_name =
      "__PSGridGetOffset" + toString(num_dim) + "D";
  SgExprListExp *func_args = isSgExprListExp(
      si::deepCopyNode(offset_exprs));
  func_args->prepend_expression(
      sb::buildVarRefExp(gv->get_name(), scope));
  return sb::buildFunctionCallExp(func_name, GetIndexType(),
                                  func_args);
}

SgClassDeclaration *ReferenceRuntimeBuilder::GetGridDecl() {
  LOG_DEBUG() << "grid type name: " << grid_type_name_ << "\n";
  SgTypedefType *grid_type = isSgTypedefType(
      si::lookupNamedTypeInParentScopes(grid_type_name_, gs_));
  SgClassType *anont = isSgClassType(grid_type->get_base_type());
  PSAssert(anont);
  return isSgClassDeclaration(anont->get_declaration());
}

SgExpression *ReferenceRuntimeBuilder::BuildGet(
    SgInitializedName *gv,
    SgExprListExp *offset_exprs,
    SgScopeStatement *scope,
    TranslationContext *tx, bool is_kernel) {
  GridType *gt = tx->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgExpression *offset = BuildOffset(
      gv, nd, offset_exprs, is_kernel, scope);
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  SgClassDeclaration *grid_decl = GetGridDecl();
  SgExpression *buf =
      sb::buildArrowExp(g,
                        sb::buildVarRefExp("p0",
                                           grid_decl->get_definition()));
  SgExpression *elm_val = sb::buildPntrArrRefExp(
      sb::buildCastExp(buf, sb::buildPointerType(gt->getElmType())),
      offset);
  //rose_util::CopyASTAttribute<GridGetAttribute>(p0, node);
  return elm_val;
}


} // namespace translator
} // namespace physis
