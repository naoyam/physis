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
  PSAssert(index_t_ = si::lookupNamedTypeInParentScopes(PS_INDEX_TYPE_NAME, gs_));
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

SgExpression *ReferenceRuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridType *gt,
    SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,    
    bool is_kernel,
    bool is_periodic) {
  SgExpression *offset =
      BuildGridOffset(gvref, gt->num_dim(), offset_exprs,
                      is_kernel, is_periodic, sil);
  SgExpression *p0 = sb::buildArrowExp(
      si::copyExpression(gvref), sb::buildVarRefExp("p0"));
  si::fixVariableReferences(p0);
  //GridType *gt = rose_util::GetASTAttribute<GridType>(gv);
  p0 = sb::buildCastExp(p0, sb::buildPointerType(gt->point_type()));
  p0 = sb::buildPntrArrRefExp(p0, offset);
  GridGetAttribute *gga = new GridGetAttribute(NULL, gt->num_dim(), is_kernel,
                                               sil);
  rose_util::AddASTAttribute<GridGetAttribute>(p0, gga);
  return p0;
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
      GetBaseType(stencil_param->get_type()));
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

SgExpression *ReferenceRuntimeBuilder::BuildGridOffset(
    SgExpression *gvref,
    int num_dim,
    SgExpressionPtrList *offset_exprs,
    bool is_kernel,
    bool is_periodic,
    const StencilIndexList *sil) {
  /*
    __PSGridGetOffsetND(g, i)
  */
  GridOffsetAttribute *goa = new GridOffsetAttribute(
      num_dim, is_periodic, sil, gvref);
  std::string func_name = "__PSGridGetOffset";
  if (is_periodic) func_name += "Periodic";
  func_name += toString(num_dim) + "D";
  SgExprListExp *offset_params = sb::buildExprListExp(gvref);
  FOREACH (it, offset_exprs->begin(),
           offset_exprs->end()) {
    si::appendExpression(offset_params,
                         *it);
    goa->AppendIndex(*it);
  }
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(func_name);
  SgFunctionCallExp *offset_fc =
      sb::buildFunctionCallExp(fs, offset_params);
  rose_util::AddASTAttribute<GridOffsetAttribute>(
      offset_fc, goa);
  return offset_fc;
}

SgClassDeclaration *ReferenceRuntimeBuilder::GetGridDecl() {
  LOG_DEBUG() << "grid type name: " << grid_type_name_ << "\n";
  SgTypedefType *grid_type = isSgTypedefType(
      si::lookupNamedTypeInParentScopes(grid_type_name_, gs_));
  SgClassType *anont = isSgClassType(grid_type->get_base_type());
  PSAssert(anont);
  return isSgClassDeclaration(anont->get_declaration());
}

/*
SgExpression *ReferenceRuntimeBuilder::BuildGet(
    SgInitializedName *gv,
    SgExprListExp *offset_exprs,
    SgScopeStatement *scope,
    TranslationContext *tx, bool is_kernel,
    bool is_periodic) {
  GridType *gt = tx->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgExpression *offset = BuildOffset(
      gv, nd, offset_exprs, is_kernel, is_periodic, scope);
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
*/


} // namespace translator
} // namespace physis
