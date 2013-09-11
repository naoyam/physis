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
  dom_type_ = isSgTypedefType(
      si::lookupNamedTypeInParentScopes(PS_DOMAIN_INTERNAL_TYPE_NAME,
                                        gs_));
  
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
                         si::copyExpression(indices[i]));
  }
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(fs, args);
  rose_util::AppendExprStatement(tb, fc);
  return tb;
}

SgExpression *ReferenceRuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,    
    bool is_kernel,
    bool is_periodic) {
  SgExpression *offset =
      BuildGridOffset(gvref, gt->rank(), offset_exprs,
                      is_kernel, is_periodic, sil);
  gvref = si::copyExpression(gvref);
  SgExpression *field = sb::buildVarRefExp("p0");
  SgExpression *p0 =
      (si::isPointerType(gvref->get_type())) ?
      isSgExpression(sb::buildArrowExp(gvref, field)) :
      isSgExpression(sb::buildDotExp(gvref, field));
  //si::fixVariableReferences(p0);
  //GridType *gt = rose_util::GetASTAttribute<GridType>(gv);
  p0 = sb::buildCastExp(p0, sb::buildPointerType(gt->point_type()));
  p0 = sb::buildPntrArrRefExp(p0, offset);
  GridGetAttribute *gga = new GridGetAttribute(
      gt, NULL, gva, is_kernel, is_periodic, sil);
  rose_util::AddASTAttribute<GridGetAttribute>(p0, gga);
  return p0;
}

SgExpression *ReferenceRuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,    
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic,
    const string &member_name) {
  SgExpression *x = BuildGridGet(gvref, gva, gt, offset_exprs,
                                 sil, is_kernel,
                                 is_periodic);
  SgExpression *xm = sb::buildDotExp(
      x, sb::buildVarRefExp(member_name));
  rose_util::CopyASTAttribute<GridGetAttribute>(xm, x);
  rose_util::RemoveASTAttribute<GridGetAttribute>(x);
  rose_util::GetASTAttribute<GridGetAttribute>(
      xm)->member_name() = member_name;
  return xm;
}

SgExpression *ReferenceRuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,    
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic,
    const string &member_name,
    const SgExpressionVector &array_indices) {
  SgExpression *get = BuildGridGet(gvref, gva, gt, offset_exprs,
                                   sil, is_kernel,
                                   is_periodic,
                                   member_name);
  FOREACH (it, array_indices.begin(), array_indices.end()) {
    get = sb::buildPntrArrRefExp(get, *it);
  }
  return get;
}


SgExpression *ReferenceRuntimeBuilder::BuildGridEmit(
    SgExpression *grid_exp,
    GridEmitAttribute *attr,
    const SgExpressionPtrList *offset_exprs,
    SgExpression *emit_val,
    SgScopeStatement *scope) {
  
  /*
    g->p1[offset] = value;
  */
  int nd = attr->gt()->rank();
  StencilIndexList sil;
  StencilIndexListInitSelf(sil, nd);
  string dst_buf_name = "p0";
  SgExpression *p1 =
      sb::buildArrowExp(grid_exp, sb::buildVarRefExp(dst_buf_name));
  p1 = sb::buildCastExp(p1, sb::buildPointerType(attr->gt()->point_type()));
  SgExpression *offset = BuildGridOffset(
      si::copyExpression(grid_exp),
      nd, offset_exprs, true, false, &sil);
  SgExpression *lhs = sb::buildPntrArrRefExp(p1, offset);
  
  if (attr->is_member_access()) {
    lhs = sb::buildDotExp(lhs, sb::buildVarRefExp(attr->member_name()));
    const vector<string> &array_offsets = attr->array_offsets();
    FOREACH (it, array_offsets.begin(), array_offsets.end()) {
      SgExpression *e = rose_util::ParseString(*it, scope);
      lhs = sb::buildPntrArrRefExp(lhs, e);
    }
  }
  LOG_DEBUG() << "emit lhs: " << lhs->unparseToString() << "\n";

  SgExpression *emit = sb::buildAssignOp(lhs, emit_val);
  LOG_DEBUG() << "emit: " << emit->unparseToString() << "\n";
  return emit;
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
  if (!si::isPointerType(grid_ref->get_type()))
    grid_ref = sb::buildAddressOfOp(grid_ref);
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
    const SgExpressionPtrList *offset_exprs,
    bool is_kernel,
    bool is_periodic,
    const StencilIndexList *sil) {
  /*
    __PSGridGetOffsetND(g, i)
  */
  std::string func_name = "__PSGridGetOffset";
  if (is_periodic) func_name += "Periodic";
  func_name += toString(num_dim) + "D";
  if (!si::isPointerType(gvref->get_type())) {
    gvref = sb::buildAddressOfOp(gvref);
  }
  SgExprListExp *offset_params = sb::buildExprListExp(gvref);
  FOREACH (it, offset_exprs->begin(),
           offset_exprs->end()) {
    si::appendExpression(offset_params, *it);
  }
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(func_name);
  SgFunctionCallExp *offset_fc =
      sb::buildFunctionCallExp(fs, offset_params);
  rose_util::AddASTAttribute<GridOffsetAttribute>(
      offset_fc, new GridOffsetAttribute(
          num_dim, is_periodic, sil));
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

SgExpression *ReferenceRuntimeBuilder::BuildDomFieldRef(SgExpression *domain,
                                                        string fname) {
  SgClassDeclaration *dom_decl =
      isSgClassDeclaration(
          isSgClassType(dom_type_->get_base_type())->get_declaration()->
          get_definingDeclaration());
  LOG_DEBUG() << "domain: " << domain->unparseToString() << "\n";
  SgExpression *field = sb::buildVarRefExp(fname,
                                           dom_decl->get_definition());
  SgType *ty = domain->get_type();
  PSAssert(ty && !isSgTypeUnknown(ty));
  if (si::isPointerType(ty)) {
    return sb::buildArrowExp(domain, field);
  } else {
    return sb::buildDotExp(domain, field);
  }
}

SgExpression *ReferenceRuntimeBuilder::BuildDomMinRef(SgExpression *domain) {
  return BuildDomFieldRef(domain, "local_min");
}

SgExpression *ReferenceRuntimeBuilder::BuildDomMaxRef(SgExpression *domain) {
  return BuildDomFieldRef(domain, "local_max");
}

SgExpression *ReferenceRuntimeBuilder::BuildDomMinRef(SgExpression *domain,
                                                      int dim) {
  SgExpression *exp = BuildDomMinRef(domain);
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}

SgExpression *ReferenceRuntimeBuilder::BuildDomMaxRef(SgExpression *domain,
                                                      int dim) {
  SgExpression *exp = BuildDomMaxRef(domain);
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}


string ReferenceRuntimeBuilder::GetStencilDomName() {
  return string("dom");
}

SgExpression *ReferenceRuntimeBuilder::BuildStencilFieldRef(
    SgExpression *stencil_ref, SgExpression *field) {
  SgType *ty = stencil_ref->get_type();
  PSAssert(ty && !isSgTypeUnknown(ty));
  if (si::isPointerType(ty)) {
    return sb::buildArrowExp(stencil_ref, field);
  } else {
    return sb::buildDotExp(stencil_ref, field);
  }
}


SgExpression *ReferenceRuntimeBuilder::BuildStencilFieldRef(
    SgExpression *stencil_ref, string name) {
  SgType *ty = stencil_ref->get_type();
  LOG_DEBUG() << "ty: " << ty->unparseToString() << "\n";
  PSAssert(ty && !isSgTypeUnknown(ty));
  SgType *stencil_type = NULL;
  if (si::isPointerType(ty)) {
    stencil_type = si::getElementType(stencil_ref->get_type());
  } else {
    stencil_type = stencil_ref->get_type();
  }
  if (isSgModifierType(stencil_type)) {
    stencil_type = isSgModifierType(stencil_type)->get_base_type();
  }
  SgClassType *stencil_class_type = isSgClassType(stencil_type);
  // If the type is resolved to the actual class type, locate the
  // actual definition of field. Otherwise, temporary create an
  // unbound reference to the name.
  SgVarRefExp *field = NULL;
  if (stencil_class_type) {
    SgClassDefinition *stencil_def =
        isSgClassDeclaration(
            stencil_class_type->get_declaration()->get_definingDeclaration())->
        get_definition();
    field = sb::buildVarRefExp(name, stencil_def);
  } else {
    // Temporary create an unbound reference; this does not pass the
    // AST consistency tests unless fixed.
    field = sb::buildVarRefExp(name);    
  }
  return BuildStencilFieldRef(stencil_ref, field);
}


SgExpression *ReferenceRuntimeBuilder::BuildStencilDomMinRef(
    SgExpression *stencil) {
  SgExpression *exp =
      BuildStencilFieldRef(stencil, GetStencilDomName());
  // s.dom.local_max
  return BuildDomMinRef(exp);  
}

SgExpression *ReferenceRuntimeBuilder::BuildStencilDomMinRef(
    SgExpression *stencil, int dim) {
  SgExpression *exp = BuildStencilDomMinRef(stencil);
  // s.dom.local_max[dim]
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}

SgExpression *ReferenceRuntimeBuilder::BuildStencilDomMaxRef(
    SgExpression *stencil) {
  SgExpression *exp =
      BuildStencilFieldRef(stencil, GetStencilDomName());
  // s.dom.local_max
  return BuildDomMaxRef(exp);  
}

SgExpression *ReferenceRuntimeBuilder::BuildStencilDomMaxRef(
    SgExpression *stencil, int dim) {
  SgExpression *exp = BuildStencilDomMaxRef(stencil);
  // s.dom.local_max[dim]
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}


} // namespace translator
} // namespace physis
