// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/reference_runtime_builder.h"
#include "translator/translation_util.h"
#include "translator/rose_fortran.h"
#include "translator/map.h"

#include <boost/foreach.hpp>

namespace si = SageInterface;
namespace sb = SageBuilder;
namespace ru = physis::translator::rose_util;
namespace rf = physis::translator::rose_fortran;

namespace physis {
namespace translator {

ReferenceRuntimeBuilder::ReferenceRuntimeBuilder(
    SgScopeStatement *global_scope):
    BuilderInterface(), gs_(global_scope) {
  dom_type_ = isSgTypedefType(
      si::lookupNamedTypeInParentScopes(PS_DOMAIN_INTERNAL_TYPE_NAME,
                                        gs_));
  
}

const std::string
ReferenceRuntimeBuilder::grid_type_name_ = "__PSGrid";

SgFunctionCallExp *ReferenceRuntimeBuilder::BuildGridGetID(
    SgExpression *grid_var) {
  SgName fname = ru::IsCLikeLanguage() ?
      PS_GRID_GET_ID_NAME : PSF_GRID_GET_ID_NAME;
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(fname);
  SgExprListExp *args = sb::buildExprListExp(grid_var);
  SgFunctionCallExp *fc = NULL;
  if (fs) fc = sb::buildFunctionCallExp(fs, args);
  else fc = sb::buildFunctionCallExp(sb::buildFunctionRefExp(fname),
                                     args);
  return fc;
}

SgExpression *ReferenceRuntimeBuilder::BuildGridBaseAddr(
    SgExpression *gvref, SgType *point_type) {
  
  SgExpression *field = sb::buildVarRefExp(PS_GRID_RAW_PTR_NAME);
  SgExpression *p =
      (si::isPointerType(gvref->get_type())) ?
      isSgExpression(sb::buildArrowExp(gvref, field)) :
      isSgExpression(sb::buildDotExp(gvref, field));
  if (point_type != NULL) {
    p = sb::buildCastExp(p, sb::buildPointerType(point_type));
  }
  return p;
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
  ru::AppendExprStatement(tb, fc);
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
                      sil, is_kernel, is_periodic);
  SgExpression *p = BuildGridBaseAddr(
      si::copyExpression(gvref), gt->point_type());
  p = sb::buildPntrArrRefExp(p, offset);
  GridGetAttribute *gga = new GridGetAttribute(
      gt, NULL, gva, is_kernel, is_periodic, sil);
  ru::AddASTAttribute<GridGetAttribute>(p, gga);
  return p;
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
  ru::CopyASTAttribute<GridGetAttribute>(xm, x);
  ru::RemoveASTAttribute<GridGetAttribute>(x);
  ru::GetASTAttribute<GridGetAttribute>(
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
  SgExpression *p1 = BuildGridBaseAddr(grid_exp, attr->gt()->point_type());
  SgExpression *offset = BuildGridOffset(
      si::copyExpression(grid_exp),
      nd, offset_exprs, &sil, true, false);
  SgExpression *lhs = sb::buildPntrArrRefExp(p1, offset);
  
  if (attr->is_member_access()) {
    // TODO (buildVarRefExp(string))
    lhs = sb::buildDotExp(lhs, sb::buildVarRefExp(attr->member_name()));
    const vector<string> &array_offsets = attr->array_offsets();
    FOREACH (it, array_offsets.begin(), array_offsets.end()) {
      SgExpression *e = ru::ParseString(*it, scope);
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
      ru::BuildFieldRef(sb::buildVarRefExp(stencil_param),
                               sb::buildVarRefExp(grid_field));
  return grid_ref;
}

SgExpression *ReferenceRuntimeBuilder::BuildGridOffset(
    SgExpression *gvref,
    int num_dim,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic) {
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
  ru::AddASTAttribute<GridOffsetAttribute>(
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
  SgClassDeclaration *dom_decl = NULL;
  if (ru::IsCLikeLanguage()) {
    dom_decl = isSgClassDeclaration(
        isSgClassType(dom_type_->get_base_type())->get_declaration()->
        get_definingDeclaration());
  } else if (ru::IsFortranLikeLanguage()) {
    dom_decl = isSgClassDeclaration(
        isSgClassType(domain->get_type())->get_declaration()->
        get_definingDeclaration());
  }
    
  //LOG_DEBUG() << "domain: " << domain->unparseToString() << "\n";
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
  if (ru::IsCLikeLanguage()) --dim;
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}

SgExpression *ReferenceRuntimeBuilder::BuildDomMaxRef(SgExpression *domain,
                                                      int dim) {
  SgExpression *exp = BuildDomMaxRef(domain);
  if (ru::IsCLikeLanguage()) --dim;  
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
  if (ru::IsFortranLikeLanguage() || !si::isPointerType(ty)) {
    return sb::buildDotExp(stencil_ref, field);
  } else {
    return sb::buildArrowExp(stencil_ref, field);
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
  SgExpression *exp =
      BuildStencilFieldRef(stencil, GetStencilDomName());
  // s.dom.local_max
  return BuildDomMinRef(exp, dim);  
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
  //SgExpression *exp = BuildStencilDomMaxRef(stencil);
  SgExpression *exp =
      BuildStencilFieldRef(stencil, GetStencilDomName());
  return BuildDomMaxRef(exp, dim);  
}

static SgType *GetDomType(StencilMap *sm) {
  SgType *t = sm->getDom()->get_type();
  PSAssert(t);
  return t;
}

SgClassDeclaration *ReferenceRuntimeBuilder::BuildStencilMapType(StencilMap *s) {
  // build stencil struct
  SgClassDeclaration *decl = NULL;
  SgClassDefinition *def = NULL;
  if (ru::IsCLikeLanguage()) {
    decl = sb::buildStructDeclaration(s->GetTypeName(), gs_);
    def = sb::buildClassDefinition(decl);
  } else {
    decl = rf::BuildDerivedTypeStatementAndDefinition(s->GetTypeName(), gs_);
    ru::SetAccessModifierUnspecified(decl);
    def = decl->get_definition();
    // inherit PSStencil
    // See rose-public ML message by Nguyen Nhat Thanh at 5:59am, May
    // 10, 2013
#if 0     // Unparse doesn't support inheritance
    SgClassType *st = isSgClassType(
    si::lookupNamedTypeInParentScopes("PSStencil", si::getScope(s->getKernel())));
    PSAssert(st);
    LOG_DEBUG() << "PS Stencil Type: " << st->unparseToString() << "\n";
    SgClassDeclaration *stdecl = isSgClassDeclaration(st->get_declaration());
    SgBaseClass *bc = new SgBaseClass(stdecl, true);
    def->append_inheritance(bc);
#endif    
  }
  si::appendStatement(
      sb::buildVariableDeclaration(
          GetStencilDomName(), GetDomType(s), NULL, def),
      def);

  SgInitializedNamePtrList &args = s->getKernel()->get_args();
  SgInitializedNamePtrList::iterator arg_begin = args.begin();
  // skip the index args
  arg_begin += ru::GetASTAttribute<Domain>(s->getDom())->num_dims();

  FOREACH(it, arg_begin, args.end()) {
    SgInitializedName *a = *it;
    SgType *type = a->get_type();
    si::appendStatement(
        ru::BuildVariableDeclaration(a->get_name(),
                                     type, NULL, def),
        def);
    if (GridType::isGridType(type)) {
      si::appendStatement(
          ru::BuildVariableDeclaration(
              a->get_name() + "_index",
              sb::buildIntType(), NULL, def),
          def);
    }
  }
  return decl;
}

SgFunctionDeclaration *ReferenceRuntimeBuilder::BuildMap(StencilMap *stencil) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  SgType *ret_type = NULL;
  const string stencil_param_name("ps_stencil_p");
  SgInitializedName *stencil_param = NULL;
  if (ru::IsFortranLikeLanguage()) {
    stencil_param = sb::buildInitializedName(
        stencil_param_name,
        sb::buildPointerType(stencil->stencil_type()));
    si::appendArg(parlist, stencil_param);
    ret_type = sb::buildVoidType();
  } else {
    ret_type = stencil->stencil_type();
  }
  
  si::appendArg(parlist, sb::buildInitializedName("dom", GetDomType(stencil)));
  SgInitializedNamePtrList &args = stencil->getKernel()->get_args();
  SgInitializedNamePtrList::iterator arg_begin = args.begin();
  arg_begin += ru::GetASTAttribute<Domain>(stencil->getDom())->num_dims();
  FOREACH(it, arg_begin, args.end()) {
    si::appendArg(parlist, isSgInitializedName(si::deepCopyNode(*it)));
  }

  SgFunctionDeclaration *mapFunc = ru::BuildFunctionDeclaration(
      stencil->GetMapName(), ret_type, parlist, gs_);
  ru::SetFunctionStatic(mapFunc);
  SgFunctionDefinition *mapDef = mapFunc->get_definition();
  SgBasicBlock *bb = mapDef->get_body();
  si::attachComment(bb, "Generated by " + string(__FUNCTION__));

  // Fortran needs parameter declarations in function body
  // WARNING: This creates a new SgInitializedName, but it should
  // actually use the same SgInitializedName as its corresponding
  // parameter that is created by the above buildInitializedName
  // routine. So, the correct way would be to use
  // buildVariableDeclaration first and then use the genereated
  // SgInitializedName for building ParameterList. Howeever, this
  // doesn't work for C, since C doesn't need separate variable
  // declaration statements for function parameters.
  // For now, this seems to work fine, although AST consistency check
  // is not executed.
  if (ru::IsFortranLikeLanguage()) {
    BOOST_FOREACH (SgInitializedName *p, parlist->get_args()) {
      SgVariableDeclaration *vd = ru::BuildVariableDeclaration(
          p->get_name(), p->get_type(), NULL, mapDef);
      si::appendStatement(vd, bb);
    }
  }
  
  SgExprListExp *stencil_fields = sb::buildExprListExp();
  SgInitializedNamePtrList &mapArgs = parlist->get_args();
  int arg_skip = ru::IsFortranLikeLanguage() ? 1 : 0;
  BOOST_FOREACH (SgInitializedName *map_arg,
                 make_pair(mapArgs.begin() + arg_skip,
                           mapArgs.end())) {
    SgExpression *exp = sb::buildVarRefExp(map_arg, mapDef);
    si::appendExpression(stencil_fields, exp);
    if (GridType::isGridType(exp->get_type())) {
      if (ru::IsCLikeLanguage()) {
        si::appendExpression(stencil_fields,
                             BuildGridGetID(exp));
      } else {
        // TODO Fortran: GridGetID not supported
        si::appendExpression(stencil_fields,
                             sb::buildIntVal(0));
      }
    }
  }

  if (ru::IsCLikeLanguage()) {
    SgInitializer *init = 
        sb::buildAggregateInitializer(stencil_fields);
    SgVariableDeclaration *svar =
        sb::buildVariableDeclaration(
            "stencil", stencil->stencil_type(),
            init, bb);
    si::appendStatement(svar, bb);
    si::appendStatement(sb::buildReturnStmt(sb::buildVarRefExp(svar)),
                        bb);
  } else if (ru::IsFortranLikeLanguage()) {
    string stype_name = stencil->stencil_type()->get_name();    
#if 0
    SgAssignOp *aop = sb::buildAssignOp(
        sb::buildFunctionRefExp(mapFunc),
        sb::buildFunctionCallExp(
            sb::buildFunctionRefExp(stype_name),
            stencil_fields));
    si::appendStatement(sb::buildExprStatement(aop), bb);
#else
    SgAllocateStatement *as = rf::BuildAllocateStatement();
    as->set_expr_list(sb::buildExprListExp(sb::buildVarRefExp(stencil_param)));
    as->set_source_expression(
        sb::buildFunctionCallExp(
            sb::buildFunctionRefExp(stype_name),
            stencil_fields));
    si::appendStatement(as, bb);
#endif
  }
  ru::ReplaceFuncBody(mapFunc, bb);
  return mapFunc;
}

SgFunctionCallExp* ReferenceRuntimeBuilder::BuildKernelCall(
    StencilMap *stencil, SgExpressionPtrList &index_args,
    SgFunctionParameterList *run_kernel_params) {
  SgExprListExp *args = BuildKernelCallArgList(
      stencil, index_args, run_kernel_params);
  SgFunctionCallExp *c =
      sb::buildFunctionCallExp(
          rose_util::getFunctionSymbol(stencil->getKernel()), args);
  return c;
}

SgExprListExp *ReferenceRuntimeBuilder::BuildKernelCallArgList(
      StencilMap *stencil,
      SgExpressionPtrList &index_args,
      SgFunctionParameterList *run_kernel_params) {      
  SgExprListExp *args = sb::buildExprListExp();
  FOREACH (it, index_args.begin(), index_args.end()) {
    si::appendExpression(args, *it);
  }
  // The params parameter has only one parameter inside, which is a
  // pointer to the StencilMap struct.
  SgInitializedName *stencil_param = run_kernel_params->get_args()[0];
  // append the fields of the stencil type to the argument list
  SgClassDefinition *stencilDef = stencil->GetStencilTypeDefinition();  
  SgDeclarationStatementPtrList &members = stencilDef->get_members();
  FOREACH (it, ++(members.begin()), members.end()) {
    SgVariableDeclaration *d = isSgVariableDeclaration(*it);
    assert(d);
    LOG_DEBUG() << "member: " << d->unparseToString() << "\n";
    SgVarRefExp *stencil = sb::buildVarRefExp(stencil_param);
    SgExpression *exp = ru::BuildFieldRef(stencil, sb::buildVarRefExp(d));
    SgVariableDefinition *var_def = d->get_definition();
    ROSE_ASSERT(var_def);
    si::appendExpression(args, exp);
    // skip the grid id
    if (GridType::isGridType(exp->get_type())) {
      ++it;
    }
  }
  return args;
}

static SgExpression *BuildRedBlackInitOffset(
    SgExpression *idx,
    vector<SgVariableDeclaration*> &indices,
    SgInitializedName *rb_param,
    int nd) {
  // idx + idx & 1 ^ (i1 + i2 + ... + c) % 2  
  SgExpression *rb_offset = sb::buildVarRefExp(rb_param);
  for (int i = 1; i < nd; ++i) {
    rb_offset = sb::buildAddOp(rb_offset, sb::buildVarRefExp(indices[i]));
  }
  rb_offset = sb::buildModOp(rb_offset, sb::buildIntVal(2));
  rb_offset = sb::buildBitXorOp(sb::buildBitAndOp(si::copyExpression(idx),
                                                  sb::buildIntVal(1)),
                                rb_offset);
  return rb_offset;
}

// NOTE: param has to be associated with an function declaration. If
// it is just built with SageBuilder and is not yet used for building
// a function declaration, BuildStencilDomMinRef will fail.
SgBasicBlock *ReferenceRuntimeBuilder::BuildRunKernelFuncBody(
    StencilMap *stencil, SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices) {
  LOG_DEBUG() << "Generating run kernel body\n";
  SgInitializedName *stencil_param = param->get_args()[0];
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));

  // Generate code like this
  // for (int k = dom.local_min[2]; k <= dom.local_max[2]-1; k++) {
  //   for (int j = dom.local_min[1]; j <= dom.local_max[1]-1; j++) {
  //     for (int i = dom.local_min[0]; i <= dom.local_max[0]-1; i++) {
  //       kernel(i, j, k, g);
  //     }
  //   }
  // }

  LOG_DEBUG() << "Generating nested loop\n";
  SgScopeStatement *parent_block = block;
  indices.resize(stencil->getNumDim(), NULL);
  for (int i = stencil->getNumDim()-1; i >= 0; --i) {
    SgVariableDeclaration *index_decl = BuildLoopIndexVarDecl(i+1, NULL, parent_block);
    indices[i] = index_decl;
    if (ru::IsCLikeLanguage()) {
      si::appendStatement(index_decl, parent_block);
    }
    SgExpression *loop_begin =
        BuildStencilDomMinRef(
            sb::buildVarRefExp(stencil_param), i+1);
    if (i == 0 && stencil->IsRedBlackVariant()) {
      loop_begin = sb::buildAddOp(
          loop_begin,
          BuildRedBlackInitOffset(loop_begin, indices,
                                  param->get_args()[1],
                                  stencil->getNumDim()));
    }
    SgInitializedName *loop_var = index_decl->get_variables()[0];
    // <= dom.local_max -1
    SgExpression *loop_end =
        sb::buildSubtractOp(
            BuildStencilDomMaxRef(
                sb::buildVarRefExp(stencil_param), i+1),
            sb::buildIntVal(1));
    SgExpression *incr =
        sb::buildIntVal((i == 0 && stencil->IsRedBlackVariant()) ? 2 : 1);
    SgBasicBlock *inner_block = sb::buildBasicBlock();
    //SgForStatement *loop_statement = sb::buildForStatement(init, test, incr, inner_block);
    SgScopeStatement *loop_statement =
        ru::BuildForLoop(loop_var, loop_begin, loop_end, incr, inner_block);    
    si::appendStatement(loop_statement, parent_block);
    rose_util::AddASTAttribute(
        loop_statement,
        new RunKernelLoopAttribute(i+1));
    parent_block = inner_block;
  }

  SgExpressionPtrList index_args;
  for (int i = 0; i < stencil->getNumDim(); ++i) {
    index_args.push_back(sb::buildVarRefExp(indices[i]));
  }
  SgFunctionCallExp *kernelCall =
      BuildKernelCall(stencil, index_args, param);
  si::appendStatement(sb::buildExprStatement(kernelCall),
                      parent_block);
  return block;
  
}

SgVariableDeclaration *ReferenceRuntimeBuilder::BuildLoopIndexVarDecl(
    int dim,
    SgExpression *init,
    SgScopeStatement *block) {
  string vn;
  switch (dim) {
    case 1:
      vn = LOOP_INDEX_VAR_NAME1;
      break;
    case 2:
      vn = LOOP_INDEX_VAR_NAME2;
      break;
    case 3:
      vn = LOOP_INDEX_VAR_NAME3;
      break;
    case 4:
      vn = LOOP_INDEX_VAR_NAME4;
      break;
    case 5:
      vn = LOOP_INDEX_VAR_NAME5;
      break;
  }
  SgVariableDeclaration *index_decl =         
      ru::BuildVariableDeclaration(
          vn, sb::buildIntType(),
          init ? sb::buildAssignInitializer(init) : NULL,
          block);
  rose_util::AddASTAttribute<RunKernelIndexVarAttribute>(
      index_decl, new RunKernelIndexVarAttribute(dim));
  return index_decl;
}

SgFunctionParameterList *ReferenceRuntimeBuilder::BuildRunKernelFuncParameterList(
    StencilMap *stencil) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();

  // Stencil type
  SgType *stencil_type = NULL;
  if (ru::IsCLikeLanguage()) {
    stencil_type = sb::buildConstType(sb::buildPointerType(
          sb::buildConstType(stencil->stencil_type())));
  } else if (ru::IsFortranLikeLanguage()) {
    stencil_type = stencil->stencil_type();
  }    

  SgInitializedName *stencil_param =
      sb::buildInitializedName(PS_STENCIL_MAP_STENCIL_PARAM_NAME, stencil_type);
  si::appendArg(parlist, stencil_param);
  
  if (stencil->IsRedBlackVariant()) {
    SgInitializedName *rb_param =
        sb::buildInitializedName(PS_STENCIL_MAP_RB_PARAM_NAME,
                                 sb::buildIntType());
    si::appendArg(parlist, rb_param);
  }
  return parlist;
}

SgFunctionDeclaration *ReferenceRuntimeBuilder::BuildRunKernelFunc(
    StencilMap *s) {
  SgFunctionParameterList *parlist =
      BuildRunKernelFuncParameterList(s);
  SgInitializedName *stencil_param = parlist->get_args()[0];  

  SgFunctionDeclaration *runFunc = ru::BuildFunctionDeclaration(
      s->GetRunName(), sb::buildVoidType(), parlist, gs_);
  rose_util::SetFunctionStatic(runFunc);
  si::attachComment(runFunc, "Generated by " + string(__FUNCTION__));

  // Build and set the function body
  vector<SgVariableDeclaration*> indices;
  si::replaceStatement(runFunc->get_definition()->get_body(),
                       BuildRunKernelFuncBody(s, parlist, indices));
  
  // Parameters and variable declarations need to be put forward in Fortran
  if (ru::IsFortranLikeLanguage()) {
    SgScopeStatement *body = runFunc->get_definition()->get_body();
    BOOST_FOREACH (SgVariableDeclaration *vd,
                   make_pair(indices.rbegin(), indices.rend())) {
      si::prependStatement(vd, body);
    }
    SgVariableDeclaration *vd = ru::BuildVariableDeclaration(
        stencil_param->get_name(), stencil_param->get_type(), NULL, body);
    LOG_DEBUG() << "sp type: " << stencil_param->get_type()->unparseToString()
                << "\n";
    si::prependStatement(vd, body);
  }
  rose_util::AddASTAttribute(
      runFunc, new RunKernelAttribute(s, stencil_param));

  return runFunc;
  
}
# if 0
SgFunctionDeclaration *ReferenceRuntimeBuilder::BuildRunKernelFunc(
    StencilMap *s, SgFunctionParameterList *params,
    SgBasicBlock *body, const vector<SgVariableDeclaration*> &indices) {
  
  SgFunctionDeclaration *runFunc = ru::BuildFunctionDeclaration(
      s->GetRunName(), sb::buildVoidType(), params, gs_);
  rose_util::SetFunctionStatic(runFunc);
  si::attachComment(runFunc, "Generated by " + string(__FUNCTION__));
  
  //  si::replaceStatement(runFunc->get_definition()->get_body(),
  //                       BuildRunKernelFuncBody(s, params, indices));
  si::replaceStatement(runFunc->get_definition()->get_body(),
                       body);
  // Parameters and variable declarations need to be put forward in Fortran
  if (ru::IsFortranLikeLanguage()) {
    SgScopeStatement *body = runFunc->get_definition()->get_body();
    BOOST_FOREACH (SgVariableDeclaration *vd,
                   make_pair(indices.rbegin(), indices.rend())) {
      si::prependStatement(vd, body);
    }
    BOOST_FOREACH (SgInitializedName *gn, params->get_args()) {
      SgVariableDeclaration *vd = ru::BuildVariableDeclaration(
          gn->get_name(), gn->get_type(), NULL, body);
      si::prependStatement(vd, body);
    }
  }
  SgInitializedName *stencil_param = params->get_args()[0];
  rose_util::AddASTAttribute(
      runFunc, new RunKernelAttribute(s, stencil_param));

  return runFunc;
  
}
#endif
SgExprListExp *ReferenceRuntimeBuilder::BuildStencilOffset(
    const StencilRange &sr, bool is_max) {
  SgExprListExp *exp_list = sb::buildExprListExp();
  IntVector offset_min, offset_max;
  int nd = sr.num_dims();
  if (sr.IsEmpty()) {
    for (int i = 0; i < nd; ++i) {
      offset_min.push_back(0);
      offset_max.push_back(0);
    }
  } else if (!sr.GetNeighborAccess(offset_min, offset_max)) {
    LOG_DEBUG() << "Stencil access is not regular: "
                << sr << "\n";
    return NULL;
  }
  for (int i = 0; i < (int)offset_min.size(); ++i) {
    PSIndex v = is_max ? offset_max[i] : offset_min[i];
    si::appendExpression(exp_list, sb::buildIntVal(v));
  }
  return exp_list;
}

SgExprListExp *ReferenceRuntimeBuilder::BuildStencilOffsetMax(
    const StencilRange &sr) {
  return BuildStencilOffset(sr, true);
}

SgExprListExp *ReferenceRuntimeBuilder::BuildStencilOffsetMin(
    const StencilRange &sr) {
  return BuildStencilOffset(sr, false);
}

SgExprListExp *ReferenceRuntimeBuilder::BuildSizeExprList(const Grid *g) {
  SgExprListExp *exp_list = sb::buildExprListExp();
  SgExpressionPtrList &args = g->new_call()->get_args()->get_expressions();  
  int nd = g->getType()->rank();
  for (int i = 0; i < PS_MAX_DIM; ++i) {
    SgExpression *e = i >= nd ?
        sb::buildIntVal(0) : si::copyExpression(args[i]);
    si::appendExpression(exp_list, e);
  }
  return exp_list;
}

} // namespace translator
} // namespace physis
