// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/cuda_runtime_builder.h"

#include "translator/translation_util.h"
#include "translator/cuda_util.h"

#include <string>

#define DIM_STR ("dim")

#define BLOCK_DIM_X_DEFAULT (64)
#define BLOCK_DIM_Y_DEFAULT (4)
#define BLOCK_DIM_Z_DEFAULT (1)

namespace si = SageInterface;
namespace sb = SageBuilder;
namespace ru = physis::translator::rose_util;
namespace cu = physis::translator::cuda_util;

namespace {

static bool IsOnDeviceGridTypeName(const std::string &tn) {
  //LOG_DEBUG() << "name: " << tn << "\n";
  if (!physis::endswith(tn, "_dev")) return false;
  std::string host_grid_name = tn.substr(0, tn.length() - 4);
  //LOG_DEBUG() << "Host grid name: " << host_grid_name << "\n";
  return physis::translator::GridType::isGridType(host_grid_name);
}

}

namespace physis {
namespace translator {

SgExpression *CUDARuntimeBuilder::BuildNumElmsExpr(
    SgExpression *dim_expr,
    int num_dims) {
  SgExpression *num_elms_rhs =
      ArrayRef(dim_expr, Int(0));
  for (int i = 1; i < num_dims; ++i) {
    num_elms_rhs =
        Mul(num_elms_rhs, ArrayRef(si::copyExpression(dim_expr), Int(i)));
  }
  return num_elms_rhs;
}

SgVariableDeclaration *CUDARuntimeBuilder::BuildNumElmsDecl(
    SgExpression *dim_expr,
    int num_dims) {
  SgExpression *num_elms_rhs = BuildNumElmsExpr(dim_expr, num_dims);
  SgVariableDeclaration *num_elms_decl =
      sb::buildVariableDeclaration(
          "num_elms", si::lookupNamedTypeInParentScopes("size_t"),
          sb::buildAssignInitializer(num_elms_rhs));
  return num_elms_decl;
}

SgVariableDeclaration *CUDARuntimeBuilder::BuildNumElmsDecl(
    SgVarRefExp *p_exp,
    SgClassDeclaration *type_decl,
    int num_dims) {
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();
  SgExpression *dim_expr =
      Arrow(p_exp, Var(isSgVariableDeclaration(members[0])));
  return BuildNumElmsDecl(dim_expr, num_dims);
}

SgVariableDeclaration *CUDARuntimeBuilder::BuildNumElmsDecl(
    SgVariableDeclaration *p_decl,
    SgClassDeclaration *type_decl,
    int num_dims) {
  return BuildNumElmsDecl(Var(p_decl),
                          type_decl, num_dims);
}

CUDARuntimeBuilder::CUDARuntimeBuilder(SgScopeStatement *global_scope,
                                       const Configuration &config,
                                       BuilderInterface *delegator):
    ReferenceRuntimeBuilder(global_scope, config, delegator),
    block_dim_x_(BLOCK_DIM_X_DEFAULT),
    block_dim_y_(BLOCK_DIM_Y_DEFAULT),
    block_dim_z_(BLOCK_DIM_Z_DEFAULT) {
  // Redefine the block size if specified in the configuration file
  cuda_block_size_vals_.clear();  /* clear */
  const pu::LuaValue *lv = config.Lookup(Configuration::CUDA_BLOCK_SIZE);
  if (lv) {
    const pu::LuaTable *tbl = lv->getAsLuaTable();
    PSAssert(tbl);
    const pu::LuaTable *tbl2 = tbl->lst().begin()->second->getAsLuaTable();
    if (tbl2) {
      if (tbl->lst().size() == 1 || !config.auto_tuning()) {
        tbl = tbl2; /* use first one */
      } else {
        block_dim_x_ = block_dim_y_ = block_dim_z_ = 0;
        /* get all selection from CUDA_BLOCK_SIZE */
        FOREACH (it, tbl->lst().begin(), tbl->lst().end()) {
          std::vector<SgExpression *> iv;
          std::vector<double> v;
          PSAssert(tbl2 = it->second->getAsLuaTable());
          PSAssert(tbl2->get(v));
          iv.push_back(Int((int)v[0]));
          iv.push_back(Int((int)v[1]));
          iv.push_back(Int((int)v[2]));
          cuda_block_size_vals_.push_back(
              sb::buildAggregateInitializer(sb::buildExprListExp(iv)));
        }
        return;
      }
    }
    std::vector<double> v;
    PSAssert(tbl->get(v));
    block_dim_x_ = (int)v[0];
    block_dim_y_ = (int)v[1];
    block_dim_z_ = (int)v[2];
  }
}

SgExpression *CUDARuntimeBuilder::BuildGridRefInRunKernel(
    SgInitializedName *gv,
    SgFunctionDeclaration *run_kernel) {
  const SgInitializedNamePtrList &plist =
      run_kernel->get_parameterList()->get_args();
  FOREACH (plist_it, plist.begin(), plist.end()) {
    SgInitializedName *pin = *plist_it;
    if (pin->get_name() == gv->get_name()) {
      // The grid parameter with the same name as gv found
      SgVariableSymbol *ps =
          isSgVariableSymbol(pin->get_symbol_from_symbol_table());
      return sb::buildAddressOfOp(Var(ps));
    }
  }
  LOG_ERROR() << "No grid parameter found.\n";
  PSAssert(0);
  return NULL;
}

static SgExpression *BuildGridMember(SgExpression *gvref,
                                     SgExpression *member) {
  if (si::isPointerType(gvref->get_type())) {
    return isSgExpression(Arrow(gvref, member));
  } else {
    return isSgExpression(Dot(gvref, member));
  }
}

SgExpression *CUDARuntimeBuilder::BuildGridOffset(
    SgExpression *gvref,
    int num_dim,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic) {
  LOG_DEBUG() << "build offset: " << gvref->unparseToString() << "\n";
  /*
    __PSGridGetOffsetND(g, i)
  */
  std::string func_name = "__PSGridGetOffset";
  if (is_periodic) func_name += "Periodic";
  func_name += toString(num_dim) + "D";
  if (is_kernel) func_name += "Dev";
  if (!si::isPointerType(gvref->get_type())) {
    gvref = sb::buildAddressOfOp(gvref);
  }
  SgExprListExp *offset_params = sb::buildExprListExp(
      gvref);
  FOREACH (it, offset_exprs->begin(),
           offset_exprs->end()) {
    LOG_DEBUG() << "offset exp: " << (*it)->unparseToString() << "\n";    
    si::appendExpression(offset_params,
                         *it);
  }
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(func_name);
  SgFunctionCallExp *offset_fc =
      sb::buildFunctionCallExp(fs, offset_params);
  ru::AddASTAttribute<GridOffsetAttribute>(
      offset_fc,
      new GridOffsetAttribute(num_dim, is_periodic, sil));
  return offset_fc;
}

SgExpression *CUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,                
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,    
    bool is_kernel,
    bool is_periodic) {
  if (gt->IsPrimitivePointType()) {
    return ReferenceRuntimeBuilder::
        BuildGridGet(gvref, gva, gt, offset_exprs, sil,
                     is_kernel, is_periodic);
  }

  // Build a function call to gt->aux_get_decl
  SgExpression *offset =
      BuildGridOffset(gvref, gt->rank(), offset_exprs,
                      sil, is_kernel, is_periodic);
  
  SgFunctionCallExp *get_call =
      sb::buildFunctionCallExp(
          sb::buildFunctionRefExp(gt->aux_get_decl()),
          sb::buildExprListExp(si::copyExpression(gvref),
                               offset));
  GridGetAttribute *gga = new GridGetAttribute(
      gt, NULL, gva, is_kernel, is_periodic, sil);
  ru::AddASTAttribute<GridGetAttribute>(get_call,
                                        gga);
  return get_call;
}

SgExpression *CUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,                    
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic,
    const string &member_name) {
  PSAssert(gt->IsUserDefinedPointType());

  if (!is_kernel) {
    LOG_ERROR() << "Not implemented\n";
    PSAbort(1);
  }
  // Build an expression like "g->x[offset]"
  
  SgExpression *offset =
      BuildGridOffset(gvref, gt->rank(), offset_exprs,
                      sil, is_kernel, is_periodic);

  gvref = si::copyExpression(gvref);
  SgExpression *member_exp = VarSafe(member_name, gt->aux_def());
  SgExpression *x = BuildGridMember(gvref, member_exp);
  x = ArrayRef(x, offset);

  GridGetAttribute *gga = new GridGetAttribute(
      gt, NULL, gva, is_kernel, is_periodic, sil, member_name);
  ru::AddASTAttribute<GridGetAttribute>(
      x, gga);
  return x;
}

SgVariableDeclaration *FindMember(SgClassDefinition *type,
                                  const string &name) {

  const SgDeclarationStatementPtrList &members =
      type->get_members();
  FOREACH (it, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*it);
    if (ru::GetName(member_decl) == name) {
      return member_decl;
    }
  }
  return NULL;
}

SgExpression *CUDARuntimeBuilder::BuildGridArrayMemberOffset(
    SgExpression *gvref,
    const GridType *gt,
    const string &member_name,
    const SgExpressionVector &array_indices) {

  SgVarRefExp *dim_ref = VarSafe("dim", gt->aux_def());
  SgExpression *g_dim = BuildGridMember(gvref, dim_ref);
      
  SgExpression *num_elms = BuildNumElmsExpr(
      g_dim, gt->rank());

  SgVariableDeclaration *md = FindMember(gt->point_def(),
                                         member_name);
  SgArrayType *mt = isSgArrayType(ru::GetType(md));
  PSAssert(mt);

  vector<size_t> dims;
  ru::GetArrayDim(mt, dims);
  SgExpression *array_offset = NULL;
  SgExpression *dim_offset = NULL;
  vector<size_t>::reverse_iterator dim_it = dims.rbegin();
  ENUMERATE (i, it, array_indices.rbegin(), array_indices.rend()) {
    if (array_offset == NULL) {
      array_offset = *it;
      dim_offset = Int(*dim_it);
    } else {
      array_offset = Add(array_offset, Mul(*it, dim_offset));
      dim_offset = Mul(si::copyExpression(dim_offset),
                       Int(*dim_it));
    }
    ++dim_it;
  }
  si::deleteAST(dim_offset);

  array_offset = Mul(array_offset, num_elms);
  
  return array_offset;
}  

SgExpression *CUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,      
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic,
    const string &member_name,
    const SgExpressionVector &array_indices) {

  PSAssert(gt->IsUserDefinedPointType());

  if (!is_kernel) {
    LOG_ERROR() << "Not implemented\n";
    PSAbort(1);
  }
  // Build an expression like "g->x[offset]"

  SgExpression *offset =
      BuildGridOffset(gvref, gt->rank(), offset_exprs,
                      sil, is_kernel, is_periodic);

  SgExpression *offset_with_array = Add(
      offset,
      BuildGridArrayMemberOffset(
          si::copyExpression(gvref), gt, member_name, array_indices));
  
  ru::CopyASTAttribute<GridOffsetAttribute>(
      offset_with_array, offset);
  ru::RemoveASTAttribute<GridOffsetAttribute>(offset);

  SgExpression *x = ArrayRef(
      BuildGridMember(si::copyExpression(gvref),
                      VarSafe(member_name, gt->aux_def())),
      offset_with_array);
  GridGetAttribute *gga = new GridGetAttribute(
      gt, NULL, gva, is_kernel, is_periodic, sil, member_name);
  ru::AddASTAttribute<GridGetAttribute>(
      x, gga);
  return x;
}


SgClassDeclaration *CUDARuntimeBuilder::BuildGridDevTypeForUserType(
    SgClassDeclaration *grid_decl, const GridType *gt) {
  /*
    Let X be the user type name, N be the number of dimension,
    
    struct __PSGridNDX_dev {
    PSIndex dim[N];
    [list of struct member arrays];
    };
  */
  string dev_type_name = "__" + gt->type_name() + "_dev";
  SgClassDeclaration *decl =
      sb::buildStructDeclaration(dev_type_name, gs_);
  SgClassDefinition *dev_def = decl->get_definition();

  // Add member dim
  SgType *dim_type = 
      sb::buildArrayType(BuildIndexType2(gs_),
                         Int(gt->rank()));
  si::appendStatement(sb::buildVariableDeclaration(DIM_STR, dim_type),
                      dev_def);
                               
  // Add a pointer for each struct member
  SgClassDefinition *def = gt->point_def();
  const SgDeclarationStatementPtrList &members =
      def->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgName member_name = ru::GetName(member_decl);
    SgType *member_type = si::getArrayElementType(
        ru::GetType(member_decl));
    SgVariableDeclaration *dev_type_member =
        sb::buildVariableDeclaration(
            member_name,
            sb::buildPointerType(member_type));
    si::appendStatement(dev_type_member, dev_def);
  }

  return decl;
}


static bool IsDimMember(SgVariableDeclaration *member) {
  SgName member_name = ru::GetName(member);
  return member_name == DIM_STR;
}


// Build a "new" function.
SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridNewFuncForUserType(
    const GridType *gt) {
  /*
    Example:
    void* __PSGridType_devNew(int num_dims, PSVectorInt dim) {
    __PSGridType_dev *p = malloc(sizeof(__PSGridType_dev));
    for (int i = 0; i < num_dim; ++i) {
    p->dim[i]  = dim[i];
    }
    cudaMalloc(&(p->x), sizeof(typeof(p->x)), dim[i]);
    cudaMalloc(&(p->y), sizeof(typeof(p->y)), dim[i]);
    cudaMalloc(&(p->z), sizeof(typeof(p->z)), dim[i]);
    return p;
    }
  */
  SgClassType *dev_type = static_cast<SgClassType*>(gt->aux_type());
  SgClassDefinition *dev_type_def = gt->aux_def();
  string func_name = dev_type->get_name() + "New";
  SgType *ret_type = sb::buildPointerType(dev_type);
  
  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();
  // int num_dims
  SgInitializedName *num_dim_p =
      sb::buildInitializedName("num_dims", sb::buildIntType());
  si::appendArg(pl, num_dim_p);
  // PSVectorInt dim
  SgInitializedName *dim_p =
      sb::buildInitializedName(
          DIM_STR,
          si::lookupNamedTypeInParentScopes("PSVectorInt", gs_));
  si::appendArg(pl, dim_p);

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildPointerType(sb::buildVoidType()), pl);
  si::setStatic(fdecl);

  SgBasicBlock *body = fdecl->get_definition()->get_body();
  
  // Allocate a struct
  // __PSGridType_dev *p = malloc(sizeof(__PSGridType_dev));
  SgVariableDeclaration *p_decl =
      sb::buildVariableDeclaration(
          "p", ret_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(
                  sb::buildFunctionCallExp(
                      "malloc",
                      sb::buildPointerType(sb::buildVoidType()),
                      sb::buildExprListExp(sb::buildSizeOfOp(dev_type))),
                  ret_type)));
  si::appendStatement(p_decl, body);
  
  // p->dim[i]  = dim[i];
  for (unsigned i = 0; i < gt->rank(); ++i) {
    SgVarRefExp *utype_dim = VarSafe(DIM_STR, dev_type_def);
    SgExpression *lhs = ArrayRef(
        Arrow(Var(p_decl), utype_dim), Int(i));
    SgExpression *rhs = ArrayRef(Var(dim_p), Int(i));
    si::appendStatement(sb::buildAssignStatement(lhs, rhs),
                        body);
  }

  // size_t num_elms = dim[0] * ...;
  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(Var(dim_p), gt->rank());
  si::appendStatement(num_elms_decl, body);
  
  // cudaMalloc(&(p->x), sizeof(typeof(p->x)) * dim[i]);
  const SgDeclarationStatementPtrList &members =
      gt->point_def()->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = ru::GetType(member_decl);
    SgExpression *size_exp =
        Mul(sb::buildSizeOfOp(
            si::getArrayElementType(member_type)),
            Var(num_elms_decl));
    if (isSgArrayType(member_type)) {
      size_exp = Mul(size_exp, Int(
          si::getArrayElementCount(isSgArrayType(member_type))));
    }
    SgVarRefExp *member_vref = VarSafe(ru::GetName(Var(member_decl)),
                                       dev_type_def);
    SgFunctionCallExp *malloc_call = cu::BuildCUDAMalloc(
        Arrow(Var(p_decl), member_vref), size_exp);
    si::appendStatement(sb::buildExprStatement(malloc_call),
                        body);
  }

  // return p;
  si::appendStatement(sb::buildReturnStmt(Var(p_decl)), body);
  
  return fdecl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridFreeFuncForUserType(
    const GridType *gt) {
  /*
    Example:
    void __PSGridType_devFree(void *v) {
    __PSGridType_dev *p = (__PSGridType_dev*)v;
    cudaFree(p->x);
    cudaFree(p->y);
    cudaFree(p->z);
    free(p);
    }
  */
  SgClassType *dev_type = static_cast<SgClassType*>(gt->aux_type());
  string func_name = dev_type->get_name() + "Free";
  SgType *ret_type = sb::buildVoidType();
  SgType *dev_ptr_type = sb::buildPointerType(dev_type);
  
  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();
  // void *v
  SgInitializedName *v_p =
      sb::buildInitializedName("v", sb::buildPointerType(sb::buildVoidType()));
  si::appendArg(pl, v_p);

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, ret_type, pl);
  si::setStatic(fdecl);  

  SgBasicBlock *body = fdecl->get_definition()->get_body();
  
  // __PSGridType_dev *p = (__PSGridType_dev*)v;
  SgVariableDeclaration *p_decl =
      sb::buildVariableDeclaration(
          "p", dev_ptr_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(Var(v_p), dev_ptr_type)));
  si::appendStatement(p_decl, body);

  // cudaFree(p->x);
  const SgDeclarationStatementPtrList &members =
      ((SgClassDeclaration*)gt->aux_decl())->
      get_definition()->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;    
    SgFunctionCallExp *call = cu::BuildCUDAFree(
        Arrow(Var(p_decl), Var(member_decl)));
    si::appendStatement(sb::buildExprStatement(call), body);
  }

  // free(p);
  si::appendStatement(
      sb::buildExprStatement(
          sb::buildFunctionCallExp(
              "free", sb::buildVoidType(),
              sb::buildExprListExp(Var(p_decl)))),
      body);
  
  return fdecl;
}

/*!
  
  \param user_type_def Original user type definition.
*/
// cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
void BuildCUDAMallocHost(SgBasicBlock *body,
                         SgClassDefinition *user_type_def,
                         SgInitializedName *num_elms,
                         SgVariableDeclaration *buf_decl) {
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();

  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = ru::GetType(member_decl);
    SgExpression *size_exp =
        Mul(sb::buildSizeOfOp(si::getArrayElementType(member_type)),
            Var(num_elms));
    if (isSgArrayType(member_type)) {
      size_exp = Mul(size_exp, Int(
          si::getArrayElementCount(isSgArrayType(member_type))));
    }
    SgFunctionCallExp *malloc_call = cu::BuildCUDAMallocHost(
        ArrayRef(Var(buf_decl), Int(i)), size_exp);
    si::appendStatement(sb::buildExprStatement(malloc_call), body);
  }
}

void BuildCUDAMemcpy(SgBasicBlock *body,
                     SgClassDefinition *user_type_def,
                     SgClassDefinition *dev_type_def,
                     SgInitializedName *num_elms,
                     SgVariableDeclaration *buf_decl,
                     SgVariableDeclaration *dev_decl,
                     bool host_to_dev) {
  
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();

  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = ru::GetType(member_decl);
    SgVarRefExp *dev_member_ref = VarSafe(ru::GetName(Var(member_decl)), dev_type_def);
    SgExpression *dev_p = Arrow(Var(dev_decl), dev_member_ref);
    SgExpression *host_p =
        ArrayRef(Var(buf_decl), Int(i));
    SgExpression *size_exp =
        Mul(sb::buildSizeOfOp(si::getArrayElementType(member_type)),
            Var(num_elms));
    if (isSgArrayType(member_type)) {
      size_exp = Mul(size_exp, Int(
          si::getArrayElementCount(isSgArrayType(member_type))));
    }
    SgFunctionCallExp *copy_call =
        host_to_dev ?
        cu::BuildCUDAMemcpy(dev_p, host_p, size_exp, cu::cudaMemcpyHostToDevice) :
        cu::BuildCUDAMemcpy(host_p, dev_p, size_exp, cu::cudaMemcpyDeviceToHost);
    si::appendStatement(sb::buildExprStatement(copy_call), body);
  }
}

void BuildCUDAFreeHost(SgBasicBlock *body,
                       SgClassDefinition *user_type_def,
                       SgVariableDeclaration *buf_decl) {
  
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();

  ENUMERATE (i, member, members.begin(), members.end()) {
    si::appendStatement(
        sb::buildExprStatement(
            cu::BuildCUDAFreeHost(ArrayRef(Var(buf_decl), Int(i)))),
        body);
  }
}

void CUDARuntimeBuilder::BuildUserTypeArrayTranspose(
    SgBasicBlock *loop_body,
    SgExpression *soa_exp,
    SgExpression *aos_exp,
    bool soa_to_aos,
    SgInitializedName *loop_counter,
    SgInitializedName *num_elms,
    SgArrayType *member_type) {
  
  int len = si::getArrayElementCount(member_type);
  SgType *elm_type = si::getArrayElementType(member_type);
  for (int i = 0; i < len; ++i) {
    SgExpression *aos_elm =
        ArrayRef(sb::buildCastExp(si::copyExpression(aos_exp),
                                  sb::buildPointerType(elm_type)),
                 Int(i));

    SgExpression *soa_index =
        Add(Var(loop_counter), Mul(Var(num_elms), Int(i)));
    SgExpression *soa_elm =
        ArrayRef(si::copyExpression(soa_exp), soa_index);
    si::appendStatement(
        sb::buildAssignStatement(
            soa_to_aos ? aos_elm : soa_elm,
            soa_to_aos ? soa_elm : aos_elm),
        loop_body);
  }
  si::deleteAST(aos_exp);
  si::deleteAST(soa_exp);
}

void CUDARuntimeBuilder::BuildUserTypeTranspose(
    SgBasicBlock *scope,
    SgClassDefinition *user_type_def,
    SgVariableDeclaration *soa_decl,
    SgVariableDeclaration *aos_decl,
    bool soa_to_aos,
    SgInitializedName *loop_counter,
    SgInitializedName *num_elms) {
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = ru::GetType(member_decl);
    // si::getArrayElementType returns type T if T is non-array type.
    SgExpression *soa_elm =
        sb::buildCastExp(ArrayRef(Var(soa_decl), Int(i)),
                         sb::buildPointerType(si::getArrayElementType(member_type)));
    SgVarRefExp *member_ref = VarSafe(member_decl);
    SgExpression *aos_elm = Dot(
        ArrayRef(Var(aos_decl), Var(loop_counter)),
        member_ref);
    if (isSgArrayType(member_type)) {
      BuildUserTypeArrayTranspose(scope, soa_elm,  aos_elm,
                                  soa_to_aos, loop_counter,
                                  num_elms, isSgArrayType(member_type));
    } else {
      soa_elm = ArrayRef(soa_elm, Var(loop_counter));
      si::appendStatement(
          sb::buildAssignStatement(
              soa_to_aos ? aos_elm : soa_elm,
              soa_to_aos ? soa_elm : aos_elm),
          scope);
    }
  }
}

SgFunctionParameterList *CUDARuntimeBuilder::BuildGridCopyFuncSigForUserType(
    bool is_copyout, SgInitializedNamePtrList &params) {
  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();
  // void *v
  SgInitializedName *v_p =
      sb::buildInitializedName(
          "v", sb::buildPointerType(sb::buildVoidType()));
  si::appendArg(pl, v_p);
  params.push_back(v_p);  
  // const void *src or void *dst
  SgType *host_param_type = 
      sb::buildPointerType(
          is_copyout ? (SgType*)sb::buildVoidType() :
          (SgType*)sb::buildConstType(sb::buildVoidType()));
  string host_name = is_copyout ? "dst" : "src";  
  SgInitializedName *host_param =
      sb::buildInitializedName(host_name, host_param_type);
  si::appendArg(pl, host_param);
  params.push_back(host_param);
  // size_t num_elms
  SgInitializedName *num_elms =
      sb::buildInitializedName("num_elms", ru::SizeType());
  si::appendArg(pl, num_elms);
  params.push_back(num_elms);
  return pl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridCopyFuncForUserType(
    const GridType *gt, bool is_copyout) {
  
  /*
    Example: Transfer SoA device data to AoS host buffer.    
    void __PSGridType_devCopyout(void *v, void *dst) {
    Type *dstp = (Type *)dst;
    __PSGridType_dev *p = (__PSGridType_dev*)v;
    void *tbuf[3];
    cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
    cudaMemcpy(tbuf[0], p->x, cudaMemcpyDeviceToHost);
    ...;
    for (int i = 0; i < num_elms; ++i) {
    dstp[i].x = ((float *)tbuf[0])[i];
    ...;
    }
    cudaFreeHost(tbuf[0]);
    ...;
    }

    Transpose AoS data to SoA for CUDA.
    void __PSGridType_devCopyin(void *v, const void *src) {
    __PSGridType_dev *p = (__PSGridType_dev*)v;
    const Type *srcp = (const Type *)src;
    void *tbuf[3];
    cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
    ...;
    for (int i = 0; i < num_elms; ++i) {
    Type v = srcp[i];
    ((float *)tbuf[0])[i] = v.x;
    ...;
    }
    cudaMemcpy(p->x, tbuf[0], cudaMemcpyHostToDevice);
    cudaFreeHost(tbuf[0]);
    ...;
    }
  */

  SgClassType *dev_type = static_cast<SgClassType*>(gt->aux_type());
  SgClassDefinition *dev_type_def = gt->aux_def();
  string func_name = dev_type->get_name();
  if (is_copyout) func_name += "Copyout"; else func_name += "Copyin";
  SgType *dev_ptr_type = sb::buildPointerType(dev_type);
  int num_point_elms = gt->point_def()->get_members().size();
  string host_name = is_copyout ? "dst" : "src";
  
  SgInitializedNamePtrList params;
  SgFunctionParameterList *pl =
      BuildGridCopyFuncSigForUserType(is_copyout, params);
  SgInitializedName *num_elms = params.back();

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildVoidType(), pl);
  si::setStatic(fdecl);

  // Function body
  SgBasicBlock *body = fdecl->get_definition()->get_body();
  
  SgVariableDeclaration *p_decl =
      sb::buildVariableDeclaration(
          "p", dev_ptr_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(Var(params[0]), dev_ptr_type)));
  si::appendStatement(p_decl, body);
  //Type *dstp = (Type *)dst;
  SgType *hostp_type =
      sb::buildPointerType(
          (is_copyout) ? gt->point_type() :
          sb::buildConstType(gt->point_type()));
  SgVariableDeclaration *hostp_decl =
      sb::buildVariableDeclaration(
          host_name + "p", hostp_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(Var(params[1]), hostp_type)));
  si::appendStatement(hostp_decl, body);
  // void *tbuf[3];
  SgVariableDeclaration *tbuf_decl =
      sb::buildVariableDeclaration(
          "tbuf",
          sb::buildArrayType(sb::buildPointerType(sb::buildVoidType()),
                             Int(num_point_elms)));
  si::appendStatement(tbuf_decl, body);
  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  BuildCUDAMallocHost(body, gt->point_def(), num_elms, tbuf_decl);
  
  if (is_copyout) {
    BuildCUDAMemcpy(
        body, gt->point_def(), dev_type_def, num_elms, tbuf_decl,
        p_decl, false);
  }
  
  // for (size_t i = 0; i < num_elms; ++i) {
  SgVariableDeclaration *init =
      sb::buildVariableDeclaration(
          "i", 
          si::lookupNamedTypeInParentScopes("size_t"),
          sb::buildAssignInitializer(Int(0)));
  SgExpression *cond =
      sb::buildLessThanOp(
          Var(init), Var(num_elms));
  SgExpression *incr = sb::buildPlusPlusOp(Var(init));
  SgBasicBlock *loop_body = sb::buildBasicBlock();
  SgForStatement *trans_loop =
      sb::buildForStatement(init, sb::buildExprStatement(cond),
                            incr, loop_body);
  si::appendStatement(trans_loop, body);

  BuildUserTypeTranspose(loop_body, gt->point_def(), tbuf_decl, hostp_decl,
                         is_copyout, init->get_variables()[0], num_elms);

  if (!is_copyout) {
    BuildCUDAMemcpy(
        body, gt->point_def(), dev_type_def, num_elms, tbuf_decl,
        p_decl, true);
  }
    
  BuildCUDAFreeHost(body, gt->point_def(), tbuf_decl);
  
  return fdecl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridGetFuncForUserType(
    const GridType *gt) {
  SgClassDeclaration *type_decl =
      (SgClassDeclaration*)gt->aux_decl();
  SgClassType *dev_type = static_cast<SgClassType*>(gt->aux_type());
  string func_name = dev_type->get_name() + "Get";

  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();
  // dev_type *g
  SgInitializedName *g_p =
      sb::buildInitializedName(
          "g", sb::buildPointerType(dev_type));
  si::appendArg(pl, g_p);
  // PSIndex offset
  SgInitializedName *offset_p =
      sb::buildInitializedName(
          "offset", BuildIndexType2(gs_));
  si::appendArg(pl, offset_p);

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, gt->point_type(), pl);
  cu::SetCUDADevice(fdecl);
  si::setStatic(fdecl);  

  // Function body
  SgBasicBlock *body = fdecl->get_definition()->get_body();
  // Type v = {g->x[offset], g->y[offset], g->z[offset]};

  SgVariableDeclaration *v_decl =
      sb::buildVariableDeclaration("v", gt->point_type());
  si::appendStatement(v_decl, body);

  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(Var(g_p), type_decl, gt->rank());
  bool has_array_type = false;
  
  const SgDeclarationStatementPtrList &members =
      gt->point_def()->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = ru::GetType(member_decl);
    SgVarRefExp *dev_member_ref = VarSafe(ru::GetName(member_decl), gt->aux_def());
    if (isSgArrayType(member_type)) {
      BuildUserTypeArrayTranspose(
          body, Arrow(Var(g_p), dev_member_ref),
          Dot(Var(v_decl), Var(member_decl)),
          true, offset_p,
          si::getFirstInitializedName(num_elms_decl),
          isSgArrayType(member_type));
      has_array_type = true;
    } else {
      SgExpression *x = ArrayRef(
          Arrow(Var(g_p), dev_member_ref), Var(offset_p));
      SgExprStatement *s = sb::buildAssignStatement(
          Dot(Var(v_decl), Var(member_decl)), x);
      si::appendStatement(s, body);
    }
  }

  if (has_array_type) {
    si::insertStatementAfter(v_decl, num_elms_decl);
  } else {
    si::deleteAST(num_elms_decl);
  }
  
  // return v;
  si::appendStatement(sb::buildReturnStmt(Var(v_decl)), body);

  return fdecl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridEmitFuncForUserType(
    const GridType *gt) {
  SgClassDeclaration *type_decl =
      (SgClassDeclaration*)gt->aux_decl();
  SgClassType *dev_type = static_cast<SgClassType*>(gt->aux_type());
  string func_name = dev_type->get_name() + "Emit";

  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();
  // dev_type *g
  SgInitializedName *g_p =
      sb::buildInitializedName("g", sb::buildPointerType(dev_type));
  si::appendArg(pl, g_p);
  // PSIndex offset
  SgInitializedName *offset_p =
      sb::buildInitializedName("offset", BuildIndexType2(gs_));
  si::appendArg(pl, offset_p);
  // point_type v;
  SgInitializedName *v_p =
      sb::buildInitializedName(
          "v", sb::buildReferenceType(gt->point_type()));
  si::appendArg(pl, v_p);

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildVoidType(), pl);
  cu::SetCUDADevice(fdecl);
  si::setStatic(fdecl);

  // Function body
  SgBasicBlock *body = fdecl->get_definition()->get_body();

  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(Var(g_p), type_decl, gt->rank());
  bool has_array_type = false;
  
  // g->x[offset] = v.x;
  // g->y[offset] = v.y;
  // g->z[offset]} = v.z;

  const SgDeclarationStatementPtrList &members =
      gt->point_def()->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = ru::GetType(member_decl);
    string member_name = ru::GetName(member_decl);
    if (isSgArrayType(member_type)) {
      BuildUserTypeArrayTranspose(
          body, Arrow(Var(g_p), VarSafe(member_name, gt->aux_def())),
          Dot(Var(v_p), Var(member_decl)),
          false, offset_p,
          si::getFirstInitializedName(num_elms_decl),
          isSgArrayType(member_type));
      has_array_type = true;
    } else {
      SgExpression *lhs = ArrayRef(
          Arrow(Var(g_p), Var(member_decl)),
          Var(offset_p));
      SgExpression *rhs = Dot(Var(v_p), Var(member_decl));
      si::appendStatement(
          sb::buildAssignStatement(lhs, rhs), body);
    }
  }
                  
  if (has_array_type) {
    si::prependStatement(num_elms_decl, body);
  } else {
    si::deleteAST(num_elms_decl);
  }

  return fdecl;
}

SgExpression *CUDARuntimeBuilder::BuildGridEmit(
    SgExpression *grid_exp,    
    GridEmitAttribute *attr,
    const SgExpressionPtrList *offset_exprs,
    SgExpression *emit_val,
    SgScopeStatement *scope) {

  GridType *gt = attr->gt();
  if (gt->IsPrimitivePointType()) {
    return ReferenceRuntimeBuilder::BuildGridEmit(
        grid_exp, attr, offset_exprs, emit_val, scope);
  }

  int nd = gt->rank();
  StencilIndexList sil;
  StencilIndexListInitSelf(sil, nd);  
  
  SgExpression *offset = BuildGridOffset(
      grid_exp, nd, offset_exprs, &sil, true, false);

  SgExpression *emit_expr = NULL;
  if (attr->is_member_access()) {
    const vector<string> &array_offsets = attr->array_offsets();
    if (array_offsets.size() > 0) {
      SgExpressionVector offset_vector;
      FOREACH (it, array_offsets.begin(), array_offsets.end()) {
        SgExpression *e = ru::ParseString(*it, scope);
        offset_vector.push_back(e);
      }
      SgExpression *array_offset = Add(
          offset, BuildGridArrayMemberOffset(
              si::copyExpression(grid_exp), gt, attr->member_name(),
              offset_vector));
      ru::CopyASTAttribute<GridOffsetAttribute>(
          array_offset, offset);
      ru::RemoveASTAttribute<GridOffsetAttribute>(offset);
      offset = array_offset;
    }
    emit_expr = sb::buildAssignOp(
        ArrayRef(
            Arrow(
                si::copyExpression(grid_exp),
                VarSafe(attr->member_name(), gt->aux_def())),
            offset), emit_val);
  } else {
    emit_expr =
        sb::buildFunctionCallExp(
            sb::buildFunctionRefExp(gt->aux_emit_decl()),
            sb::buildExprListExp(
                si::copyExpression(grid_exp),
                offset, emit_val));
  }

  return emit_expr;
}

// Expressions themselves in index_args are used (no copy)
SgExprListExp *CUDARuntimeBuilder::BuildKernelCallArgList(
    StencilMap *stencil,
    SgExpressionPtrList &index_args,
    SgFunctionParameterList *params) {

  SgExprListExp *args = sb::buildExprListExp();
  FOREACH(it, index_args.begin(), index_args.end()) {
    si::appendExpression(args, *it);
  }

  const SgInitializedNamePtrList &param_ins = params->get_args();
  SgInitializedNamePtrList::const_iterator pend = param_ins.end();
  if (stencil->IsRedBlackVariant()) {
    // skip the color param
    --pend;
  }
  // skip the domain parameter  
  FOREACH(it, ++(param_ins.begin()), pend) {
    SgInitializedName *in = *it;
    SgExpression *exp = Var(in);
    SgType *param_type = in->get_type();
    //LOG_DEBUG() << "Param type: " << param_type->unparseToString() << "\n";
    // Pass a pointer if the param is a grid variable
    if (isSgNamedType(param_type)) {
      const string tn = isSgNamedType(param_type)->get_name().getString();
      if (IsOnDeviceGridTypeName(tn)) {
        //LOG_DEBUG() << "Grid parameter: " << tn << "\n";              
        exp = sb::buildAddressOfOp(exp);
      }
    }
    si::appendExpression(args, exp);
  }

  return args;
}

void CUDARuntimeBuilder::BuildKernelIndices(
    StencilMap *stencil,
    SgBasicBlock *call_site,
    vector<SgVariableDeclaration*> &indices) {
  int dim = stencil->getNumDim();
  
  // x = blockIdx.x * blockDim.x + threadIdx.x;
  SgExpression *init_x =
      Add(Mul(cu::BuildCudaIdxExp(cu::kBlockIdxX),
              cu::BuildCudaIdxExp(cu::kBlockDimX)),
          cu::BuildCudaIdxExp(cu::kThreadIdxX));
  if (stencil->IsRedBlackVariant()) {
    init_x = Mul(init_x, Int(2));
  }
  SgVariableDeclaration *x_index = BuildLoopIndexVarDecl(
      1, init_x, call_site);
  if (ru::IsCLikeLanguage()) {
    si::appendStatement(x_index, call_site);
  }
  indices.push_back(x_index);

  if (dim >= 2) {
    // y = blockIdx.y * blockDim.y + threadIdx.y;        
    SgVariableDeclaration *y_index = BuildLoopIndexVarDecl(
        2, Add(Mul(cu::BuildCudaIdxExp(cu::kBlockIdxY),
                   cu::BuildCudaIdxExp(cu::kBlockDimY)),
               cu::BuildCudaIdxExp(cu::kThreadIdxY)),
        call_site);
    if (ru::IsCLikeLanguage()) {
      si::appendStatement(y_index, call_site);
    }
    indices.push_back(y_index);
  }
  
  if (dim >= 3) {
    SgVariableDeclaration *z_index = BuildLoopIndexVarDecl(
        3, NULL, call_site);
    if (ru::IsCLikeLanguage()) {
      si::appendStatement(z_index, call_site);
    }
    indices.push_back(z_index);
  }
}

SgScopeStatement *CUDARuntimeBuilder::BuildKernelCallPreamble1D(
    StencilMap *stencil,    
    SgFunctionParameterList *param,    
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  SgInitializedName *dom_arg = Builder()->GetDomArgParamInRunKernelFunc(param, 1);    
  SgVariableDeclaration* t[] = {indices[0]};
  vector<SgVariableDeclaration*> range_checking_idx(t, t + 1);
  si::appendStatement(
      BuildDomainInclusionCheck(
          range_checking_idx, dom_arg, sb::buildReturnStmt()),
      call_site);
  return call_site;
}   

SgScopeStatement *CUDARuntimeBuilder::BuildKernelCallPreamble2D(
    StencilMap *stencil,    
    SgFunctionParameterList *param,    
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  SgInitializedName *dom_arg = Builder()->GetDomArgParamInRunKernelFunc(param, 2);  
  SgVariableDeclaration* t[] = {indices[0], indices[1]};
  vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
  si::appendStatement(
      BuildDomainInclusionCheck(
          range_checking_idx, dom_arg, sb::buildReturnStmt()),
      call_site);
  return call_site;
}   
  
SgScopeStatement *CUDARuntimeBuilder::BuildKernelCallPreamble3D(
    StencilMap *stencil,
    SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  int dim = 3;
  SgInitializedName *dom_arg = Builder()->GetDomArgParamInRunKernelFunc(param, dim);
  SgExpression *loop_begin =
      BuildDomMinRef(Var(dom_arg), dim);
  SgStatement *loop_init = sb::buildAssignStatement(
      Var(indices.back()), loop_begin);
  SgExpression *loop_end =
      BuildDomMaxRef(Var(dom_arg), dim);
  SgStatement *loop_test = sb::buildExprStatement(
      sb::buildLessThanOp(Var(indices.back()),
                          loop_end));

  SgVariableDeclaration* t[] = {
    stencil->IsRedBlackVariant() ? NULL: indices[0], indices[1]};
  vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
  si::appendStatement(
      BuildDomainInclusionCheck(
          range_checking_idx, dom_arg, sb::buildReturnStmt()),
      call_site);

  SgExpression *loop_incr = sb::buildPlusPlusOp(Var(indices.back()));
  SgBasicBlock *kernel_call_block = sb::buildBasicBlock();
  SgStatement *loop
      = sb::buildForStatement(loop_init, loop_test,
                              loop_incr, kernel_call_block);
  si::appendStatement(loop, call_site);
  ru::AddASTAttribute(loop, new RunKernelLoopAttribute(dim));

  if (stencil->IsRedBlackVariant()) {
    SgExpression *rb_offset_init =
        Add(Var(indices[0]),
            sb::buildBitAndOp(Add(Var(indices[1]), Var(indices[2]),
                                  Var(param->get_args().back())),
                              Int(1)));
    SgVariableDeclaration *x_index_rb =
        sb::buildVariableDeclaration(
            indices[0]->get_variables()[0]->get_name() + "_rb",
            sb::buildIntType(), sb::buildAssignInitializer(rb_offset_init),
            kernel_call_block);
    indices[0] = x_index_rb;
    si::appendStatement(x_index_rb, kernel_call_block);
    SgVariableDeclaration* t[] = {x_index_rb};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 1);
    si::appendStatement(
        BuildDomainInclusionCheck(
            range_checking_idx, dom_arg, sb::buildContinueStmt()),
        kernel_call_block);
  }
  
  return kernel_call_block;
}

SgInitializedName *CUDARuntimeBuilder::GetDomArgParamInRunKernelFunc(
    SgFunctionParameterList *pl, int dim) {
  return pl->get_args()[0];
}

SgScopeStatement *CUDARuntimeBuilder::BuildKernelCallPreamble(
    StencilMap *stencil,
    SgFunctionParameterList *param,    
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  int dim = stencil->getNumDim();
  if (dim == 1) {
    call_site = BuildKernelCallPreamble1D(
        stencil, param, indices, call_site);
  } else if (dim == 2) {
    call_site = BuildKernelCallPreamble2D(
        stencil, param, indices, call_site);
  } else if (dim == 3) {
    call_site = BuildKernelCallPreamble3D(
        stencil, param, indices, call_site);
  } else {
    LOG_ERROR()
        << "Dimension larger than 3 not supported; given dimension: "
        << dim << "\n";
    PSAbort(1);
  }
  return call_site;
}

SgBasicBlock *CUDARuntimeBuilder::BuildRunKernelFuncBody(
    StencilMap *stencil, SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices) {
  LOG_DEBUG() << __FUNCTION__ << "\n";
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));
  
  Builder()->BuildKernelIndices(stencil, block, indices);

  SgScopeStatement *kernel_call_block = 
      BuildKernelCallPreamble(stencil, param, indices, block);
  
  SgExpressionPtrList index_args;
  FOREACH (it, indices.begin(), indices.end()) {
    index_args.push_back(Var(*it));
  }

  SgExprStatement *kernel_call =
      sb::buildExprStatement(
          BuildKernelCall(stencil, index_args, param));

  si::appendStatement(kernel_call, kernel_call_block);
  
  return block;
}

static SgClassDefinition *GetPSDomainStruct() {
  static  SgClassDeclaration *ps_domain_struct = NULL;
  if (ps_domain_struct == NULL) {
    SgTypedefSymbol *ts = si::lookupTypedefSymbolInParentScopes("__PSDomain");
    PSAssert(ts);
    SgType *bt = ts->get_declaration()->get_base_type();
    PSAssert(bt);
    LOG_DEBUG() << "PSDomain typedef'ed base type: " << bt->unparseToString() << "\n";
    PSAssert(isSgClassType(bt));
    ps_domain_struct = isSgClassDeclaration(isSgClassType(bt)->get_declaration());
  }
  PSAssert(ps_domain_struct);  
  return isSgClassDeclaration(ps_domain_struct->get_definingDeclaration())->get_definition();
}

static SgVariableDeclaration *GetPSDomainMember(const string &member_name) {
  SgClassDefinition *dom = GetPSDomainStruct();
  SgDeclarationStatementPtrList &members = dom->get_members();
  BOOST_FOREACH(SgDeclarationStatement *decl, members) {
    SgVariableDeclaration *vd = isSgVariableDeclaration(decl);
    PSAssert(vd);
    if (si::getFirstVariable(*vd).get_name() == member_name) {
      return vd;
    }
  }
  LOG_ERROR() << "No such member in PSDomain: " << member_name << "\n";
  return NULL;
}

SgIfStmt *CUDARuntimeBuilder::BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgInitializedName *dom_arg, SgStatement *true_stmt) {
  
  // check x and y domain coordinates, like:
  // if (x < dom.local_min[0] || x >= dom.local_max[0] ||
  //     y < dom.local_min[1] || y >= dom.local_max[1]) {
  //   return;
  // }

  SgExpression *test_all = NULL;
  ENUMERATE (dim, index_it, indices.begin(), indices.end()) {
    // No check for the unit-stride dimension when red-black ordering
    // is used
    SgVariableDeclaration *idx = *index_it;
    // NULL indicates no check required.
    if (idx == NULL) continue;
    SgExpression *dom_min = ArrayRef(
        Dot(Var(dom_arg), Var(GetPSDomainMember("local_min"))), Int(dim));
    SgExpression *dom_max = ArrayRef(
        Dot(Var(dom_arg), Var(GetPSDomainMember("local_max"))), Int(dim));
    SgExpression *test = sb::buildOrOp(
        sb::buildLessThanOp(Var(idx), dom_min),
        sb::buildGreaterOrEqualOp(Var(idx), dom_max));
    if (test_all) {
      test_all = sb::buildOrOp(test_all, test);
    } else {
      test_all = test;
    }
  }
  SgIfStmt *ifstmt =
      sb::buildIfStmt(test_all, true_stmt, NULL);
  return ifstmt;
}

SgFunctionParameterList *CUDARuntimeBuilder::BuildRunKernelFuncParameterList(
    StencilMap *stencil) {
  SgFunctionParameterList *params = sb::buildFunctionParameterList();
  SgClassDefinition *param_struct_def = stencil->GetStencilTypeDefinition();
  PSAssert(param_struct_def);

  SgInitializedName *dom_arg = NULL;
  // Build the parameter list for the function
  const SgDeclarationStatementPtrList &members =
      param_struct_def->get_members();
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    SgName arg_name = vars[0]->get_name();
    SgType *arg_type = vars[0]->get_type();
    if (GridType::isGridType(arg_type)) {
      LOG_DEBUG() << "Grid type param\n";
      SgType *gt = BuildOnDeviceGridType(
          ru::GetASTAttribute<GridType>(arg_type));
      LOG_DEBUG() << "On dev type: " << gt->unparseToString() << "\n";
      arg_type = gt;
      // skip the grid index
      ++member;
    }
    LOG_DEBUG() << "type: " << arg_type->unparseToString() << "\n";    
    SgInitializedName *arg = sb::buildInitializedName(
        arg_name, arg_type);
    si::appendArg(params, arg);    
    if (Domain::isDomainType(arg_type) && dom_arg == NULL) {
      dom_arg = arg;
    }
  }
  PSAssert(dom_arg);

  // Append RB color param
  if (stencil->IsRedBlackVariant()) {
    SgInitializedName *rb_param =
        sb::buildInitializedName(PS_STENCIL_MAP_RB_PARAM_NAME,
                                 sb::buildIntType());
    si::appendArg(params, rb_param);
  }
  
  return params;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildRunKernelFunc(StencilMap *stencil) {
  SgFunctionParameterList *params = Builder()->BuildRunKernelFuncParameterList(stencil);
  LOG_INFO() << "Declaring function: " << stencil->GetRunName() << "\n";
  SgFunctionDeclaration *run_func =
      sb::buildDefiningFunctionDeclaration(
          stencil->GetRunName(), sb::buildVoidType(),
          params, gs_);
  si::attachComment(run_func, "Generated by " + string(__FUNCTION__));
  cu::SetCUDAKernel(run_func);
  // Build and set the function body
  vector<SgVariableDeclaration*> indices;  
  SgBasicBlock *func_body = BuildRunKernelFuncBody(
      stencil, params, indices);
  ru::ReplaceFuncBody(run_func, func_body);
  // Mark this function as RunKernel
  ru::AddASTAttribute(run_func, new RunKernelAttribute(stencil));
  // Comment out since this is probably not necessary.
  //si::fixVariableReferences(run_func);
  return run_func;
}

SgType *CUDARuntimeBuilder::BuildOnDeviceGridType(GridType *gt) {
  PSAssert(gt);
  // If this type is a user-defined type, its device type is stored at
  // gt->aux_type when building the type definition.
  if (gt->aux_type()) return gt->aux_type();
  
  string ondev_type_name = "__" + gt->type_name() + "_dev";
  LOG_DEBUG() << "On device grid type name: "
              << ondev_type_name << "\n";
  SgType *t =
      si::lookupNamedTypeInParentScopes(ondev_type_name, gs_);
  PSAssert(t);
  return t;
}

SgExpression *CUDARuntimeBuilder::BuildGridGetDev(SgExpression *grid_var,
                                                  GridType *gt) {
  return sb::buildPointerDerefExp(
      sb::buildCastExp(
          sb::buildArrowExp(grid_var, sb::buildOpaqueVarRefExp("dev")),
          sb::buildPointerType(Builder()->BuildOnDeviceGridType(gt))));
}

SgVariableDeclaration *CUDARuntimeBuilder::BuildGridDimDeclaration(
    const SgName &name,
    int dim,
    SgExpression *dom_dim_x,
    SgExpression *dom_dim_y,    
    SgExpression *block_dim_x,
    SgExpression *block_dim_y,
    SgScopeStatement *scope) {
  SgExpression *dim_x =
      sb::buildDivideOp(
          dom_dim_x,
          sb::buildCastExp(block_dim_x, sb::buildDoubleType()));
  dim_x = BuildFunctionCall("ceil", dim_x);
  dim_x = sb::buildCastExp(dim_x, sb::buildIntType());
  SgExpression *dim_y = NULL;  
  if (dim >= 2) {
    dim_y = sb::buildDivideOp(
        dom_dim_y, sb::buildCastExp(block_dim_y,
                                    sb::buildDoubleType()));
    dim_y = BuildFunctionCall("ceil", dim_y);
    dim_y = sb::buildCastExp(dim_y, sb::buildIntType());
  } else {
    dim_y = Int(1);
  }
  SgExpression *dim_z = Int(1);
  SgVariableDeclaration *grid_dim =
      cu::BuildDim3Declaration(name, dim_x, dim_y, dim_z, scope);
  return grid_dim;
}

void CUDARuntimeBuilder::BuildRunFuncBody(
    Run *run, SgFunctionDeclaration *run_func) {
  SgBasicBlock *block = run_func->get_definition()->get_body();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));  
  // int i;
  SgVariableDeclaration *loop_index =
      sb::buildVariableDeclaration("i", sb::buildIntType(), NULL, block);
  si::appendStatement(loop_index, block);
  // i = 0;
  SgStatement *loop_init =
      sb::buildAssignStatement(Var(loop_index),
                               Int(0));
  // i < iter
  SgStatement *loop_test =
      sb::buildExprStatement(
          sb::buildLessThanOp(Var(loop_index), VarSafe("iter", block)));
  // ++i
  SgExpression *loop_incr = sb::buildPlusPlusOp(Var(loop_index));
  // Generate loop body
  SgBasicBlock *loop_body = BuildRunFuncLoopBody(run, run_func);
  SgForStatement *loop =
      sb::buildForStatement(loop_init, loop_test, loop_incr, loop_body);

  TraceStencilRun(run, loop, block);
  
  // cudaThreadSynchronize after each loop if requested
  if (config_.LookupFlag(Configuration::CUDA_KERNEL_ERROR_CHECK)) {
    si::insertStatementAfter(
        loop,
        sb::buildExprStatement(cu::BuildCUDADeviceSynchronize()));
    si::insertStatementBefore(
        loop,
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("cudaGetLastError"), NULL)));
    si::insertStatementAfter(
        loop,
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("__PSCheckCudaError"),
                sb::buildExprListExp(
                    sb::buildStringVal("Kernel Execution Failed!")))));
  }
  
  return;
}

SgBasicBlock *CUDARuntimeBuilder::BuildRunFuncLoopBody(
    Run *run, SgFunctionDeclaration *run_func) {
  SgBasicBlock *outer_block = run_func->get_definition()->get_body();
  
  SgBasicBlock *loop_body = sb::buildBasicBlock();

  // Generates a call to each of the stencil function specified in the
  // PSStencilRun.
  ENUMERATE(stencil_idx, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *sm = it->second;
    int nd = sm->getNumDim();

    // Generate cache config code for each kernel
    SgFunctionSymbol *func_sym = rose_util::getFunctionSymbol(sm->run());
    PSAssert(func_sym);

    string stencil_name = "s" + toString(stencil_idx);
    SgVariableSymbol *stencil_symbol =
        si::lookupVariableSymbolInParentScopes(
            stencil_name, outer_block);
    PSAssert(stencil_symbol);
    SgExprListExp *args = BuildCUDAKernelArgList(sm, stencil_symbol);

    SgVariableDeclaration *block_dim =
        cu::BuildDim3Declaration(
            stencil_name + "_block_dim",
            BuildBlockDimX(nd),
            BuildBlockDimY(nd),
            BuildBlockDimZ(nd),
            outer_block);
    si::appendStatement(block_dim, outer_block);

    SgExpression *dom_max0 = Builder()->BuildStencilDomMaxRef(
        VarSafe(stencil_symbol), 1);
    SgExpression *dom_max1 = Builder()->BuildStencilDomMaxRef(
        VarSafe(stencil_symbol), 2);

    if (sm->IsRedBlackVariant()) {
      dom_max0 = sb::buildDivideOp(dom_max0, Int(2));
    }

    SgVariableDeclaration *grid_dim =
        Builder()->BuildGridDimDeclaration(
            stencil_name + "_grid_dim", sm->getNumDim(),
            dom_max0, dom_max1, BuildBlockDimX(nd),
            BuildBlockDimY(nd), outer_block);
    si::appendStatement(grid_dim, outer_block);

    // Generate Kernel invocation code
    SgCudaKernelExecConfig *cuda_config =
        cu::BuildCudaKernelExecConfig(Var(grid_dim), Var(block_dim),
                                      NULL, NULL);
    SgCudaKernelCallExp *cuda_call =
        cu::BuildCudaKernelCallExp(sb::buildFunctionRefExp(sm->run()),
                                   args, cuda_config);
    si::appendStatement(sb::buildExprStatement(cuda_call), loop_body);    
    if (sm->IsRedBlackVariant()) {
      if (sm->IsRedBlack()) {
        SgCudaKernelCallExp *black_call =
            isSgCudaKernelCallExp(si::copyExpression(cuda_call));
        si::appendExpression(black_call->get_args(),
                             Int(1));
        si::appendStatement(sb::buildExprStatement(black_call), loop_body);
      }
      si::appendExpression(
          cuda_call->get_args(),
          Int(sm->IsBlack() ? 1 : 0));
    }
    //appendGridSwap(sm, stencil_name, false, loop_body);
  }
  return loop_body;
}

SgExpression *CUDARuntimeBuilder::BuildBlockDimX(int nd) {
  if (block_dim_x_ <= 0) {
    /* auto tuning & has dynamic arguments */
    return Var("x");
  }
  return Int(block_dim_x_);
}

SgExpression *CUDARuntimeBuilder::BuildBlockDimY(int nd) {
  if (nd < 2) {
    return Int(1);
  }
  if (block_dim_y_ <= 0) {
    /* auto tuning & has dynamic arguments */
    return Var("y");
  }
  return Int(block_dim_y_);  
}

SgExpression *CUDARuntimeBuilder::BuildBlockDimZ(int nd) {
  if (nd < 3) {
    return Int(1);
  }
  if (block_dim_z_ <= 0) {
    /* auto tuning & has dynamic arguments */
    return Var("z");
  }
  return Int(block_dim_z_);
}

SgExprListExp *CUDARuntimeBuilder::BuildCUDAKernelArgList(
    StencilMap *sm, SgVariableSymbol *sv) {
  // Build an argument list by expanding members of the parameter struct
  // e.g., struct {a, b, c}; -> (s.a, s.b, s.c)
  SgExprListExp *args = sb::buildExprListExp();
  SgClassDefinition *stencil_def = sm->GetStencilTypeDefinition();
  PSAssert(stencil_def);

  // Enumerate members of parameter struct
  const SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    SgExpression *arg =
        sb::buildDotExp(Var(sv), Var(member_decl));
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    // If the type of the member is grid, pass the device pointer.
    //GridType *gt = tx_->findGridType(vars[0]->get_type());
    GridType *gt = ru::GetASTAttribute<GridType>(vars[0]->get_type());
    if (gt) {
      arg = Builder()->BuildGridGetDev(arg, gt);
      // skip the grid index
      ++member;
    }
    si::appendExpression(args, arg);
  }
  return args;
}

/** add dynamic parameter
 * @param[in/out] parlist ... parameter list
 */
void CUDARuntimeBuilder::AddDynamicParameter(
    SgFunctionParameterList *parlist) {
  si::appendArg(parlist, sb::buildInitializedName("x", sb::buildIntType()));
  si::appendArg(parlist, sb::buildInitializedName("y", sb::buildIntType()));
  si::appendArg(parlist, sb::buildInitializedName("z", sb::buildIntType()));
}
/** add dynamic argument
 * @param[in/out] args ... arguments
 * @param[in] a_exp ... index expression
 */
void CUDARuntimeBuilder::AddDynamicArgument(
    SgExprListExp *args, SgExpression *a_exp) {
  SgExpression *a =
      sb::buildPntrArrRefExp(
          Var(
              sb::buildVariableDeclaration(
                  "__cuda_block_size",
                  sb::buildArrayType(cuda_block_size_type()))),
          a_exp);
  si::appendExpression(args, Dot(a, Var("x")));
  si::appendExpression(args, Dot(a, Var("y")));
  si::appendExpression(args, Dot(a, Var("z")));
}
/** add some code after dlclose()
 * @param[in] scope
 */
void CUDARuntimeBuilder::AddSyncAfterDlclose(
    SgScopeStatement *scope) {
  /* adHoc: cudaThreadSynchronize() need after dlclose().
   * if not, sometimes fail kernel calling.
   */
  si::appendStatement(
      sb::buildExprStatement(cu::BuildCUDADeviceSynchronize()),
      scope);
}

} // namespace translator
} // namespace physis

