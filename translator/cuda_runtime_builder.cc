// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/cuda_runtime_builder.h"

#include "translator/translation_util.h"
#include "translator/cuda_builder.h"
#include "translator/SageBuilderEx.h"

#include <string>

#define DIM_STR ("dim")

namespace si = SageInterface;
namespace sb = SageBuilder;
namespace sbx = physis::translator::SageBuilderEx;
namespace ru = physis::translator::rose_util;

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

static SgExpression *BuildNumElmsExpr(
    SgExpression *dim_expr,
    int num_dims) {
  SgExpression *num_elms_rhs =
      sb::buildPntrArrRefExp(dim_expr, sb::buildIntVal(0));
  for (int i = 1; i < num_dims; ++i) {
    num_elms_rhs =
        sb::buildMultiplyOp(
            num_elms_rhs,
            sb::buildPntrArrRefExp(
                si::copyExpression(dim_expr),
                sb::buildIntVal(i)));
  }
  return num_elms_rhs;
}

static SgVariableDeclaration *BuildNumElmsDecl(
    SgExpression *dim_expr,
    int num_dims) {
  SgExpression *num_elms_rhs = BuildNumElmsExpr(dim_expr, num_dims);
  SgVariableDeclaration *num_elms_decl =
      sb::buildVariableDeclaration(
          "num_elms", si::lookupNamedTypeInParentScopes("size_t"),
          sb::buildAssignInitializer(num_elms_rhs));
  return num_elms_decl;
}

static SgVariableDeclaration *BuildNumElmsDecl(
    SgVarRefExp *p_exp,
    SgClassDeclaration *type_decl,
    int num_dims) {
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();
  SgExpression *dim_expr =
      sb::buildArrowExp(p_exp,
                        sb::buildVarRefExp(
                            isSgVariableDeclaration(members[0])));
  return BuildNumElmsDecl(dim_expr, num_dims);
}

static SgVariableDeclaration *BuildNumElmsDecl(
    SgVariableDeclaration *p_decl,
    SgClassDeclaration *type_decl,
    int num_dims) {
  return BuildNumElmsDecl(sb::buildVarRefExp(p_decl),
                          type_decl, num_dims);
}

SgExpression *CUDARuntimeBuilder::BuildGridRefInRunKernel(
    SgInitializedName *gv,
    SgFunctionDeclaration *run_kernel) {
  // Find the parameter with the same name as gv.
  // TODO:
  const SgInitializedNamePtrList &plist =
      run_kernel->get_parameterList()->get_args();
  FOREACH (plist_it, plist.begin(), plist.end()) {
    SgInitializedName *pin = *plist_it;
    if (pin->get_name() == gv->get_name()) {
      // The grid parameter with the same name as gv found
      SgVariableSymbol *ps =
          isSgVariableSymbol(pin->get_symbol_from_symbol_table());
      return sb::buildAddressOfOp(sb::buildVarRefExp(ps));
    }
  }
  LOG_ERROR() << "No grid parameter found.\n";
  PSAssert(0);
  return NULL;
}

static SgExpression *BuildGridMember(SgExpression *gvref,
                                     SgExpression *member) {
  if (si::isPointerType(gvref->get_type())) {
    return isSgExpression(sb::buildArrowExp(gvref, member));
  } else {
    return isSgExpression(sb::buildDotExp(gvref, member));
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
  rose_util::AddASTAttribute<GridOffsetAttribute>(
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
  rose_util::AddASTAttribute<GridGetAttribute>(get_call,
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
  SgExpression *member_exp = sb::buildVarRefExp(member_name);
  SgExpression *x = BuildGridMember(gvref, member_exp);
  x = sb::buildPntrArrRefExp(x, offset);

  GridGetAttribute *gga = new GridGetAttribute(
      gt, NULL, gva, is_kernel, is_periodic, sil, member_name);
  rose_util::AddASTAttribute<GridGetAttribute>(
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
    if (rose_util::GetName(member_decl) == name) {
      return member_decl;
    }
  }
  return NULL;
}

static void GetArrayDim(SgArrayType *at,
                        vector<size_t> &dims) {
  
  SgExpression *idx = at->get_index();
  PSAssert (idx != NULL);

  size_t d;
  PSAssert(rose_util::GetIntLikeVal(idx, d));
  dims.push_back(d);
  if (isSgArrayType(at->get_base_type()))
    GetArrayDim(isSgArrayType(at->get_base_type()), dims);
}

SgExpression *CUDARuntimeBuilder::BuildGridArrayMemberOffset(
    SgExpression *gvref,
    const GridType *gt,
    const string &member_name,
    const SgExpressionVector &array_indices) {

  SgVarRefExp *dim_ref = sb::buildOpaqueVarRefExp("dim");
  SgExpression *g_dim = BuildGridMember(gvref, dim_ref);
      
  SgExpression *num_elms = BuildNumElmsExpr(
      g_dim, gt->rank());

  SgVariableDeclaration *md = FindMember(gt->point_def(),
                                         member_name);
  SgArrayType *mt = isSgArrayType(rose_util::GetType(md));
  PSAssert(mt);

  vector<size_t> dims;
  GetArrayDim(mt, dims);
  SgExpression *array_offset = NULL;
  SgExpression *dim_offset = NULL;
  vector<size_t>::reverse_iterator dim_it = dims.rbegin();
  ENUMERATE (i, it, array_indices.rbegin(), array_indices.rend()) {
    if (array_offset == NULL) {
      array_offset = *it;
      dim_offset = sb::buildIntVal(*dim_it);
    } else {
      array_offset =
          sb::buildAddOp(
              array_offset,
              sb::buildMultiplyOp(
                  *it, dim_offset));
      dim_offset = sb::buildMultiplyOp(
          si::copyExpression(dim_offset),
          sb::buildIntVal(*dim_it));
    }
    ++dim_it;
  }
  si::deleteAST(dim_offset);

  array_offset = sb::buildMultiplyOp(
      array_offset, num_elms);
  
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

  SgExpression *offset_with_array = sb::buildAddOp(
      offset,
      BuildGridArrayMemberOffset(si::copyExpression(gvref), gt,
                                 member_name,
                                 array_indices));
  
  rose_util::CopyASTAttribute<GridOffsetAttribute>(
      offset_with_array, offset);
  rose_util::RemoveASTAttribute<GridOffsetAttribute>(offset);

  SgExpression *x = sb::buildPntrArrRefExp(
      BuildGridMember(si::copyExpression(gvref),
                      sb::buildOpaqueVarRefExp(member_name)),
      offset_with_array);
  GridGetAttribute *gga = new GridGetAttribute(
      gt, NULL, gva, is_kernel, is_periodic, sil, member_name);
  rose_util::AddASTAttribute<GridGetAttribute>(
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
                         sb::buildIntVal(gt->rank()));
  si::appendStatement(sb::buildVariableDeclaration(DIM_STR, dim_type),
                      dev_def);
                               
  // Add a pointer for each struct member
  SgClassDefinition *def = gt->point_def();
  const SgDeclarationStatementPtrList &members =
      def->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgName member_name = rose_util::GetName(member_decl);
    SgType *member_type = si::getArrayElementType(
        rose_util::GetType(member_decl));
    SgVariableDeclaration *dev_type_member =
        sb::buildVariableDeclaration(
            member_name,
            sb::buildPointerType(member_type));
    si::appendStatement(dev_type_member, dev_def);
  }

  return decl;
}


static bool IsDimMember(SgVariableDeclaration *member) {
  SgName member_name = rose_util::GetName(member);
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

  // Build the function body
  SgBasicBlock *body = sb::buildBasicBlock();
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
    SgExpression *lhs = sb::buildPntrArrRefExp(
        sb::buildArrowExp(sb::buildVarRefExp(p_decl),
                          sb::buildOpaqueVarRefExp(DIM_STR)),
        sb::buildIntVal(i));
    SgExpression *rhs = sb::buildPntrArrRefExp(sb::buildVarRefExp(dim_p),
                                               sb::buildIntVal(i));
    si::appendStatement(sb::buildAssignStatement(lhs, rhs),
                        body);
  }

  // size_t num_elms = dim[0] * ...;
  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(sb::buildVarRefExp(dim_p),
                       gt->rank());
  si::appendStatement(num_elms_decl, body);
  
  // cudaMalloc(&(p->x), sizeof(typeof(p->x)) * dim[i]);
  const SgDeclarationStatementPtrList &members =
      gt->point_def()->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = rose_util::GetType(member_decl);
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(
                si::getArrayElementType(member_type)),
            sb::buildVarRefExp(num_elms_decl));
    if (isSgArrayType(member_type)) {
      size_exp = sb::buildMultiplyOp(
          size_exp, sb::buildIntVal(
              si::getArrayElementCount(isSgArrayType(member_type))));
    }
    SgFunctionCallExp *malloc_call = BuildCudaMalloc(
        sb::buildArrowExp(sb::buildVarRefExp(p_decl),
                          sb::buildOpaqueVarRefExp(
                              rose_util::GetName(
                                  sb::buildVarRefExp(member_decl)))),
        size_exp);
    si::appendStatement(sb::buildExprStatement(malloc_call),
                        body);
  }


  // return p;
  si::appendStatement(sb::buildReturnStmt(sb::buildVarRefExp(p_decl)),
                      body);
  
  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildPointerType(sb::buildVoidType()), pl);
  rose_util::ReplaceFuncBody(fdecl, body);
  si::setStatic(fdecl);
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

  SgBasicBlock *body = sb::buildBasicBlock();
  // __PSGridType_dev *p = (__PSGridType_dev*)v;
  SgVariableDeclaration *p_decl =
      sb::buildVariableDeclaration(
          "p", dev_ptr_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(sb::buildVarRefExp(v_p),
                               dev_ptr_type)));
  si::appendStatement(p_decl, body);

  // cudaFree(p->x);
  const SgDeclarationStatementPtrList &members =
      ((SgClassDeclaration*)gt->aux_decl())->
      get_definition()->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;    
    SgFunctionCallExp *call = BuildCudaFree(
        sb::buildArrowExp(sb::buildVarRefExp(p_decl),
                          sb::buildVarRefExp(member_decl)));
    si::appendStatement(sb::buildExprStatement(call),
                        body);
  }

  // free(p);
  si::appendStatement(
      sb::buildExprStatement(
          sb::buildFunctionCallExp(
              "free", sb::buildVoidType(),
              sb::buildExprListExp(sb::buildVarRefExp(p_decl)))),
      body);
  
  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, ret_type, pl);
  rose_util::ReplaceFuncBody(fdecl, body);
  si::setStatic(fdecl);  
  return fdecl;
}

/*!
  
  \param user_type_def Original user type definition.
 */
// cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
void AppendCUDAMallocHost(SgBasicBlock *body,
                          SgClassDefinition *user_type_def,
                          SgVariableDeclaration *num_elms_decl,
                          SgVariableDeclaration *buf_decl) {
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();

  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = rose_util::GetType(member_decl);
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(si::getArrayElementType(member_type)),
            sb::buildVarRefExp(num_elms_decl));
    if (isSgArrayType(member_type)) {
      size_exp = sb::buildMultiplyOp(
          size_exp, sb::buildIntVal(
              si::getArrayElementCount(isSgArrayType(member_type))));
    }
    SgFunctionCallExp *malloc_call = BuildCudaMallocHost(
        sb::buildPntrArrRefExp(
            sb::buildVarRefExp(buf_decl),
            sb::buildIntVal(i)),
        size_exp);
    si::appendStatement(sb::buildExprStatement(malloc_call),
                        body);
  }
}

void AppendCUDAMemcpy(SgBasicBlock *body,
                      SgClassDefinition *user_type_def,                      
                      SgVariableDeclaration *num_elms_decl,
                      SgVariableDeclaration *buf_decl,
                      SgVariableDeclaration *dev_decl,
                      bool host_to_dev) {
  
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();

  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = rose_util::GetType(member_decl);
    SgExpression *dev_p =
        sb::buildArrowExp(sb::buildVarRefExp(dev_decl),
                          sb::buildOpaqueVarRefExp(
                              rose_util::GetName(
                                  sb::buildVarRefExp(member_decl))));
    SgExpression *host_p =
        sb::buildPntrArrRefExp(
            sb::buildVarRefExp(buf_decl),
            sb::buildIntVal(i));
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(
                si::getArrayElementType(member_type)),
            sb::buildVarRefExp(num_elms_decl));
    if (isSgArrayType(member_type)) {
      size_exp = sb::buildMultiplyOp(
          size_exp, sb::buildIntVal(
              si::getArrayElementCount(isSgArrayType(member_type))));
    }
    SgFunctionCallExp *copy_call =
        host_to_dev ?
        BuildCudaMemcpyHostToDevice(
            dev_p, host_p, size_exp) :
        BuildCudaMemcpyDeviceToHost(
            host_p, dev_p, size_exp);
    si::appendStatement(
        sb::buildExprStatement(copy_call),
        body);
  }
}


void AppendCUDAFreeHost(SgBasicBlock *body,
                        SgClassDefinition *user_type_def,
                        SgVariableDeclaration *buf_decl) {
  
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();

  ENUMERATE (i, member, members.begin(), members.end()) {
    si::appendStatement(
        sb::buildExprStatement(
            BuildCudaFreeHost(
                sb::buildPntrArrRefExp(
                    sb::buildVarRefExp(buf_decl),
                    sb::buildIntVal(i)))),
        body);
  }
}

void AppendArrayTranspose(
    SgBasicBlock *loop_body,
    SgExpression *soa_exp,
    SgExpression *aos_exp,
    bool soa_to_aos,
    SgInitializedName *loop_counter,
    SgVariableDeclaration *num_elms_decl,
    SgArrayType *member_type) {
  
  int len = si::getArrayElementCount(member_type);
  SgType *elm_type = si::getArrayElementType(member_type);
  for (int i = 0; i < len; ++i) {
    SgExpression *aos_elm =
        sb::buildPntrArrRefExp(
            sb::buildCastExp(
                si::copyExpression(aos_exp),
                sb::buildPointerType(elm_type)),
            sb::buildIntVal(i));

    SgExpression *soa_index =
        sb::buildAddOp(
            sb::buildVarRefExp(loop_counter),
            sb::buildMultiplyOp(sb::buildVarRefExp(num_elms_decl),
                                sb::buildIntVal(i)));
    SgExpression *soa_elm =
        sb::buildPntrArrRefExp(si::copyExpression(soa_exp),
                               soa_index);
    si::appendStatement(
          sb::buildAssignStatement(
              soa_to_aos ? aos_elm : soa_elm,
              soa_to_aos ? soa_elm : aos_elm),
          loop_body);
  }
  si::deleteAST(aos_exp);
  si::deleteAST(soa_exp);
}

void AppendTranspose(SgBasicBlock *scope,
                     SgClassDefinition *user_type_def,
                     SgVariableDeclaration *soa_decl,
                     SgVariableDeclaration *aos_decl,
                     bool soa_to_aos,
                     SgInitializedName *loop_counter,
                     SgVariableDeclaration *num_elms_decl) {
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = rose_util::GetType(member_decl);
    const string member_name = rose_util::GetName(member_decl);
    // si::getArrayElementType returns type T if T is non-array type.
    SgExpression *soa_elm =
        sb::buildCastExp(
            sb::buildPntrArrRefExp(
                sb::buildVarRefExp(soa_decl),
                sb::buildIntVal(i)),
            sb::buildPointerType(si::getArrayElementType(member_type)));
    SgExpression *aos_elm = sb::buildDotExp(
        sb::buildPntrArrRefExp(
            sb::buildVarRefExp(aos_decl),
            sb::buildVarRefExp(loop_counter)),
        sb::buildOpaqueVarRefExp(member_name));
    if (isSgArrayType(member_type)) {
      AppendArrayTranspose(scope, soa_elm,  aos_elm,
                           soa_to_aos, loop_counter,
                           num_elms_decl, isSgArrayType(member_type));
    } else {
      soa_elm =
          sb::buildPntrArrRefExp(
              soa_elm,
              sb::buildVarRefExp(loop_counter));
      si::appendStatement(
          sb::buildAssignStatement(
              soa_to_aos ? aos_elm : soa_elm,
              soa_to_aos ? soa_elm : aos_elm),
          scope);
    }
  }
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
  string func_name = dev_type->get_name();
  if (is_copyout) func_name += "Copyout"; else func_name += "Copyin";
  SgType *dev_ptr_type = sb::buildPointerType(dev_type);
  int num_point_elms = gt->point_def()->get_members().size();
  SgClassDeclaration *type_decl =
      (SgClassDeclaration*)gt->aux_decl();
  string host_name = is_copyout ? "dst" : "src";
  
  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();
  // void *v
  SgInitializedName *v_p =
      sb::buildInitializedName(
          "v", sb::buildPointerType(sb::buildVoidType()));
  si::appendArg(pl, v_p);
  // const void *src or void *dst
  SgType *host_param_type = 
      sb::buildPointerType(
          is_copyout ? (SgType*)sb::buildVoidType() :
          (SgType*)sb::buildConstType(sb::buildVoidType()));
  SgInitializedName *host_param =
      sb::buildInitializedName(
          host_name, host_param_type);
  si::appendArg(pl, host_param);

  // Function body
  SgBasicBlock *body = sb::buildBasicBlock();
  
  SgVariableDeclaration *p_decl =
      sb::buildVariableDeclaration(
          "p", dev_ptr_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(sb::buildVarRefExp(v_p),
                               dev_ptr_type)));
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
              sb::buildCastExp(sb::buildVarRefExp(host_param),
                               hostp_type)));
  si::appendStatement(hostp_decl, body);
  // void *tbuf[3];
  SgVariableDeclaration *tbuf_decl =
      sb::buildVariableDeclaration(
          "tbuf",
          sb::buildArrayType(sb::buildPointerType(sb::buildVoidType()),
                             sb::buildIntVal(num_point_elms)));
  si::appendStatement(tbuf_decl, body);
  // size_t num_elms = dim[0] * ...;
  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(p_decl, type_decl, gt->rank());
  si::appendStatement(num_elms_decl, body);
  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  AppendCUDAMallocHost(
      body, gt->point_def(), num_elms_decl, tbuf_decl);
  
  if (is_copyout) {
    AppendCUDAMemcpy(
        body, gt->point_def(), num_elms_decl, tbuf_decl,
        p_decl, false);
  }
  
  // for (size_t i = 0; i < num_elms; ++i) {
  SgVariableDeclaration *init =
      sb::buildVariableDeclaration(
          "i", 
          si::lookupNamedTypeInParentScopes("size_t"),
          sb::buildAssignInitializer(sb::buildIntVal(0)));
  SgExpression *cond =
      sb::buildLessThanOp(
          sb::buildVarRefExp(init),
          sb::buildVarRefExp(num_elms_decl));
  SgExpression *incr = sb::buildPlusPlusOp(sb::buildVarRefExp(init));
  SgBasicBlock *loop_body = sb::buildBasicBlock();
  SgForStatement *trans_loop =
      sb::buildForStatement(init, sb::buildExprStatement(cond),
                            incr, loop_body);
  si::appendStatement(trans_loop, body);

  AppendTranspose(loop_body, gt->point_def(),
                  tbuf_decl,
                  hostp_decl,
                  is_copyout,
                  init->get_variables()[0], num_elms_decl);

  if (!is_copyout) {
    AppendCUDAMemcpy(
        body, gt->point_def(), num_elms_decl, tbuf_decl,
        p_decl, true);
  }
    
  AppendCUDAFreeHost(body, gt->point_def(), tbuf_decl);
  
  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildVoidType(), pl);
  rose_util::ReplaceFuncBody(fdecl, body);
  si::setStatic(fdecl);    
  return fdecl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridCopyinFuncForUserType(
    const GridType *gt) {
  return BuildGridCopyFuncForUserType(gt, false);
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridCopyoutFuncForUserType(
    const GridType *gt) {
  return BuildGridCopyFuncForUserType(gt, true);
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

  // Function body
  SgBasicBlock *body = sb::buildBasicBlock();
  // Type v = {g->x[offset], g->y[offset], g->z[offset]};

  SgVariableDeclaration *v_decl =
      sb::buildVariableDeclaration(
          "v", gt->point_type());
  si::appendStatement(v_decl, body);

  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(sb::buildVarRefExp(g_p),
                       type_decl, gt->rank());
  bool has_array_type = false;
  
  const SgDeclarationStatementPtrList &members =
      gt->point_def()->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = rose_util::GetType(member_decl);
    string member_name = rose_util::GetName(member_decl);
    if (isSgArrayType(member_type)) {
      AppendArrayTranspose(
          body,
          sb::buildArrowExp(sb::buildVarRefExp(g_p),
                            sb::buildOpaqueVarRefExp(member_name)),
          sb::buildDotExp(sb::buildVarRefExp(v_decl),
                          sb::buildOpaqueVarRefExp(member_name)),
          true, offset_p,
          num_elms_decl, isSgArrayType(member_type));
      has_array_type = true;
    } else {
      SgExpression *x = sb::buildPntrArrRefExp(
          sb::buildArrowExp(sb::buildVarRefExp(g_p),
                            sb::buildOpaqueVarRefExp(member_name)),
          sb::buildVarRefExp(offset_p));
      SgExprStatement *s = sb::buildAssignStatement(
          sb::buildDotExp(sb::buildVarRefExp(v_decl),
                          sb::buildOpaqueVarRefExp(member_name)),
          x);
      
      si::appendStatement(s, body);
    }
  }

  if (has_array_type) {
    si::insertStatementAfter(v_decl, num_elms_decl);
  } else {
    si::deleteAST(num_elms_decl);
  }
  
  // return v;
  si::appendStatement(
      sb::buildReturnStmt(sb::buildVarRefExp(v_decl)),
      body);

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, gt->point_type(), pl);
  fdecl->get_functionModifier().setCudaDevice();
  si::setStatic(fdecl);
  rose_util::ReplaceFuncBody(fdecl, body);
  si::setStatic(fdecl);  
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
      sb::buildInitializedName(
          "g", sb::buildPointerType(dev_type));
  si::appendArg(pl, g_p);
  // PSIndex offset
  SgInitializedName *offset_p =
      sb::buildInitializedName(
          "offset", BuildIndexType2(gs_));
  si::appendArg(pl, offset_p);
  // point_type v;
  SgInitializedName *v_p =
      sb::buildInitializedName(
          "v", sb::buildReferenceType(gt->point_type()));
  si::appendArg(pl, v_p);

  // Function body
  SgBasicBlock *body = sb::buildBasicBlock();

  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(sb::buildVarRefExp(g_p),
                       type_decl, gt->rank());
  bool has_array_type = false;
  
  // g->x[offset] = v.x;
  // g->y[offset] = v.y;
  // g->z[offset]} = v.z;

  const SgDeclarationStatementPtrList &members =
      gt->point_def()->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = rose_util::GetType(member_decl);
    string member_name = rose_util::GetName(member_decl);
    if (isSgArrayType(member_type)) {
      AppendArrayTranspose(
          body,
          sb::buildArrowExp(sb::buildVarRefExp(g_p),
                            sb::buildVarRefExp(member_name)),
          sb::buildDotExp(sb::buildVarRefExp(v_p),
                          sb::buildVarRefExp(member_name)),
          false, offset_p,
          num_elms_decl, isSgArrayType(member_type));
      has_array_type = true;
    } else {
      SgExpression *lhs = sb::buildPntrArrRefExp(
          sb::buildArrowExp(sb::buildVarRefExp(g_p),
                            sb::buildVarRefExp(member_decl)),
          sb::buildVarRefExp(offset_p));
      SgExpression *rhs =
          sb::buildDotExp(sb::buildVarRefExp(v_p),
                          sb::buildVarRefExp(member_decl));
      si::appendStatement(
          sb::buildAssignStatement(lhs, rhs), body);
    }
  }

                  
  if (has_array_type) {
    si::prependStatement(num_elms_decl, body);
  } else {
    si::deleteAST(num_elms_decl);
  }

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildVoidType(), pl);
  fdecl->get_functionModifier().setCudaDevice();
  si::setStatic(fdecl);
  rose_util::ReplaceFuncBody(fdecl, body);
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
        SgExpression *e = rose_util::ParseString(*it, scope);
        offset_vector.push_back(e);
      }
      SgExpression *array_offset =
          sb::buildAddOp(
              offset,
              BuildGridArrayMemberOffset(
                  si::copyExpression(grid_exp),
                  gt, attr->member_name(),
                  offset_vector));
      rose_util::CopyASTAttribute<GridOffsetAttribute>(
          array_offset, offset);
      rose_util::RemoveASTAttribute<GridOffsetAttribute>(offset);
      offset = array_offset;
    }
    emit_expr =
        sb::buildAssignOp(
            sb::buildPntrArrRefExp(
                sb::buildArrowExp(
                    si::copyExpression(grid_exp),
                    sb::buildVarRefExp(attr->member_name())),
                offset),
            emit_val);
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
  // skip the domain parameter
  if (stencil->IsRedBlackVariant()) {
    // skip the color param
    --pend;
  }
  FOREACH(it, ++(param_ins.begin()), pend) {
    SgInitializedName *in = *it;
    SgExpression *exp = sb::buildVarRefExp(in);
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
      Add(Mul(sbx::buildCudaIdxExp(sbx::kBlockIdxX),
              sbx::buildCudaIdxExp(sbx::kBlockDimX)),
          sbx::buildCudaIdxExp(sbx::kThreadIdxX));
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
        2, Add(Mul(sbx::buildCudaIdxExp(sbx::kBlockIdxY),
                   sbx::buildCudaIdxExp(sbx::kBlockDimY)),
               sbx::buildCudaIdxExp(sbx::kThreadIdxY)),
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
    SgInitializedName *dom_arg,
    SgFunctionParameterList *param,    
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
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
    SgInitializedName *dom_arg,
    SgFunctionParameterList *param,    
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  SgVariableDeclaration* t[] = {indices[0], indices[2]};
  vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
  si::appendStatement(
      BuildDomainInclusionCheck(
          range_checking_idx, dom_arg, sb::buildReturnStmt()),
      call_site);
  return call_site;
}   
  
SgScopeStatement *CUDARuntimeBuilder::BuildKernelCallPreamble3D(
    StencilMap *stencil,
    SgInitializedName *dom_arg,
    SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  int dim = 3;
  SgExpression *loop_begin =
      BuildDomMinRef(sb::buildVarRefExp(dom_arg), dim);
  SgStatement *loop_init = sb::buildAssignStatement(
      sb::buildVarRefExp(indices.back()), loop_begin);
  SgExpression *loop_end =
      BuildDomMaxRef(sb::buildVarRefExp(dom_arg), dim);
  SgStatement *loop_test = sb::buildExprStatement(
      sb::buildLessThanOp(sb::buildVarRefExp(indices.back()),
                          loop_end));

  SgVariableDeclaration* t[] = {
    stencil->IsRedBlackVariant() ? NULL: indices[0], indices[1]};
  vector<SgVariableDeclaration*> range_checking_idx(t, t + 2);
  si::appendStatement(
      BuildDomainInclusionCheck(
          range_checking_idx, dom_arg, sb::buildReturnStmt()),
      call_site);

  SgExpression *loop_incr =
      sb::buildPlusPlusOp(sb::buildVarRefExp(indices.back()));
  SgBasicBlock *kernel_call_block = sb::buildBasicBlock();
  SgStatement *loop
      = sb::buildForStatement(loop_init, loop_test,
                              loop_incr, kernel_call_block);
  si::appendStatement(loop, call_site);
  rose_util::AddASTAttribute(
      loop,
      new RunKernelLoopAttribute(dim));

  if (stencil->IsRedBlackVariant()) {
    SgExpression *rb_offset_init =
        Add(sb::buildVarRefExp(indices[0]),
            sb::buildBitAndOp(
                Add(Var(indices[1]),
                    sb::buildVarRefExp(indices[2]),
                    sb::buildVarRefExp(param->get_args().back())),
                Int(1)));
    si::appendStatement(
        sb::buildAssignStatement(Var(indices[0]), rb_offset_init),
        kernel_call_block);
    SgVariableDeclaration* t[] = {indices[0]};
    vector<SgVariableDeclaration*> range_checking_idx(t, t + 1);
    si::appendStatement(
        BuildDomainInclusionCheck(
            range_checking_idx, dom_arg, sb::buildContinueStmt()),
        kernel_call_block);
  }
  
  return kernel_call_block;
}

SgScopeStatement *CUDARuntimeBuilder::BuildKernelCallPreamble(
    StencilMap *stencil,
    SgInitializedName *dom_arg,
    SgFunctionParameterList *param,    
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  int dim = stencil->getNumDim();
  if (dim == 1) {
    call_site = BuildKernelCallPreamble1D(
        stencil, dom_arg, param, indices, call_site);
  } else if (dim == 2) {
    call_site = BuildKernelCallPreamble2D(
        stencil, dom_arg, param, indices, call_site);
  } else if (dim == 3) {
    call_site = BuildKernelCallPreamble3D(
        stencil, dom_arg, param, indices, call_site);
  } else {
    LOG_ERROR()
        << "Dimension larger than 3 not supported; given dimension: "
        << dim << "\n";
    PSAbort(1);
  }
  return call_site;
}

SgBasicBlock *CUDARuntimeBuilder::BuildRunKernelBody(
    StencilMap *stencil, SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices) {
  LOG_DEBUG() << __FUNCTION__;
  SgInitializedName *dom_arg = param->get_args()[0];
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));
  
  BuildKernelIndices(stencil, block, indices);
  
  SgExpressionPtrList index_args;
  FOREACH (it, indices.begin(), indices.end()) {
    index_args.push_back(sb::buildVarRefExp(*it));
  }

  SgExprStatement *kernel_call =
      sb::buildExprStatement(
          BuildKernelCall(stencil, index_args, param));

  SgScopeStatement *kernel_call_block = 
      BuildKernelCallPreamble(stencil, dom_arg, param, indices, block);
  si::appendStatement(kernel_call, kernel_call_block);
  
  return block;
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
    SgExpression *dom_min = sb::buildPntrArrRefExp(
        sb::buildDotExp(sb::buildVarRefExp(dom_arg),
                        sb::buildVarRefExp("local_min")),
        Int(dim));
    SgExpression *dom_max = sb::buildPntrArrRefExp(
        sb::buildDotExp(sb::buildVarRefExp(dom_arg),
                        sb::buildVarRefExp("local_max")),
        Int(dim));
    SgExpression *test = sb::buildOrOp(
        sb::buildLessThanOp(sb::buildVarRefExp(idx), dom_min),
        sb::buildGreaterOrEqualOp(sb::buildVarRefExp(idx), dom_max));
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

} // namespace translator
} // namespace physis

