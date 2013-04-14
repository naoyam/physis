// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/cuda_runtime_builder.h"

#include "translator/translation_util.h"
#include "translator/cuda_builder.h"

#define DIM_STR ("dim")

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

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

SgExpression *CUDARuntimeBuilder::BuildGridOffset(
    SgExpression *gvref,
    int num_dim,
    const SgExpressionPtrList *offset_exprs,
    bool is_kernel,
    bool is_periodic,
    const StencilIndexList *sil) {
  LOG_DEBUG() << "build offset: " << gvref->unparseToString() << "\n";
  /*
    __PSGridGetOffsetND(g, i)
  */
  GridOffsetAttribute *goa = new GridOffsetAttribute(
      num_dim, is_periodic,
      sil, gvref);  
  std::string func_name = "__PSGridGetOffset";
  if (is_periodic) func_name += "Periodic";
  func_name += toString(num_dim) + "D";
  if (is_kernel) func_name += "Dev";
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

SgExpression *CUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,    
    bool is_kernel,
    bool is_periodic) {
  if (gt->IsPrimitivePointType()) {
    return ReferenceRuntimeBuilder::
        BuildGridGet(gvref, gt, offset_exprs, sil,
                     is_kernel, is_periodic);
  }

  // Build a function call to gt->aux_get_decl
  SgExpression *offset =
      BuildGridOffset(gvref, gt->num_dim(), offset_exprs,
                      is_kernel, is_periodic, sil);
  
  SgFunctionCallExp *get_call =
      sb::buildFunctionCallExp(
          sb::buildFunctionRefExp(gt->aux_get_decl()),
          sb::buildExprListExp(si::copyExpression(gvref),
                               offset));
  GridGetAttribute *gga = new GridGetAttribute(
      NULL, gt->num_dim(), is_kernel, is_periodic,
      sil, offset);
  rose_util::AddASTAttribute<GridGetAttribute>(get_call,
                                               gga);
  return get_call;
}

SgExpression *CUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,      
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
      BuildGridOffset(gvref, gt->num_dim(), offset_exprs,
                      is_kernel, is_periodic, sil);
  
  SgExpression *x = sb::buildPntrArrRefExp(
      sb::buildArrowExp(
          si::copyExpression(gvref),
          sb::buildVarRefExp(member_name)),
      offset);
  GridGetAttribute *gga = new GridGetAttribute(
      NULL, gt->num_dim(), is_kernel, is_periodic,
      sil, offset, member_name);
  rose_util::AddASTAttribute<GridGetAttribute>(
      x, gga);
  return x;
}

SgClassDeclaration *CUDARuntimeBuilder::BuildGridDevTypeForUserType(
    SgClassDeclaration *grid_decl, GridType *gt) {
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
                         sb::buildIntVal(gt->num_dim()));
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
    SgType *member_type = rose_util::GetType(member_decl);
    SgVariableDeclaration *dev_type_member =
        sb::buildVariableDeclaration(
            member_name,
            sb::buildPointerType(member_type));
    si::appendStatement(dev_type_member, dev_def);
  }

  return decl;
}

SgVariableDeclaration *BuildNumElmsDecl(
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

  SgVariableDeclaration *num_elms_decl =
      sb::buildVariableDeclaration(
          "num_elms", si::lookupNamedTypeInParentScopes("size_t"),
          sb::buildAssignInitializer(num_elms_rhs));
  return num_elms_decl;
}

SgVariableDeclaration *BuildNumElmsDecl(
    SgVariableDeclaration *p_decl,
    SgClassDeclaration *type_decl,
    int num_dims) {
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();
  SgExpression *dim_expr =
      sb::buildArrowExp(
          sb::buildVarRefExp(p_decl),
          sb::buildVarRefExp(
              isSgVariableDeclaration(members[0])));
  return BuildNumElmsDecl(dim_expr, num_dims);
}

bool IsDimMember(SgVariableDeclaration *member) {
  SgName member_name = rose_util::GetName(member);
  return member_name == DIM_STR;
}  

// Build a "new" function.
SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridNewForUserType(
    GridType *gt) {
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
  for (unsigned i = 0; i < gt->num_dim(); ++i) {
    SgExpression *lhs = sb::buildPntrArrRefExp(
        sb::buildArrowExp(sb::buildVarRefExp(p_decl),
                          sb::buildVarRefExp(DIM_STR)),
        sb::buildIntVal(i));
    SgExpression *rhs = sb::buildPntrArrRefExp(sb::buildVarRefExp(dim_p),
                                               sb::buildIntVal(i));
    si::appendStatement(sb::buildAssignStatement(lhs, rhs),
                        body);
  }

  // size_t num_elms = dim[0] * ...;
  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(sb::buildVarRefExp(dim_p),
                       gt->num_dim());
  si::appendStatement(num_elms_decl, body);
  
  // cudaMalloc(&(p->x), sizeof(typeof(p->x)) * dim[i]);
  const SgDeclarationStatementPtrList &members =
      ((SgClassDeclaration*)gt->aux_decl())->
      get_definition()->get_members();
  FOREACH (member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;
    SgType *member_type =
        isSgPointerType(rose_util::GetType(member_decl))
        ->get_base_type();
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(member_type),
            sb::buildVarRefExp(num_elms_decl));
    SgFunctionCallExp *malloc_call = BuildCudaMalloc(
        sb::buildArrowExp(sb::buildVarRefExp(p_decl),
                          sb::buildVarRefExp(member_decl)),
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
  return fdecl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridFreeForUserType(
    GridType *gt) {
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
  return fdecl;
}

// cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
void AppendCUDAMallocHost(SgBasicBlock *body,
                          SgClassDeclaration *type_decl,
                          SgVariableDeclaration *num_elms_decl,
                          SgVariableDeclaration *buf_decl) {
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();

  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;
    SgType *member_type =
        isSgPointerType(rose_util::GetType(member_decl))
        ->get_base_type();
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(member_type),
            sb::buildVarRefExp(num_elms_decl));
    SgFunctionCallExp *malloc_call = BuildCudaMallocHost(
        sb::buildPntrArrRefExp(
            sb::buildVarRefExp(buf_decl),
            sb::buildIntVal(i-1)),
        size_exp);
    si::appendStatement(sb::buildExprStatement(malloc_call),
                        body);
  }
}

void AppendCUDAMemcpy(SgBasicBlock *body,
                      SgClassDeclaration *type_decl,
                      SgVariableDeclaration *num_elms_decl,
                      SgVariableDeclaration *buf_decl,
                      SgVariableDeclaration *dev_decl,
                      bool host_to_dev) {
  
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();

  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;
    SgType *member_type =
        isSgPointerType(rose_util::GetType(member_decl))->
        get_base_type();
    SgExpression *dev_p =
        sb::buildArrowExp(sb::buildVarRefExp(dev_decl),
                          sb::buildVarRefExp(member_decl));
    SgExpression *host_p =
        sb::buildPntrArrRefExp(
            sb::buildVarRefExp(buf_decl),
            sb::buildIntVal(i-1));
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(member_type),
            sb::buildVarRefExp(num_elms_decl));
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
                        SgClassDeclaration *type_decl,
                        SgVariableDeclaration *buf_decl) {
  
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();

  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;
    si::appendStatement(
        sb::buildExprStatement(
            BuildCudaFreeHost(
                sb::buildPntrArrRefExp(
                    sb::buildVarRefExp(buf_decl),
                    sb::buildIntVal(i-1)))),
        body);
  }
}

void AppendTranspose(SgBasicBlock *loop_body,
                     SgClassDeclaration *type_decl,
                     SgVariableDeclaration *soa_decl,
                     SgVariableDeclaration *aos_decl,
                     bool soa_to_aos,
                     SgVariableDeclaration *loop_counter) {
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;    
    SgType *member_type = rose_util::GetType(member_decl);    
    SgExpression *aos_elm =
        sb::buildDotExp(
            sb::buildPntrArrRefExp(
                sb::buildVarRefExp(aos_decl),
                sb::buildVarRefExp(loop_counter)),
            sb::buildVarRefExp(member_decl));
    SgExpression *soa_elm =
        sb::buildPntrArrRefExp(
            sb::buildCastExp(
                sb::buildPntrArrRefExp(
                    sb::buildVarRefExp(soa_decl),
                    sb::buildIntVal(i-1)),
                member_type),
            sb::buildVarRefExp(loop_counter));
    si::appendStatement(
        sb::buildAssignStatement(
            soa_to_aos ? aos_elm : soa_elm,
            soa_to_aos ? soa_elm : aos_elm),
        loop_body);
  }
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridCopyForUserType(
    GridType *gt, bool is_copyout) {
  
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
      BuildNumElmsDecl(p_decl, type_decl, gt->num_dim());
  si::appendStatement(num_elms_decl, body);
  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  AppendCUDAMallocHost(
      body, type_decl, num_elms_decl, tbuf_decl);
  
  if (is_copyout) {
    AppendCUDAMemcpy(
        body, type_decl, num_elms_decl, tbuf_decl,
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

  AppendTranspose(loop_body, type_decl,
                  tbuf_decl, hostp_decl, is_copyout, init);

  if (!is_copyout) {
    AppendCUDAMemcpy(
        body, type_decl, num_elms_decl, tbuf_decl,
        p_decl, true);
  }
    
  AppendCUDAFreeHost(body, type_decl, tbuf_decl);
  
  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildVoidType(), pl);
  rose_util::ReplaceFuncBody(fdecl, body);
  return fdecl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridCopyinForUserType(
    GridType *gt) {
  return BuildGridCopyForUserType(gt, false);
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridCopyoutForUserType(
    GridType *gt) {
  return BuildGridCopyForUserType(gt, true);
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridGetForUserType(
    GridType *gt) {
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

  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();
  SgExprListExp *init_rhs = sb::buildExprListExp();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;
    SgExpression *x = sb::buildPntrArrRefExp(
        sb::buildArrowExp(sb::buildVarRefExp(g_p),
                          sb::buildVarRefExp(member_decl)),
        sb::buildVarRefExp(offset_p));
    si::appendExpression(init_rhs, x);
  }
  SgVariableDeclaration *v_decl =
      sb::buildVariableDeclaration(
          "v", gt->point_type(),
          sb::buildAggregateInitializer(init_rhs));
  si::appendStatement(v_decl, body);

  // return v;
  si::appendStatement(
      sb::buildReturnStmt(sb::buildVarRefExp(v_decl)),
      body);

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, gt->point_type(), pl);
  fdecl->get_functionModifier().setCudaDevice();
  si::setStatic(fdecl);
  rose_util::ReplaceFuncBody(fdecl, body);
  return fdecl;
}

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridEmitForUserType(
    GridType *gt) {
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
  
  // g->x[offset] = v.x;
  // g->y[offset] = v.y;
  // g->z[offset]} = v.z;
  const SgDeclarationStatementPtrList &members =
      type_decl->get_definition()->get_members();
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    if (IsDimMember(member_decl)) continue;
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

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, sb::buildVoidType(), pl);
  fdecl->get_functionModifier().setCudaDevice();
  si::setStatic(fdecl);
  rose_util::ReplaceFuncBody(fdecl, body);
  return fdecl;
}


} // namespace translator
} // namespace physis

