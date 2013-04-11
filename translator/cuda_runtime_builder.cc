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
    SgExpressionPtrList *offset_exprs,
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

SgClassDeclaration *CUDARuntimeBuilder::BuildGridDevType(
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
  si::appendStatement(sb::buildVariableDeclaration("dim", dim_type),
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

SgVariableDeclaration *BuildNumElmsDecl(SgExpression *dim,
                                        int num_dims) {
  SgExpression *num_elms_rhs =
      sb::buildPntrArrRefExp(dim, sb::buildIntVal(0));
  for (int i = 1; i < num_dims; ++i) {
    num_elms_rhs =
        sb::buildMultiplyOp(
            num_elms_rhs,
            sb::buildPntrArrRefExp(
                si::copyExpression(dim),
                sb::buildIntVal(i)));
  }

  SgVariableDeclaration *num_elms_decl =
      sb::buildVariableDeclaration(
          "num_elms", si::lookupNamedTypeInParentScopes("size_t"),
          sb::buildAssignInitializer(num_elms_rhs));
  return num_elms_decl;
}

// Build a "new" function.
SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridNew(
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
          "dim",
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
                          sb::buildVarRefExp("dim")),
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
    SgName member_name = rose_util::GetName(member_decl);
    if (member_name == "dim") continue;
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


SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridFree(
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
    SgName member_name = rose_util::GetName(member_decl);
    if (member_name == "dim") continue;
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

SgFunctionDeclaration *CUDARuntimeBuilder::BuildGridCopyin(
    GridType *gt) {
  /*
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
  string func_name = dev_type->get_name() + "Copyin";
  SgType *ret_type = sb::buildVoidType();
  SgType *dev_ptr_type = sb::buildPointerType(dev_type);
  int num_point_elms = gt->point_def()->get_members().size();
  const SgDeclarationStatementPtrList &members =
      ((SgClassDeclaration*)gt->aux_decl())->
      get_definition()->get_members();
  
  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();

  // void *v
  SgInitializedName *v_p =
      sb::buildInitializedName("v", sb::buildPointerType(sb::buildVoidType()));
  si::appendArg(pl, v_p);
  // const void *src
  SgInitializedName *src_p =
      sb::buildInitializedName(
          "src",
          sb::buildPointerType(sb::buildConstType(sb::buildVoidType())));
  si::appendArg(pl, src_p);

  // Function body
  SgBasicBlock *body = sb::buildBasicBlock();
  
  SgVariableDeclaration *p_decl =
      sb::buildVariableDeclaration(
          "p", dev_ptr_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(sb::buildVarRefExp(v_p),
                               dev_ptr_type)));
  si::appendStatement(p_decl, body);
  //const Type *srcp = (const Type *)src;
  SgType *srcp_type =
      sb::buildPointerType(sb::buildConstType(gt->point_type()));
  SgVariableDeclaration *srcp_decl =
      sb::buildVariableDeclaration(
          "srcp", srcp_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(sb::buildVarRefExp(src_p),
                               srcp_type)));
  si::appendStatement(srcp_decl, body);

  SgExpression *dim_expr =
      sb::buildArrowExp(
          sb::buildVarRefExp(p_decl),
          sb::buildVarRefExp(
              isSgVariableDeclaration(members[0])));

  // void *tbuf[3];
  SgVariableDeclaration *tbuf_decl =
      sb::buildVariableDeclaration(
          "tbuf",
          sb::buildArrayType(sb::buildPointerType(sb::buildVoidType()),
                             sb::buildIntVal(num_point_elms)));
  si::appendStatement(tbuf_decl, body);


  // size_t num_elms = dim[0] * ...;
  SgVariableDeclaration *num_elms_decl =
      BuildNumElmsDecl(
          si::copyExpression(dim_expr),
          gt->num_dim());
  si::appendStatement(num_elms_decl, body);

  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgName member_name = rose_util::GetName(member_decl);
    if (member_name == "dim") continue;
    SgType *member_type =
        isSgPointerType(rose_util::GetType(member_decl))
        ->get_base_type();
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(member_type),
            sb::buildVarRefExp(num_elms_decl));
    SgFunctionCallExp *malloc_call = BuildCudaMallocHost(
        sb::buildPntrArrRefExp(
            sb::buildVarRefExp(tbuf_decl),
            sb::buildIntVal(i-1)),
        size_exp);
    si::appendStatement(sb::buildExprStatement(malloc_call),
                        body);
    
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

  // Type v = srcp[i];
  SgExpression *srcpi =
      sb::buildPntrArrRefExp(sb::buildVarRefExp(srcp_decl),
                             sb::buildVarRefExp(init));
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgName member_name = rose_util::GetName(member_decl);
    if (member_name == "dim") continue;
    // tbuf[i] = srcp[i].member
    si::appendStatement(
        sb::buildAssignStatement(
            sb::buildPntrArrRefExp(
                sb::buildPntrArrRefExp(
                    sb::buildVarRefExp(tbuf_decl),
                    sb::buildIntVal(i-1)),
                sb::buildVarRefExp(init)),
            sb::buildDotExp(si::copyExpression(srcpi),
                            sb::buildVarRefExp(member_decl))),
        loop_body);
  }
    
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgName member_name = rose_util::GetName(member_decl);
    SgType *member_type = rose_util::GetType(member_decl);    
    if (member_name == "dim") continue;
    SgExpression *dst =
        sb::buildArrowExp(sb::buildVarRefExp(p_decl),
                          sb::buildVarRefExp(member_decl));
    SgExpression *src =
        sb::buildPntrArrRefExp(
            sb::buildVarRefExp(tbuf_decl),
            sb::buildIntVal(i-1));
    SgExpression *size_exp =
        sb::buildMultiplyOp(
            sb::buildSizeOfOp(member_type),
            sb::buildVarRefExp(num_elms_decl));
    si::appendStatement(
        sb::buildExprStatement(
            BuildCudaMemcpyHostToDevice(dst, src,
                                        size_exp)),
        body);
    si::appendStatement(
        sb::buildExprStatement(
            BuildCudaFreeHost(si::copyExpression(src))),
        body);
  }
  
  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name, ret_type, pl);
  rose_util::ReplaceFuncBody(fdecl, body);
  return fdecl;
}
} // namespace translator
} // namespace physis

