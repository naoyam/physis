// Copyright 2010, Tokyo Institute of Technology Matsuoka Lab.
#include "SageBuilderEx.h"
#include "grid.h"
#include "translator/map.h"

namespace physis {
namespace translator {
namespace SageBuilderEx {

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace {
SgClassDeclaration *cuda_dim3_decl() {
  static SgClassDeclaration *dim3_decl = NULL;
  if (!dim3_decl) {
    dim3_decl = sb::buildStructDeclaration(SgName("dim3"), NULL);
    si::appendStatement(
        sb::buildVariableDeclaration("x", sb::buildIntType()),
        dim3_decl->get_definition());
    si::appendStatement(
        sb::buildVariableDeclaration("y", sb::buildIntType()),
        dim3_decl->get_definition());
    si::appendStatement(
        sb::buildVariableDeclaration("z", sb::buildIntType()),
        dim3_decl->get_definition());
  }
  return dim3_decl;
}

SgClassDefinition *cuda_dim3_def() {
  static SgClassDefinition *dim3_def = NULL;
  if (!dim3_def) {
    dim3_def = sb::buildClassDefinition(cuda_dim3_decl());
  }
  return dim3_def;
}

SgMemberFunctionDeclaration *cuda_dim3_ctor_decl() {
  static SgMemberFunctionDeclaration *ctor_decl = NULL;
  if (!ctor_decl) {
    /*
    ctor_decl = buildMemberFunctionDeclaration(SgName("dim3"),
                                               NULL,
                                               NULL,
                                               cuda_dim3_decl());
    */
    ctor_decl = sb::buildNondefiningMemberFunctionDeclaration(
        SgName("dim3"), sb::buildVoidType(),
        sb::buildFunctionParameterList(),
        cuda_dim3_decl()->get_definition());
    //si::appendStatement(ctor_decl, cuda_dim3_decl()->get_definition());
  }
  return ctor_decl;
}

SgEnumDeclaration *cuda_enum_cuda_func_cache() {
  static SgEnumDeclaration *enum_func_cache = NULL;
  static const char *enum_names[] = {"cudaFuncCachePreferNone",
                                     "cudaFuncCachePreferShared",
                                     "cudaFuncCachePreferL1"};
  if (!enum_func_cache) {
    enum_func_cache =
        sb::buildEnumDeclaration(SgName("cudaFuncCache"), sb::topScopeStack());
    SgInitializedNamePtrList &members = enum_func_cache->get_enumerators();
    for (int i = 0; i < 3; i++) {
      SgInitializedName *in =
          sb::buildInitializedName(
              SgName(enum_names[i]), enum_func_cache->get_type());
      members.push_back(in);
      in->set_parent(enum_func_cache);
      in->set_scope(sb::topScopeStack());
    }
  }
  return enum_func_cache;
}

SgFunctionDeclaration *cuda_func_set_cache_config() {
  SgFunctionDeclaration *func_set_cache_config = NULL;
  if (!func_set_cache_config) {
    SgFunctionParameterTypeList *params =
        sb::buildFunctionParameterTypeList(sb::buildIntType());
    func_set_cache_config =
        sb::buildNondefiningFunctionDeclaration(
            SgName("cudaFuncSetCacheConfig"),
            cuda_enum_cuda_func_cache()->get_type(),
            sb::buildFunctionParameterList(params),
            sb::topScopeStack());
  }
  return func_set_cache_config;
}
}  // namespace
/*
SgMemberFunctionDeclaration *buildMemberFunctionDeclaration(
    const SgName &name,
    SgFunctionType *type,
    SgFunctionDefinition *definition,
    SgClassDeclaration *class_decl) {
  SgMemberFunctionDeclaration *func_decl =
      new SgMemberFunctionDeclaration(name, type, definition);
  ROSE_ASSERT(func_decl);

  if (class_decl) {
    func_decl->set_associatedClassDeclaration(class_decl);
    SgClassDefinition *class_def = class_decl->get_definition();
    if (class_def) {
      si::appendStatement(func_decl, class_def);
    }
  }
  si::setOneSourcePositionForTransformation(func_decl);
  return func_decl;
}
*/
SgFunctionCallExp *buildCudaCallFuncSetCacheConfig(
    SgFunctionSymbol *kernel,
    const cudaFuncCache cache_config) {
  ROSE_ASSERT(kernel);
  SgEnumVal *enum_val = buildEnumVal(cache_config,
                                     cuda_enum_cuda_func_cache());
  SgExprListExp *args =
      sb::buildExprListExp(sb::buildFunctionRefExp(kernel), enum_val);

  // build a call to grid_new
  SgFunctionCallExp *call =
      sb::buildFunctionCallExp(
          sb::buildFunctionRefExp(cuda_func_set_cache_config()),
          args);
  return call;
}

SgVariableDeclaration *buildDim3Declaration(const SgName &name,
                                            SgExpression *dimx,
                                            SgExpression *dimy,
                                            SgExpression *dimz,
                                            SgScopeStatement *scope) {
  SgClassType *dim3_type = cuda_dim3_decl()->get_type();
  SgExprListExp *expr_list = sb::buildExprListExp();
  si::appendExpression(expr_list, dimx);
  si::appendExpression(expr_list, dimy);
  si::appendExpression(expr_list, dimz);  
  SgConstructorInitializer *init =
      sb::buildConstructorInitializer(cuda_dim3_ctor_decl(),
                                      expr_list,
                                      sb::buildVoidType(),
                                      false, false, true, false);
  SgVariableDeclaration *ret =
      sb::buildVariableDeclaration(name, dim3_type, init, scope);
  return ret;
}

SgCudaKernelCallExp *buildCudaKernelCallExp(SgFunctionRefExp *func_ref,
                                            SgExprListExp *args,
                                            SgCudaKernelExecConfig *config) {
  ROSE_ASSERT(func_ref);
  ROSE_ASSERT(args);
  ROSE_ASSERT(config);

  SgCudaKernelCallExp *cuda_call =
      new SgCudaKernelCallExp(func_ref, args, config);
  ROSE_ASSERT(cuda_call);

  func_ref->set_parent(cuda_call);
  args->set_parent(cuda_call);
  si::setOneSourcePositionForTransformation(cuda_call);
  return cuda_call;
}

SgEnumVal* buildEnumVal(unsigned int value, SgEnumDeclaration* decl) {
  ROSE_ASSERT(decl);
  SgInitializedNamePtrList &members = decl->get_enumerators();
  ROSE_ASSERT(value < members.size());
  SgInitializedName *name = members[value];
  SgEnumVal* enum_val= sb::buildEnumVal_nfi(value, decl, name->get_name());
  ROSE_ASSERT(enum_val);
  si::setOneSourcePositionForTransformation(enum_val);
  return enum_val;
}

SgCudaKernelExecConfig *buildCudaKernelExecConfig(SgExpression *grid,
                                                  SgExpression *blocks,
                                                  SgExpression *shared,
                                                  SgExpression *stream) {
  if (stream != NULL && shared == NULL) {
    // Unless shared is given non-null value, stream parameter will be
    // ignored. 
    shared = sb::buildIntVal(0);
  }
  SgCudaKernelExecConfig *cuda_config =
      sb::buildCudaKernelExecConfig_nfi(grid, blocks, shared, stream);
  ROSE_ASSERT(cuda_config);
  si::setOneSourcePositionForTransformation(cuda_config);
  return cuda_config;
}

SgExpression *buildGridDimVarExp(SgExpression *grid_var, int dim) {
  ROSE_ASSERT(grid_var);
  SgExpression *dim_var_exp =
      sb::buildPntrArrRefExp(
          sb::buildDotExp(
              grid_var,
              sb::buildVarRefExp(SgName("dim"))),
          sb::buildIntVal(dim));
  return dim_var_exp;
}

// WARNING: Do not use this; instead refer to the runtime builder
// classes. This is Nomura's original, and uses dimension of the grid
// that is found first in the argument list. That is not correct. CUDA
// grid dimension must be the stencil's domain.
SgExpression *buildStencilDimVarExp(StencilMap *stencil,
                                    SgExpression *stencil_var,
                                    int dim) {
  ROSE_ASSERT(stencil);
  SgClassDefinition *param_struct = stencil->GetStencilTypeDefinition();
  ROSE_ASSERT(param_struct);

  SgExpression *dim_var_exp = NULL;

  // Enumerate members of parameter struct
  const SgDeclarationStatementPtrList &members = param_struct->get_members();
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    if (GridType::isGridType(vars[0]->get_type())) {
      dim_var_exp =
          sb::buildPntrArrRefExp(
              sb::buildArrowExp(
                  sb::buildDotExp(stencil_var,
                                  sb::buildVarRefExp(member_decl)),
                  sb::buildVarRefExp(SgName("dim"))),
              sb::buildIntVal(dim));
      break;
    }
  }
  return dim_var_exp;
}

SgExpression *buildCudaIdxExp(const CudaDimentionIdx idx) {
  static SgVariableDeclaration *threadIdx = NULL;  
  static SgVariableDeclaration *blockIdx = NULL;
  static SgVariableDeclaration *blockDim = NULL;  
  if (!blockIdx) {
    threadIdx = sb::buildVariableDeclaration("threadIdx",
                                            cuda_dim3_decl()->get_type());
    blockIdx = sb::buildVariableDeclaration("blockIdx",
                                            cuda_dim3_decl()->get_type());
    blockDim = sb::buildVariableDeclaration("blockDim",
                                            cuda_dim3_decl()->get_type());
  }
  SgVarRefExp *var = NULL;
  SgVarRefExp *xyz = NULL;
  switch (idx) {
    case kBlockDimX:
    case kBlockDimY:
    case kBlockDimZ:
      var = sb::buildVarRefExp(blockDim);
      break;
    case kBlockIdxX:
    case kBlockIdxY:
    case kBlockIdxZ:
      var = sb::buildVarRefExp(blockIdx);
      break;
    case kThreadIdxX:
    case kThreadIdxY:
    case kThreadIdxZ:
      var = sb::buildVarRefExp(threadIdx);
      break;
    default:
      ROSE_ASSERT(false);
  }
  switch (idx) {
    case kBlockDimX:
    case kBlockIdxX:
    case kThreadIdxX:
      xyz = sb::buildVarRefExp("x");
      break;
    case kBlockDimY:
    case kBlockIdxY:
    case kThreadIdxY:
      xyz = sb::buildVarRefExp("y");
      break;
    case kBlockDimZ:
    case kBlockIdxZ:
    case kThreadIdxZ:
      xyz = sb::buildVarRefExp("z");
      break;
  }
  return sb::buildDotExp(var, xyz);
}

} // namespace SageBuilderEx
} // namespace translator
} // namespace physis
