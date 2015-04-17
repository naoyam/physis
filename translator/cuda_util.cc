// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/cuda_util.h"
#include "translator/rose_util.h"

namespace sb = SageBuilder;
namespace si = SageInterface;
namespace ru = physis::translator::rose_util;

namespace physis {
namespace translator {
namespace cuda_util {

SgFunctionCallExp *BuildCUDADeviceSynchronize(void) {
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(
      "cudaDeviceSynchronize", sb::buildVoidType());
  return fc;
}

SgFunctionCallExp *BuildCUDADim3(SgExpression *x, SgExpression *y,
                                 SgExpression *z) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("dim3");
  SgExprListExp *args = sb::buildExprListExp(x, y, z);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(fs, args);
  return call;
}

SgFunctionCallExp *BuildCUDADim3(int x) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("dim3");
  SgExprListExp *args = sb::buildExprListExp(Int(x));
  SgFunctionCallExp *call = sb::buildFunctionCallExp(fs, args);
  return call;
}

SgFunctionCallExp *BuildCUDADim3(int x, int y) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("dim3");
  SgExprListExp *args = sb::buildExprListExp(Int(x), Int(y));
  SgFunctionCallExp *call = sb::buildFunctionCallExp(fs, args);
  return call;
}

SgFunctionCallExp *BuildCUDADim3(int x, int y, int z) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("dim3");
  SgExprListExp *args = sb::buildExprListExp(Int(x), Int(y), Int(z));
  SgFunctionCallExp *call = sb::buildFunctionCallExp(fs, args);
  return call;
}

SgFunctionCallExp *BuildCUDAStreamSynchronize(SgExpression *strm) {
  SgExprListExp *args = sb::buildExprListExp(strm);
  SgFunctionCallExp *call = sb::buildFunctionCallExp("cudaStreamSynchronize",
                                                     sb::buildVoidType(),
                                                     args);
  return call;
}

SgType *BuildCudaErrorType() {
  return si::lookupNamedTypeInParentScopes("cudaError_t");
}

SgFunctionCallExp *BuildCUDAMalloc(SgExpression *buf, SgExpression *size) {
  SgExprListExp *args = sb::buildExprListExp(
      sb::buildCastExp(sb::buildAddressOfOp(buf),
                       sb::buildPointerType(
                           sb::buildPointerType(sb::buildVoidType()))),
      size);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaMalloc",
      BuildCudaErrorType(),
      args);
  return call;
}

SgFunctionCallExp *BuildCUDAFree(SgExpression *p) {
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaFree",
      BuildCudaErrorType(),
      sb::buildExprListExp(p));
  return call;
}

SgFunctionCallExp *BuildCUDAMallocHost(SgExpression *buf, SgExpression *size) {
  SgExprListExp *args = sb::buildExprListExp(
      sb::buildCastExp(sb::buildAddressOfOp(buf),
                       sb::buildPointerType(
                           sb::buildPointerType(sb::buildVoidType()))),
      size);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaMallocHost",
      BuildCudaErrorType(),
      args);
  return call;
}

SgFunctionCallExp *BuildCUDAFreeHost(SgExpression *p) {
  SgFunctionCallExp *call = sb::buildFunctionCallExp(
      "cudaFreeHost",
      BuildCudaErrorType(),
      sb::buildExprListExp(p));
  return call;
}

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
  if (!enum_func_cache) {
    SgEnumSymbol *es = si::lookupEnumSymbolInParentScopes("cudaFuncCache");
    PSAssert(es);
    enum_func_cache = isSgEnumDeclaration(es->get_declaration()->get_definingDeclaration());
  }
  return enum_func_cache;
}

SgEnumDeclaration *cuda_enum_cuda_memcpy_kind() {
  static SgEnumDeclaration *enum_memcpy_kind = NULL;
  if (!enum_memcpy_kind) {
    SgEnumSymbol *es = si::lookupEnumSymbolInParentScopes("cudaMemcpyKind");
    PSAssert(es);
    enum_memcpy_kind = isSgEnumDeclaration(es->get_declaration()->get_definingDeclaration());    
  }
  return enum_memcpy_kind;
}

}  // namespace

SgFunctionCallExp *BuildCudaCallFuncSetCacheConfig(
    SgFunctionSymbol *kernel,
    const CudaFuncCache cache_config) {
  ROSE_ASSERT(kernel);
  SgEnumVal *enum_val = ru::BuildEnumVal(
      cache_config, cuda_enum_cuda_func_cache());
  
  SgExprListExp *args =
      sb::buildExprListExp(sb::buildFunctionRefExp(kernel), enum_val);

  // build a call to cudaFuncSetCacheConfig
  SgFunctionSymbol *fs =
      si::lookupFunctionSymbolInParentScopes("cudaFuncSetCacheConfig");
  PSAssert(fs);
  SgFunctionCallExp *call =
      sb::buildFunctionCallExp(fs, args);
  return call;
}

SgFunctionCallExp *BuildCUDAMemcpy(
    SgExpression *dst,
    SgExpression *src,
    SgExpression *size,
    CUDAMemcpyKind kind) {
  SgEnumVal *enum_val = ru::BuildEnumVal(
      kind, cuda_enum_cuda_memcpy_kind());
  
  SgExprListExp *args =
      sb::buildExprListExp(dst, src, size, enum_val);

  // build a call to cudaMemcpy
  SgFunctionSymbol *fs =
      si::lookupFunctionSymbolInParentScopes("cudaMemcpy");
  PSAssert(fs);
  SgFunctionCallExp *call =
      sb::buildFunctionCallExp(fs, args);
  return call;
}

SgVariableDeclaration *BuildDim3Declaration(const SgName &name,
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

SgCudaKernelCallExp *BuildCudaKernelCallExp(SgFunctionRefExp *func_ref,
                                            SgExprListExp *args,
                                            SgCudaKernelExecConfig *config) {
  ROSE_ASSERT(func_ref);
  ROSE_ASSERT(args);
  ROSE_ASSERT(config);

#if 1
  // This version works both with ROSE edg3 and edg4x
  SgCudaKernelCallExp *cuda_call =
      sb::buildCudaKernelCallExp_nfi(func_ref, args, config);
  ROSE_ASSERT(cuda_call);
  si::setOneSourcePositionForTransformation(cuda_call);
#else
  // This version works with ROSE edg3, but not with edg4x
  SgCudaKernelCallExp *cuda_call =
      new SgCudaKernelCallExp(func_ref, args, config);
  ROSE_ASSERT(cuda_call);
  func_ref->set_parent(cuda_call);
  args->set_parent(cuda_call);
  si::setOneSourcePositionForTransformation(cuda_call);
#endif
  return cuda_call;
}

SgCudaKernelExecConfig *BuildCudaKernelExecConfig(
    SgExpression *grid,
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

SgExpression *BuildCudaIdxExp(const CudaDimentionIdx idx) {
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


void SetCUDAKernel(SgFunctionDeclaration *func) {
  LOG_DEBUG() << "Make function, "
              << func->get_name() << ", a CUDA kernel function\n";
  SgFunctionDeclaration *decl;
  decl = isSgFunctionDeclaration(func->get_definingDeclaration());
  decl->get_functionModifier().setCudaKernel();
  // With Edg4, there is a non-defining declaration, and it needs to
  // be marked as a cuda kernel. With Edg3, there is no non-defining declaration.
  if (func->get_firstNondefiningDeclaration()) {
    decl = isSgFunctionDeclaration(func->get_firstNondefiningDeclaration());
    decl->get_functionModifier().setCudaKernel();
  }
}

void SetCUDADevice(SgFunctionDeclaration *func) {
  LOG_DEBUG() << "Make function, "
              << func->get_name() << ", a CUDA device function\n";
  SgFunctionDeclaration *decl;
  decl = isSgFunctionDeclaration(func->get_definingDeclaration());
  decl->get_functionModifier().setCudaDevice();
  // With Edg4, there is a non-defining declaration, and it needs to
  // be marked as a cuda kernel. With Edg3, there is no non-defining declaration.
  if (func->get_firstNondefiningDeclaration()) {
    decl = isSgFunctionDeclaration(func->get_firstNondefiningDeclaration());
    decl->get_functionModifier().setCudaDevice();
  }
}

} // namespace cuda_util
} // namespace translator
} // namespace physis
