// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_SAGEBUILDEREX_H_
#define PHYSIS_TRANSLATOR_SAGEBUILDEREX_H_

#include <rose.h>

#include "translator/map.h"

namespace physis {
namespace translator {
namespace SageBuilderEx {

// --- Extention of SageBuilder ---
SgEnumVal* buildEnumVal(unsigned int value, SgEnumDeclaration* decl);

SgMemberFunctionDeclaration *buildMemberFunctionDeclaration(
    const SgName &name,
    SgFunctionType *type,
    SgFunctionDefinition *definition,
    SgClassDeclaration *class_decl);

SgExpression *buildStencilDimVarExp(StencilMap *stencil,
                                    SgExpression *stencil_var,
                                    int dim);

// --- CUDA support ---
enum cudaFuncCache {
  cudaFuncCachePreferNone,
  cudaFuncCachePreferShared,
  cudaFuncCachePreferL1
};

enum CudaDimentionIdx {
  kBlockDimX,
  kBlockDimY,
  kBlockDimZ,
  kBlockIdxX,
  kBlockIdxY,
  kBlockIdxZ,
  kThreadIdxX,
  kThreadIdxY,
  kThreadIdxZ
};

SgFunctionCallExp *buildCudaCallFuncSetCacheConfig(
    SgFunctionSymbol *kernel,
    const cudaFuncCache cache_config);

SgVariableDeclaration *buildDim3Declaration(const SgName &name,
                                            SgExpression *dimx,
                                            SgExpression *dimy,
                                            SgExpression *dimz,
                                            SgScopeStatement *scope);

SgCudaKernelCallExp *buildCudaKernelCallExp(SgFunctionRefExp *func_ref,
                                            SgExprListExp *args,
                                            SgCudaKernelExecConfig *config);

SgCudaKernelExecConfig *buildCudaKernelExecConfig(SgExpression *grid,
                                                  SgExpression *blocks,
                                                  SgExpression *shared = NULL,
                                                  SgExpression *stream = NULL);

SgExpression *buildCudaIdxExp(const CudaDimentionIdx idx);

} // namespace SageBuilderEx
} // namespace translator
} // namespace physis

#endif  /* SAGEBUILDEREX_H_ */
