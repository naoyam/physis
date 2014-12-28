// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_CUDA_UTIL_H_
#define PHYSIS_TRANSLATOR_CUDA_UTIL_H_

#include "translator/translator_common.h"

namespace physis {
namespace translator {
namespace cuda_util {

SgFunctionCallExp *BuildCUDADeviceSynchronize(void);
SgFunctionCallExp *BuildCUDAStreamSynchronize(SgExpression *strm);
SgFunctionCallExp *BuildCUDADim3(SgExpression *x, SgExpression *y=NULL,
                                 SgExpression *z=NULL);
SgFunctionCallExp *BuildCUDADim3(int x);
SgFunctionCallExp *BuildCUDADim3(int x, int y);
SgFunctionCallExp *BuildCUDADim3(int x, int y, int z);

SgType *BuildCudaErrorType();
SgFunctionCallExp *BuildCUDAMalloc(SgExpression *buf, SgExpression *size);
SgFunctionCallExp *BuildCUDAFree(SgExpression *p);
SgFunctionCallExp *BuildCUDAMallocHost(SgExpression *buf, SgExpression *size);
SgFunctionCallExp *BuildCUDAFreeHost(SgExpression *p);
SgFunctionCallExp *BuildCUDAMemcpyHostToDevice(SgExpression *dst,
                                               SgExpression *src,
                                               SgExpression *size);
SgFunctionCallExp *BuildCUDAMemcpyDeviceToHost(SgExpression *dst,
                                               SgExpression *src,
                                               SgExpression *size);

enum CudaFuncCache {
  cudaFuncCachePreferNone,
  cudaFuncCachePreferShared,
  cudaFuncCachePreferL1,
  cudaFuncCachePreferEqual
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

SgFunctionCallExp *BuildCudaCallFuncSetCacheConfig(
    SgFunctionSymbol *kernel,
    const CudaFuncCache cache_config);

SgVariableDeclaration *BuildDim3Declaration(const SgName &name,
                                            SgExpression *dimx,
                                            SgExpression *dimy,
                                            SgExpression *dimz,
                                            SgScopeStatement *scope);

SgCudaKernelCallExp *BuildCudaKernelCallExp(SgFunctionRefExp *func_ref,
                                            SgExprListExp *args,
                                            SgCudaKernelExecConfig *config);

SgCudaKernelExecConfig *BuildCudaKernelExecConfig(SgExpression *grid,
                                                  SgExpression *blocks,
                                                  SgExpression *shared = NULL,
                                                  SgExpression *stream = NULL);

SgExpression *BuildCudaIdxExp(const CudaDimentionIdx idx);

} // namespace cuda_util
} // namespace translator
} // namespace physis



#endif /* PHYSIS_TRANSLATOR_CUDA_BUILDER_H_ */
