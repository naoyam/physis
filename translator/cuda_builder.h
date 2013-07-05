// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_CUDA_BUILDER_H_
#define PHYSIS_TRANSLATOR_CUDA_BUILDER_H_

#include "translator/translator_common.h"

namespace physis {
namespace translator {

SgFunctionCallExp *BuildCudaThreadSynchronize(void);
SgFunctionCallExp *BuildCudaDeviceSynchronize(void);
SgFunctionCallExp *BuildCudaStreamSynchronize(SgExpression *strm);
SgFunctionCallExp *BuildCudaDim3(SgExpression *x, SgExpression *y=NULL,
                                 SgExpression *z=NULL);
SgType *BuildCudaErrorType();
SgFunctionCallExp *BuildCudaMalloc(SgExpression *buf, SgExpression *size);
SgFunctionCallExp *BuildCudaFree(SgExpression *p);
SgFunctionCallExp *BuildCudaMallocHost(SgExpression *buf, SgExpression *size);
SgFunctionCallExp *BuildCudaFreeHost(SgExpression *p);
SgFunctionCallExp *BuildCudaMemcpyHostToDevice(SgExpression *dst,
                                               SgExpression *src,
                                               SgExpression *size);
SgFunctionCallExp *BuildCudaMemcpyDeviceToHost(SgExpression *dst,
                                               SgExpression *src,
                                               SgExpression *size);

} // namespace translator
} // namespace physis



#endif /* PHYSIS_TRANSLATOR_CUDA_BUILDER_H_ */
