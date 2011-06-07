// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_MPI_CUDA_RUNTIME_H_
#define PHYSIS_RUNTIME_MPI_CUDA_RUNTIME_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_mpi_cuda.h"
#include "runtime/rpc_mpi_cuda.h"

namespace physis {
namespace runtime {

typedef void (*__PSStencilRunClientFunction)(int, void **);
extern __PSStencilRunClientFunction *__PS_stencils;

// are these necessary to be exported here? if they are only used in
// mpi_cuda_runtime.cc, declaration at that file is enough.
extern ProcInfo *pinfo;
extern MasterMPICUDA *master;
extern ClientMPICUDA *client;
extern GridSpaceMPICUDA *gs;


} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_MPI_CUDA_RUNTIME_H_ */
