// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_MPI_RUNTIME_H_
#define PHYSIS_RUNTIME_MPI_RUNTIME_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_mpi.h"
#include "runtime/rpc_mpi.h"

namespace physis {
namespace runtime {

typedef void (*__PSStencilRunClientFunction)(int, void **);
extern __PSStencilRunClientFunction *__PS_stencils;

extern ProcInfo *pinfo;
extern Master *master;
extern Client *client;
extern GridSpaceMPI *gs;

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_MPI_RUNTIME_H_ */
