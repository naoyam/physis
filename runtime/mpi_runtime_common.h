// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_MPI_RUNTIME_COMMON_H_
#define PHYSIS_RUNTIME_MPI_RUNTIME_COMMON_H_

#include "runtime/runtime_common.h"
#include "runtime/rpc_mpi.h"
#include "runtime/grid_mpi.h"

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
