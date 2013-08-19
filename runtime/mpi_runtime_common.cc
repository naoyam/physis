// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include <stdarg.h>
#include <map>
#include <string>

#include "runtime/mpi_runtime_common.h"

#include "mpi.h"

#include "physis/physis_mpi.h"
#include "physis/physis_util.h"
#include "runtime/grid_mpi_debug_util.h"
#include "runtime/mpi_util.h"
#include "runtime/grid_mpi2.h"
#include "runtime/rpc2.h"
#include "runtime/inter_proc_comm_mpi.h"

using std::map;
using std::string;

using namespace physis::runtime;

using physis::IndexArray;
using physis::IntArray;
using physis::SizeArray;

namespace physis {
namespace runtime {

__PSStencilRunClientFunction *__PS_stencils;

ProcInfo *pinfo;
Master *master;
Client *client;
GridSpaceMPI *gs;

} // namespace runtime
} // namespace physis

#ifdef __cplusplus
extern "C" {
#endif




#ifdef __cplusplus
}
#endif

