#ifndef PHYSIS_RUNTIME_MPI_RUNTIME_H_
#define PHYSIS_RUNTIME_MPI_RUNTIME_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_mpi_openmp.h"
#include "runtime/rpc_mpi_openmp.h"

namespace physis {
namespace runtime {

typedef void (*__PSStencilRunClientFunction)(int, void **);
extern __PSStencilRunClientFunction *__PS_stencils;

extern ProcInfoOpenMP *pinfo;
extern MasterOpenMP *master;
extern ClientOpenMP *client;
extern GridSpaceMPIOpenMP *gs;

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_MPI_RUNTIME_H_ */
