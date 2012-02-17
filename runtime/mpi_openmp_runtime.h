#ifndef PHYSIS_RUNTIME_MPI_OPENMP_RUNTIME_H_
#define PHYSIS_RUNTIME_MPI_OPENMP_RUNTIME_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_mpi_openmp.h"
#include "runtime/rpc_mpi_openmp.h"

#ifdef USE_OPENMP_NUMA
#define PROCINFO  ProcInfoOpenMP
#define MASTER    MasterOpenMP
#define CLIENT    ClientOpenMP
#define GRIDSPACEMPI  GridSpaceMPIOpenMP
#define GRIDMPI   GridMPIOpenMP
#else
#define PROCINFO  ProcInfo
#define MASTER    Master
#define CLIENT    Client
#define GRIDSPACEMPI  GridSpaceMPI
#define GRIDMPI   GridMPI
#endif

namespace physis {
namespace runtime {

typedef void (*__PSStencilRunClientFunction)(int, void **);
extern __PSStencilRunClientFunction *__PS_stencils;

extern PROCINFO *pinfo;
extern MASTER *master;
extern CLIENT *client;
extern GRIDSPACEMPI *gs;

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_MPI_RUNTIME_H_ */
