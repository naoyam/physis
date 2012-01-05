#ifndef PHYSIS_RUNTIME_MPI_OPENCL_RUNTIME_H_
#define PHYSIS_RUNTIME_MPI_OPENCL_RUNTIME_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_mpi_opencl.h"
#include "runtime/rpc_mpi_opencl.h"
#include "runtime/rpc_opencl_mpi.h"
#include <vector>

#define NUM_CLINFO_BOUNDARY_KERNEL 16

namespace physis {
namespace runtime {

typedef void (*__PSStencilRunClientFunction)(int, void **);
extern __PSStencilRunClientFunction *__PS_stencils;

extern ProcInfo *pinfo;
extern MasterMPIOpenCL *master;
extern ClientMPIOpenCL *client;
extern GridSpaceMPIOpenCL *gs;


} // namespace runtime
} // namespace physis

namespace physis {
namespace runtime {

extern CLMPIinfo *clinfo_generic;
extern CLMPIinfo *clinfo_inner;
extern CLMPIinfo *clinfo_boundary_copy;
extern std::vector<CLMPIinfo *> clinfo_boundary_kernel;

extern CLMPIinfo *clinfo_nowusing;

} // namespace runtime
} // namespace physis

namespace {
void InitOpenCL(
    int my_rank, int num_local_processes, int *argc, char ***argv
);
void DestroyOpenCL(void);
}

#endif /* PHYSIS_RUNTIME_MPI_OPENCL_RUNTIME_H_ */
