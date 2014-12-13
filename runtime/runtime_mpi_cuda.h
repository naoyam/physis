// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_MPI_CUDA_H_
#define PHYSIS_RUNTIME_RUNTIME_MPI_CUDA_H_

#include "runtime/runtime_mpi.h"
#include "runtime/grid_mpi_cuda_exp.h"
#include "runtime/grid_space_mpi_cuda.h"
#include "runtime/rpc_cuda.h"

namespace physis {
namespace runtime {

void InitCUDA(int my_rank, int num_local_processes);
int GetNumberOfLocalProcesses(int *argc, char ***argv);

template <class GridSpaceType>
class RuntimeMPICUDA: public RuntimeMPI<GridSpaceType> {
public:
  RuntimeMPICUDA() {}
  virtual ~RuntimeMPICUDA() {}
 protected:
  virtual void InitGridSpace(int *argc, char ***argv,
                             int domain_rank, const IndexArray &domain_size,
                             int proc_num_dims, const IndexArray &proc_size,
                             InterProcComm &ipc);
  virtual void InitRPC(InterProcComm *ipc);
};

template <class GridSpaceType>
void RuntimeMPICUDA<GridSpaceType>::InitGridSpace(int *argc, char ***argv,
                                                  int domain_rank, const IndexArray &domain_size,
                                                  int proc_num_dims, const IndexArray &proc_size,
                                                  InterProcComm &ipc) {
  int num_local_processes = GetNumberOfLocalProcesses(argc, argv);
  InitCUDA(ipc.GetRank(), num_local_processes);
  this->gs_ = new GridSpaceType(domain_rank, domain_size, proc_num_dims,
                                proc_size, ipc);
  LOG_INFO() << "Grid space: " << *this->gs_ << "\n";
}

template <class GridSpaceType>
void RuntimeMPICUDA<GridSpaceType>::InitRPC(InterProcComm *ipc) {
  if (ipc->GetRank() == Proc::GetRootRank()) {  
    LOG_DEBUG() << "I'm the master.\n";
    this->proc_ = new MasterCUDA<GridSpaceType>(ipc, this->client_funcs_, this->gs_);
    LOG_INFO() << *this->proc_ << "\n";     
  } else {    
    LOG_DEBUG() << "I'm a client.\n";
    this->proc_ = new ClientCUDA<GridSpaceType>(ipc, this->client_funcs_, this->gs_);
    LOG_INFO() << *this->proc_ << "\n";
  }
}


} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RUNTIME_MPI_CUDA_H_ */

