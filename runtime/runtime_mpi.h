// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_MPI_H_
#define PHYSIS_RUNTIME_RUNTIME_MPI_H_

#include <stdarg.h>

#include "runtime/runtime.h"
#include "runtime/grid_mpi.h"
#include "runtime/grid_space_mpi.h"
#include "runtime/proc.h"
#include "runtime/rpc.h"
#include "runtime/ipc_mpi.h"

namespace physis {
namespace runtime {

template <class GridSpaceType>
class RuntimeMPI: public Runtime<GridSpaceType> {
 public:
  RuntimeMPI();
  virtual ~RuntimeMPI();
  virtual void Init(int *argc, char ***argv, int grid_num_dims,
                    va_list vl);
  virtual __PSStencilRunClientFunction *client_funcs() {
    return client_funcs_;
  }
  virtual Proc *proc() {
    return proc_;
  }
  virtual int IsMaster() {
    return proc_->rank() == Proc::GetRootRank();
  }
  void Listen();
  
 protected:
  __PSStencilRunClientFunction *client_funcs_;
  Proc *proc_;

  virtual void InitDomainSize(int domain_rank, va_list vl,
                              IndexArray &domain_size);
  virtual InterProcComm *GetIPC(int *argc, char ***argv);
  virtual void InitStencilFuncs(va_list vl);
  virtual void InitProcSize(int *argc, char ***argv,
                            int domain_rank, int num_ipc_procs,
                            IndexArray &proc_size, int &proc_num_dims);
  
  virtual void InitGridSpace(int *argc, char ***argv,
                             int domain_rank, const IndexArray &domain_size,
                             int proc_num_dims, const IndexArray &proc_size,
                             InterProcComm &ipc);
  virtual void InitRPC(InterProcComm *ipc);
  
};

template <class GridSpaceType>
RuntimeMPI<GridSpaceType>::RuntimeMPI(): Runtime<GridSpaceType>() {
}

template <class GridSpaceType>
RuntimeMPI<GridSpaceType>::~RuntimeMPI() {
}

template <class GridSpaceType>
void RuntimeMPI<GridSpaceType>::Init(int *argc, char ***argv, int domain_rank,
                                     va_list vl) {
  Runtime<GridSpaceType>::Init(argc, argv, domain_rank, vl);
  
  IndexArray domain_size;
  InitDomainSize(domain_rank, vl, domain_size);

  InitStencilFuncs(vl);  

  InterProcComm *ipc = GetIPC(argc, argv);
  
  IntArray proc_size;
  int proc_num_dims;
  InitProcSize(argc, argv, domain_rank, ipc->GetNumProcs(),
               proc_size, proc_num_dims);

  InitGridSpace(argc, argv, domain_rank, domain_size, proc_num_dims,
                proc_size, *ipc);
      
  InitRPC(ipc);  
}

template <class GridSpaceType>
void RuntimeMPI<GridSpaceType>::InitDomainSize(int domain_rank, va_list vl,
                                               IndexArray &domain_size) {
  for (int i = 0; i < domain_rank; ++i) {
    domain_size[i] = va_arg(vl, PSIndex);
  }
}

template <class GridSpaceType>
InterProcComm *RuntimeMPI<GridSpaceType>::GetIPC(int *argc, char ***argv) {
  InterProcComm *ipc = InterProcCommMPI::GetInstance();
  ipc->Init(argc, argv);
  return ipc;
}

template <class GridSpaceType>
void RuntimeMPI<GridSpaceType>::InitGridSpace(int *argc, char ***argv, int domain_rank,
                                              const IndexArray &domain_size,
                                              int proc_num_dims, const IndexArray &proc_size,
                                              InterProcComm &ipc) {
  this->gs_ = new GridSpaceType(domain_rank, domain_size,
                                proc_num_dims, proc_size, ipc);
  //LOG_INFO() << "Grid space: " << *gs_ << "\n";
}

template <class GridSpaceType>
void RuntimeMPI<GridSpaceType>::InitProcSize(int *argc, char ***argv,
                                             int domain_rank, int num_ipc_procs,
                                             IntArray &proc_size, int &proc_num_dims) {
  proc_size.Set(1);
  proc_num_dims = GetProcessDim(argc, argv, proc_size);
  if (proc_num_dims > 0) {
    int num_procs = proc_size.accumulate(proc_num_dims);
    if (num_procs != num_ipc_procs) {    
      LOG_ERROR() << "Number of process dictated by process dimension"
          " does not match with the number of processes launched by"
          " the underlying multi-processing runtime.\n";
      PSAbort(1);
    }
    if (proc_num_dims > domain_rank) {
      LOG_ERROR() << "Rank of domain must not exceed that of processes\n";
      PSAbort(1);
    }
  } else {
    // 1-D process decomposition by default
    LOG_INFO() <<
        "No process dimension specified; defaulting to 1D decomposition\n";
    proc_num_dims = domain_rank;
    proc_size[proc_num_dims-1] = num_ipc_procs;
  }
  
  LOG_INFO() << "Number of process dimensions: " << proc_num_dims << "\n";
  LOG_INFO() << "Process size: " << proc_size << "\n";
}

// Set the stencil client functions
template <class GridSpaceType>
void RuntimeMPI<GridSpaceType>::InitStencilFuncs(va_list vl) {
  int num_stencil_run_calls = va_arg(vl, int);
  __PSStencilRunClientFunction *stencil_funcs
      = va_arg(vl, __PSStencilRunClientFunction*);
  
  client_funcs_ = new __PSStencilRunClientFunction[num_stencil_run_calls];
  for (int i = 0; i < num_stencil_run_calls; ++i) {
    client_funcs_[i] = stencil_funcs[i];
  }
  return;
}

template <class GridSpaceType>
void RuntimeMPI<GridSpaceType>::InitRPC(InterProcComm *ipc) {
  if (ipc->GetRank() == Proc::GetRootRank()) {
    LOG_DEBUG() << "I'm the master.\n";
    proc_ = new Master<GridSpaceType>(ipc, client_funcs_, this->gs_);
    LOG_INFO() << *proc_ << "\n";
  } else {    
    LOG_DEBUG() << "I'm a client.\n";
    proc_ = new Client<GridSpaceType>(ipc, client_funcs_, this->gs_);
    LOG_INFO() << *proc_ << "\n";
  }
}

template <class GridSpaceType>
void RuntimeMPI<GridSpaceType>::Listen() {
  assert(!IsMaster());
  static_cast<Client<GridSpaceType>*>(proc_)->Listen();
}

} // namespace runtime
} // namespace physis



#endif /* PHYSIS_RUNTIME_RUNTIME_MPI_H_ */
