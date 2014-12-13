// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/runtime_mpi.h"

#include "runtime/ipc_mpi.h"
#include "runtime/proc.h"
#include "runtime/rpc.h"

namespace physis {
namespace runtime {

RuntimeMPI::RuntimeMPI(): Runtime() {
}

RuntimeMPI::~RuntimeMPI() {
}

void RuntimeMPI::Init(int *argc, char ***argv, int domain_rank,
                      va_list vl) {
  Runtime::Init(argc, argv, domain_rank, vl);
  
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

void RuntimeMPI::InitDomainSize(int domain_rank, va_list vl,
                                IndexArray &domain_size) {
  for (int i = 0; i < domain_rank; ++i) {
    domain_size[i] = va_arg(vl, PSIndex);
  }
}


InterProcComm *RuntimeMPI::GetIPC(int *argc, char ***argv) {
  InterProcComm *ipc = InterProcCommMPI::GetInstance();
  ipc->Init(argc, argv);
  return ipc;
}

void RuntimeMPI::InitGridSpace(int *argc, char ***argv, int domain_rank,
                               const IndexArray &domain_size,
                               int proc_num_dims, const IndexArray &proc_size,
                               InterProcCommMPI &ipc) {
  gs_ = new GridSpaceMPI<GridMPI>(domain_rank, domain_size,
                                  proc_num_dims, proc_size, ipc);
  //LOG_INFO() << "Grid space: " << *gs_ << "\n";
}

void RuntimeMPI::InitProcSize(int *argc, char ***argv,
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
void RuntimeMPI::InitStencilFuncs(va_list vl) {
  int num_stencil_run_calls = va_arg(vl, int);
  __PSStencilRunClientFunction *stencil_funcs
      = va_arg(vl, __PSStencilRunClientFunction*);
  
  client_funcs_ = new __PSStencilRunClientFunction[num_stencil_run_calls];
  for (int i = 0; i < num_stencil_run_calls; ++i) {
    client_funcs_[i] = stencil_funcs[i];
  }
  return;
}

void RuntimeMPI::InitRPC(InterProcComm *ipc) {
  if (ipc->GetRank() == Proc::GetRootRank()) {
    LOG_DEBUG() << "I'm the master.\n";
    proc_ = new Master<GridMPI>(ipc, client_funcs_,
                                gs());
    LOG_INFO() << *proc_ << "\n";
  } else {    
    LOG_DEBUG() << "I'm a client.\n";
    proc_ = new Client<GridMPI>(ipc, client_funcs_, gs());
    LOG_INFO() << *proc_ << "\n";
  }
}

void RuntimeMPI::Listen() {
  assert(!IsMaster());
  static_cast<Client<GridMPI>*>(proc_)->Listen();
}

} // namespace runtime
} // namespace physis
