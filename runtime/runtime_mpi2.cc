// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/runtime_mpi2.h"
#include "runtime/ipc_mpi.h"

namespace physis {
namespace runtime {

void RuntimeMPI2::Init(int *argc, char ***argv,
                       int grid_num_dims,
                       va_list vl) {
  Runtime::Init(argc, argv, grid_num_dims, vl);
  IndexArray grid_size;
  for (int i = 0; i < grid_num_dims; ++i) {
    grid_size[i] = va_arg(vl, PSIndex);
  }
  int num_stencil_run_calls = va_arg(vl, int);
  __PSStencilRunClientFunction *stencil_funcs
      = va_arg(vl, __PSStencilRunClientFunction*);
  va_end(vl);

  InterProcComm *ipc = InterProcCommMPI::GetInstance();
  ipc->Init(argc, argv);
    
  int num_procs = ipc->GetNumProcs();
  int rank = ipc->GetRank();

  IntArray proc_size;
  proc_size.Set(1);
  int proc_num_dims = GetProcessDim(argc, argv, proc_size);
  if (proc_num_dims < 0) {
    proc_num_dims = grid_num_dims;
    // 1-D process decomposition by default
    LOG_INFO() <<
        "No process dimension specified; defaulting to 1D decomposition\n";
    proc_size[proc_num_dims-1] = num_procs;
  }

  LOG_INFO() << "Process size: " << proc_size << "\n";

  gs_ = new GridSpaceMPI2(grid_num_dims, grid_size,
                          proc_num_dims, proc_size, rank);

  LOG_INFO() << "Grid space: " << *gs_ << "\n";

  // Set the stencil client functions
  client_funcs_ =
      (__PSStencilRunClientFunction*)malloc(
          sizeof(__PSStencilRunClientFunction)
          * num_stencil_run_calls);
  memcpy(client_funcs_, stencil_funcs,
         sizeof(__PSStencilRunClientFunction) *
         num_stencil_run_calls);

  if (rank != 0) {
    LOG_DEBUG() << "I'm a client.\n";
    Client2 *client = new Client2(
        rank, num_procs, ipc, client_funcs_,
        static_cast<GridSpaceMPI*>(gs_));
    proc_ = client;
    LOG_INFO() << *client << "\n";
  } else {
    LOG_DEBUG() << "I'm the master.\n";
    proc_ = new Master2(
        rank, num_procs, ipc, client_funcs_,
        static_cast<GridSpaceMPI*>(gs_));
    LOG_INFO() << *proc_ << "\n";      
  }
  
  return;
}


} // namespace runtime
} // namespace physis
