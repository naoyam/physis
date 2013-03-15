// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_RUNTIME_MPI_H_
#define PHYSIS_RUNTIME_RUNTIME_MPI_H_

#include <stdarg.h>

#include "runtime/runtime.h"
#include "runtime/grid_mpi.h"
#include "runtime/proc.h"
#include "runtime/rpc.h"

namespace physis {
namespace runtime {

class RuntimeMPI: public Runtime {
 public:
  RuntimeMPI();
  virtual ~RuntimeMPI();
  virtual void Init(int *argc, char ***argv, int grid_num_dims,
                    va_list vl);
  virtual GridSpaceMPI *gs() {
    return static_cast<GridSpaceMPI*>(gs_);
  }
  virtual __PSStencilRunClientFunction *client_funcs() {
    return client_funcs_;
  }
  virtual Proc *proc() {
    return proc_;
  }
  virtual int IsMaster() {
    return proc_->rank() == Master::GetMasterRank();
  }
  void Listen();
  
 protected:
  __PSStencilRunClientFunction *client_funcs_;
  Proc *proc_;
  
};

} // namespace runtime
} // namespace physis



#endif /* PHYSIS_RUNTIME_RUNTIME_MPI_H_ */
