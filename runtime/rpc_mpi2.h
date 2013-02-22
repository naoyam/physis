// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.
#ifndef PHYSIS_RUNTIME_RPC_MPI2_H_
#define PHYSIS_RUNTIME_RPC_MPI2_H_

#include "runtime/runtime_common.h"
#include "runtime/rpc_mpi.h"

namespace physis {
namespace runtime {

class Master2: public Master {
 public:
  Master2(const ProcInfo &pinfo, GridSpaceMPI *gs, MPI_Comm comm):
      Master(pinfo, gs, comm) {}
  virtual ~Master2() {}
  virtual GridMPI *GridNew(PSType type, int elm_size,
                           int num_dims,
                           const IndexArray &size,
                           const IndexArray &global_offset,
                           const IndexArray &stencil_offset_min,
                           const IndexArray &stencil_offset_max,                           
                           int attr);
  virtual void GridCopyoutLocal(GridMPI *g, void *buf);
  virtual void GridCopyinLocal(GridMPI *g, const void *buf);
};

class Client2: public Client {
 public:
  Client2(const ProcInfo &pinfo, GridSpaceMPI *gs, MPI_Comm comm):
      Client(pinfo, gs, comm) {}
  virtual ~Client2() {}
  virtual void GridNew();
  virtual void GridCopyout(int id);
  virtual void GridCopyin(int id);  
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RPC_MPI2_H_ */
