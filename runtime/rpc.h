// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_RPC_H_
#define PHYSIS_RUNTIME_RPC_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_mpi.h"
#include "runtime/proc.h"

namespace physis {
namespace runtime {

typedef std::vector<std::pair<int, void*> > StencilArgVector;

enum RT_FUNC_KIND {
  FUNC_INVALID, FUNC_NEW, FUNC_DELETE,
  FUNC_COPYIN, FUNC_COPYOUT,
  FUNC_GET, FUNC_SET,
  FUNC_RUN, FUNC_FINALIZE, FUNC_BARRIER,
  FUNC_GRID_REDUCE
};

struct Request {
  RT_FUNC_KIND kind;
  int opt;
  Request(RT_FUNC_KIND k=FUNC_INVALID, int opt=0)
      : kind(k), opt(opt) {}
};

struct RequestNEW {
  PSType type;
  int elm_size;
  int num_dims;
  IndexArray size;
  bool double_buffering;
  IndexArray global_offset;
  IndexArray stencil_offset_min;
  IndexArray stencil_offset_max;
  int attr;
};

class Client: public Proc {
 protected:
  GridSpaceMPI *gs_;
  bool done_;
 public:
  Client(int rank, int num_procs, InterProcComm *ipc,
         __PSStencilRunClientFunction *stencil_runs,
         GridSpaceMPI *gs);
  virtual ~Client() {}
  virtual void Listen();  
  virtual void Finalize();
  virtual void Barrier();
  virtual void GridNew();
  virtual void GridDelete(int id);
  virtual void GridCopyin(int id);
  virtual void GridCopyout(int id);
  virtual void GridSet(int id);
  virtual void GridGet(int id);  
  virtual void StencilRun(int id);
  virtual void GridReduce(int id);
  static int GetMasterRank() {
    return Proc::GetRootRank();
  }
};

class Master: public Proc {
 protected:
  Counter gridCounter;
  GridSpaceMPI *gs_;  
  void NotifyCall(enum RT_FUNC_KIND fkind, int opt=0);
 public:
  Master(int rank, int num_procs, InterProcComm *ipc,
         __PSStencilRunClientFunction *stencil_runs,
         GridSpaceMPI *gs);
  virtual ~Master() {}
  virtual void Finalize();
  virtual void Barrier();
  virtual GridMPI *GridNew(PSType type, int elm_size,
                           int num_dims,
                           const IndexArray &size,
                           const IndexArray &global_offset,
                           const IndexArray &stencil_offset_min,
                           const IndexArray &stencil_offset_max,
                           int attr);
  virtual void GridDelete(GridMPI *g);
  virtual void GridCopyin(GridMPI *g, const void *buf);
  virtual void GridCopyinLocal(GridMPI *g, const void *buf);  
  virtual void GridCopyout(GridMPI *g, void *buf);
  virtual void GridCopyoutLocal(GridMPI *g, void *buf);  
  virtual void GridSet(GridMPI *g, const void *buf, const IndexArray &index);
  virtual void GridGet(GridMPI *g, void *buf, const IndexArray &index);  
  virtual void StencilRun(int id, int iter, int num_stencils,
                          void **stencils, unsigned *stencil_sizes);
  virtual void GridReduce(void *buf, PSReduceOp op, GridMPI *g);
  static int GetMasterRank() {
    return Proc::GetRootRank();
  }
  
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RPC_H_ */