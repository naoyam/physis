// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_RPC_MPI_H_
#define PHYSIS_RUNTIME_RPC_MPI_H_

#include "runtime/runtime_common.h"
#include "runtime/grid_mpi.h"

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

class ProcInfo {
 public:
  int rank_;
  int num_procs_;
  ProcInfo(int rank, int num_procs):
      rank_(rank), num_procs_(num_procs) {}
  std::ostream &print(std::ostream &os) const;
  int rank() const { return rank_; }
  bool IsRoot() const { return rank_ == 0; }
};

class Client {
 protected:
  const ProcInfo &pinfo_;
  GridSpaceMPI *gs_;
  MPI_Comm comm_;        
 public:
  Client(const ProcInfo &pinfo, GridSpaceMPI *gs, MPI_Comm comm);
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
};

class Master {
 protected:
  Counter gridCounter;
  const ProcInfo &pinfo_;
  GridSpaceMPI *gs_;  
  MPI_Comm comm_;
  void NotifyCall(enum RT_FUNC_KIND fkind, int opt=0);
 public:
  Master(const ProcInfo &pinfo, GridSpaceMPI *gs, MPI_Comm comm);
  virtual ~Master() {}
  virtual void Finalize();
  virtual void Barrier();
  virtual GridMPI *GridNew(PSType type, int elm_size,
                           int num_dims,
                           const IndexArray &size,
                           bool double_buffering,
                           const IndexArray &global_offset,
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
};

} // namespace runtime
} // namespace physis

inline
std::ostream &operator<<(std::ostream &os, const physis::runtime::ProcInfo &pinfo) {
  return pinfo.print(os);
}


#endif /* PHYSIS_RUNTIME_RPC_MPI_H_ */
