#ifndef PHYSIS_RUNTIME_RPC_MPI_OPENMP_H_
#define PHYSIS_RUNTIME_RPC_MPI_OPENMP_H_

#include "runtime/grid_mpi_openmp.h"
#include "runtime/rpc_mpi.h"

namespace physis {
namespace runtime {

enum RT_FUNC_MP_KIND {
  FUNC_MP_INVALID, FUNC_MP_NEW, FUNC_MP_DELETE,
  FUNC_MP_COPYIN, FUNC_MP_COPYOUT,
  FUNC_MP_GET, FUNC_MP_SET,
  FUNC_MP_RUN, FUNC_MP_FINALIZE, FUNC_MP_BARRIER,
  FUNC_MP_GRID_REDUCE,
  FUNC_MP_INIT_NUMA
};

class ProcInfoOpenMP: public ProcInfo {
 public:
  ProcInfoOpenMP(int rank, int num_procs, IntArray &division_size):
      ProcInfo(rank, num_procs), division_size_(division_size) {}

 protected:
  IntArray division_size_;

 public:
  IntArray division_size() const { return division_size_; }
};

class ClientOpenMP: public Client {
 protected:
  const ProcInfoOpenMP &pinfo_mp_;   
  GridSpaceMPIOpenMP *gs_mp_;  
 public:
  ClientOpenMP(
      const ProcInfoOpenMP &pinfo_mp, GridSpaceMPIOpenMP *gs_mp, MPI_Comm comm);
  virtual void Listen(); 
  virtual void GridNew();
  virtual void GridCopyin(int id);
  virtual void GridCopyout(int id);

  virtual void GridInitNUMA(int gridid);
};

class MasterOpenMP: public Master {
 protected:
  void NotifyCall(enum RT_FUNC_MP_KIND fkind, int opt=0);
 protected:
  const ProcInfoOpenMP &pinfo_mp_;
  GridSpaceMPIOpenMP *gs_mp_;
 public:
  MasterOpenMP(
      const ProcInfoOpenMP &pinfo_mp, GridSpaceMPIOpenMP *gs_mp, MPI_Comm comm);
  virtual void Finalize();
  virtual void Barrier();
  virtual GridMPIOpenMP *GridNew(PSType type, int elm_size,
                                 int num_dims,
                                 const IntArray &size,
                                 bool double_buffering,
                                 const IntArray &global_offset,
                                 int attr);
  virtual void GridDelete(GridMPIOpenMP *g);
  virtual void GridCopyin(GridMPIOpenMP *g, void *buf);
  virtual void GridCopyinLocal(GridMPIOpenMP *g, void *buf);  
  virtual void GridCopyout(GridMPIOpenMP *g, void *buf);
  virtual void GridCopyoutLocal(GridMPIOpenMP *g, void *buf); 
  virtual void GridSet(GridMPIOpenMP *g, const void *buf, const IntArray &index);
  virtual void GridGet(GridMPIOpenMP *g, void *buf, const IntArray &index);  
  virtual void StencilRun(int id, int iter, int num_stencils,
                          void **stencils, unsigned *stencil_sizes);
  virtual void GridReduce(void *buf, PSReduceOp op, GridMPIOpenMP *g);

  virtual void GridInitNUMA(GridMPIOpenMP *g, unsigned int maxMPthread);
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RPC_MPI_OPENMP_H_ */
