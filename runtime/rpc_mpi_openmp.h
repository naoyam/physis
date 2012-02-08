#ifndef PHYSIS_RUNTIME_RPC_MPI_OPENMP_H_
#define PHYSIS_RUNTIME_RPC_MPI_OPENMP_H_

#include "runtime/grid_mpi_openmp.h"
#include "runtime/rpc_mpi.h"

namespace physis {
namespace runtime {

class ProcInfoOpenMP: public ProcInfo {
 public:
#if 0
  int rank_;
  int num_procs_;
#endif
  ProcInfoOpenMP(int rank, int num_procs, IntArray &division_size):
      ProcInfo(rank, num_procs), division_size_(division_size) {}
#if 0
  std::ostream &print(std::ostream &os) const;
  int rank() const { return rank_; }
  bool IsRoot() const { return rank_ == 0; }
#endif

 public:
  IntArray division_size_;
};

class ClientOpenMP: public Client {
 protected:
#if 0
  const ProcInfo &pinfo_;
  GridSpaceMPI *gs_;
  MPI_Comm comm_;  
#endif 
 protected:
  const ProcInfoOpenMP &pinfo_mp_;   
  GridSpaceMPIOpenMP *gs_mp_;  
 public:
  ClientOpenMP(
    const ProcInfoOpenMP &pinfo_mp, GridSpaceMPIOpenMP *gs_mp, MPI_Comm comm);
#if 0
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
#endif
};

class MasterOpenMP: public Master {
 protected:
#if 0
  Counter gridCounter;
  const ProcInfo &pinfo_;
  GridSpaceMPI *gs_;  
  MPI_Comm comm_;
  void NotifyCall(enum RT_FUNC_KIND fkind, int opt=0);
#endif
 protected:
  const ProcInfoOpenMP &pinfo_mp_;
  GridSpaceMPIOpenMP *gs_mp_;
 public:
  MasterOpenMP(
    const ProcInfoOpenMP &pinfo_mp, GridSpaceMPIOpenMP *gs_mp, MPI_Comm comm);
#if 0
  virtual ~Master() {}
  virtual void Finalize();
  virtual void Barrier();
  virtual GridMPI *GridNew(PSType type, int elm_size,
                           int num_dims,
                           const IntArray &size,
                           bool double_buffering,
                           const IntArray &global_offset,
                           int attr);
  virtual void GridDelete(GridMPI *g);
  virtual void GridCopyin(GridMPI *g, const void *buf);
  virtual void GridCopyinLocal(GridMPI *g, const void *buf);  
  virtual void GridCopyout(GridMPI *g, void *buf);
  virtual void GridCopyoutLocal(GridMPI *g, void *buf);  
  virtual void GridSet(GridMPI *g, const void *buf, const IntArray &index);
  virtual void GridGet(GridMPI *g, void *buf, const IntArray &index);  
  virtual void StencilRun(int id, int iter, int num_stencils,
                  void **stencils, int *stencil_sizes);
  virtual void GridReduce(void *buf, PSReduceOp op, GridMPI *g);
#endif
};

} // namespace runtime
} // namespace physis

#if 0
inline
std::ostream &operator<<(std::ostream &os, const physis::runtime::ProcInfo &pinfo) {
  return pinfo.print(os);
}
#endif


#endif /* PHYSIS_RUNTIME_RPC_MPI_OPENMP_H_ */
