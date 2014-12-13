// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_IPC_MPI_H_
#define PHYSIS_RUNTIME_IPC_MPI_H_

#include "mpi.h"

#include "runtime/ipc.h"


namespace physis {
namespace runtime {

class InterProcCommMPI: public InterProcComm {
 protected:
  InterProcCommMPI(): comm_(MPI_COMM_WORLD), initialized_(false) {}
  virtual ~InterProcCommMPI() {}  
 public:
  static InterProcCommMPI* GetInstance();
  virtual void *CreateRequest() const;
  virtual IPC_ERROR_T Init(int *argc, char ***argv);
  virtual IPC_ERROR_T Finalize();
  virtual int GetRank() const;
  virtual int GetNumProcs() const;
  virtual IPC_ERROR_T Send(void *buf, size_t len, int dest);
  virtual IPC_ERROR_T Isend(void *buf, size_t len,
                            int dest, void *req);
  virtual IPC_ERROR_T Recv(void *buf, size_t len, int src);
  virtual IPC_ERROR_T Irecv(void *buf, size_t len,
                    int src, void *req);
  virtual IPC_ERROR_T Wait(void *req);
  //virtual IPC_ERROR_T WaitAll();
  virtual IPC_ERROR_T Test(void *req, bool *flag);
  virtual IPC_ERROR_T Bcast(void *buf, size_t len, int root);
  virtual IPC_ERROR_T Reduce(void *src, void *dst,
                     int count, PSType type,
                     PSReduceOp op, int root);
  virtual IPC_ERROR_T Barrier();

 protected:
  MPI_Comm comm_;
  static InterProcCommMPI *singleton_;
  bool initialized_;
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_IPC_MPI_H_ */


