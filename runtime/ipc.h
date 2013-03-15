// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_IPC_H_
#define PHYSIS_RUNTIME_IPC_H_

#include "runtime/runtime_common.h"

namespace physis {
namespace runtime {

// Singleton class
class InterProcComm {
 protected:
  InterProcComm() {}
  virtual ~InterProcComm() {}
 public:
  typedef enum {IPC_SUCCESS = 0, IPC_FAILURE = 1} IPC_ERROR_T;
  virtual void *CreateRequest() const = 0;
  virtual IPC_ERROR_T Init(int *argc, char ***argv) = 0;
  virtual IPC_ERROR_T Finalize() = 0;
  virtual int GetRank() const = 0;
  virtual int GetNumProcs() const = 0;
  virtual IPC_ERROR_T Send(void *buf, size_t len,
                           int dest) = 0;
  virtual IPC_ERROR_T Isend(void *buf, size_t len,
                            int dest, void *req) = 0;
  virtual IPC_ERROR_T Recv(void *buf, size_t len, int src) = 0;
  virtual IPC_ERROR_T Irecv(void *buf, size_t len,
                    int src, void *req) = 0;
  virtual IPC_ERROR_T Wait(void *req) = 0;
  virtual IPC_ERROR_T WaitAll() = 0;
  virtual IPC_ERROR_T Test(void *req) = 0;
  virtual IPC_ERROR_T Bcast(void *buf, size_t len, int root) = 0;
  virtual IPC_ERROR_T Reduce(void *src, void *dst,
                             int count,
                             PSType type,
                             PSReduceOp op, int root) = 0;
  virtual IPC_ERROR_T Barrier() = 0;

};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_IPC_H_ */
