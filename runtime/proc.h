// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_PROC_H_
#define PHYSIS_RUNTIME_PROC_H_

#include "runtime/runtime_common.h"
#include "runtime/ipc.h"

namespace physis {
namespace runtime {

class Proc {
 protected:
  int rank_;
  int num_procs_;
  InterProcComm *ipc_;
  __PSStencilRunClientFunction *stencil_runs_;
 public:
  Proc(int rank, int num_procs, InterProcComm *ipc,
       __PSStencilRunClientFunction *stencil_runs):
      rank_(rank), num_procs_(num_procs), ipc_(ipc),
      stencil_runs_(stencil_runs) {}
  virtual ~Proc() {}
  std::ostream &print(std::ostream &os) const;
  int rank() const { return rank_; }
  InterProcComm *ipc() { return ipc_; }  
  static int GetRootRank() { return 0; }
  bool IsRoot() const { return rank_ == GetRootRank(); }
};

} // namespace runtime
} // namespace physis

inline
std::ostream &operator<<(std::ostream &os, const physis::runtime::Proc &proc) {
  return proc.print(os);
}

#endif /* PHYSIS_RUNTIME_PROC_H_ */
