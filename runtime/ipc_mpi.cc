// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/ipc_mpi.h"

#include <limits.h>

#include "runtime/mpi_wrapper.h"
#include "runtime/mpi_util.h"

namespace physis {
namespace runtime {

InterProcCommMPI *InterProcCommMPI::singleton_ = NULL;

InterProcCommMPI *InterProcCommMPI::GetInstance() {
  if (singleton_ == NULL) {
    singleton_ = new InterProcCommMPI();
  }
  return singleton_;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Init(int *argc,
                                                  char ***argv) {
  PS_MPI_Init(argc, argv);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Finalize() {
  PS_MPI_Finalize();
  return IPC_SUCCESS;
}

int InterProcCommMPI::GetRank() const {
  int rank;
  assert(PS_MPI_Comm_rank(comm_, &rank) == MPI_SUCCESS);
  return rank;
}

int InterProcCommMPI::GetNumProcs() const {
  int np;
  assert(PS_MPI_Comm_size(comm_, &np) == MPI_SUCCESS);
  return np;
}


void *InterProcCommMPI::CreateRequest() const {
  MPI_Request *r = new MPI_Request;
  return (void*)r;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Send(
    void *buf, size_t len, int dest) {
  int tag = 0;
  assert(PS_MPI_Send(buf, len, MPI_BYTE,
                     dest, tag, comm_) == MPI_SUCCESS);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Isend(
    void *buf, size_t len, int dest, void *req) {
  MPI_Request *mpi_req = static_cast<MPI_Request*>(req);
  int tag = 0;
  assert(PS_MPI_Isend(buf, len, MPI_BYTE,
                      dest, tag, comm_, mpi_req)
         == MPI_SUCCESS);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Recv(
    void *buf, size_t len, int src) {
  int tag = 0;
  assert(PS_MPI_Recv(buf, len, MPI_BYTE, src, tag, comm_,
                     MPI_STATUS_IGNORE) == MPI_SUCCESS);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Irecv(
    void *buf, size_t len, int src, void *req) {
  int tag = 0;
  MPI_Request *mpi_req = static_cast<MPI_Request*>(req);  
  assert(PS_MPI_Irecv(buf, len, MPI_BYTE, src, tag, comm_,
                      mpi_req)
         == MPI_SUCCESS);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Wait(void *req) {
  assert(PS_MPI_Wait() == MPI_SUCCESS);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::WaitAll() {
  assert(PS_MPI_Wait() == MPI_SUCCESS);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Test(void *req) {
  assert(PS_MPI_Test(static_cast<MPI_Request*>(req))
         == MPI_SUCCESS);
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Bcast(void *buf, size_t len, int root) {
  // MPI Bcast uses int type for length parameter
  while (len > 0 ) {
    int bcast_len = (int)std::min((size_t)INT_MAX, len);
    assert(PS_MPI_Bcast(buf, bcast_len, MPI_BYTE, root, comm_)
           == MPI_SUCCESS);
    len -= bcast_len;    
  }
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Reduce(void *src, void *dst,
                             int count, PSType type,
                             PSReduceOp op, int root) {
  assert(MPI_SUCCESS ==
         PS_MPI_Reduce(src, dst, count, GetMPIDataType(type),
                       GetMPIOp(op), root, comm_));
  return IPC_SUCCESS;
}

InterProcComm::IPC_ERROR_T InterProcCommMPI::Barrier() {
  assert(MPI_SUCCESS == PS_MPI_Barrier(comm_));
  return IPC_SUCCESS;
}
} // namespace runtime
} // namespace physis

