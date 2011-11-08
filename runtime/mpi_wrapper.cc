// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/mpi_wrapper.h"
#include "physis/physis_util.h"
#include "runtime/mpi_util.h"

namespace physis {
namespace runtime {

int PS_MPI_Send( void *buf, int count, MPI_Datatype datatype, int dest, 
                 int tag, MPI_Comm comm ) {
  LOG_VERBOSE() << "MPI_Send " << count << " entries to " << dest << "\n";
  CHECK_MPI(MPI_Send(buf, count, datatype, dest, tag, comm));
  return MPI_SUCCESS;
}

int PS_MPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                 MPI_Comm comm, MPI_Request *request) {
  LOG_VERBOSE() << "MPI_Isend " << count << " entries to " << dest << "\n";
  CHECK_MPI(MPI_Isend(buf, count, datatype, dest, tag, comm, request));
  return MPI_SUCCESS;
}

int PS_MPI_Recv( void *buf, int count, MPI_Datatype datatype, int source, 
                 int tag, MPI_Comm comm, MPI_Status *status ) {
  LOG_VERBOSE() << "MPI_Recv " << count << " entries from " << source << "\n";  
  CHECK_MPI(MPI_Recv(buf, count, datatype, source, tag, comm, status));
  return MPI_SUCCESS;
}

int PS_MPI_Irecv( void *buf, int count, MPI_Datatype datatype, int source, 
                  int tag, MPI_Comm comm, MPI_Request *request) {
  LOG_VERBOSE() << "MPI_IRecv " << count << " entries from " << source << "\n";    
  CHECK_MPI(MPI_Irecv(buf, count, datatype, source, tag, comm, request));
  return MPI_SUCCESS;
}


int PS_MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, 
                  MPI_Comm comm ) {
  CHECK_MPI(MPI_Bcast(buffer, count, datatype, root, comm));
  return MPI_SUCCESS;
}


int PS_MPI_Reduce(void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op,
                  int root, MPI_Comm comm) {
  CHECK_MPI(MPI_Reduce(sendbuf, recvbuf, count, datatype,
                       op, root, comm));
  return MPI_SUCCESS;
}

} // namespace runtime
} // namespace physis

