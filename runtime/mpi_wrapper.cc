// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/mpi_wrapper.h"
#include "physis/physis_util.h"
#include "runtime/mpi_util.h"

namespace physis {
namespace runtime {

int PS_MPI_Init(int *argc, char ***argv) {
  CHECK_MPI(MPI_Init(argc, argv));
  return MPI_SUCCESS;
}

int PS_MPI_Finalize() {
  CHECK_MPI(MPI_Finalize());
  return MPI_SUCCESS;
}

int PS_MPI_Comm_rank(MPI_Comm comm, int *rank) {
  CHECK_MPI(MPI_Comm_rank(comm, rank));
  return MPI_SUCCESS;
}

int PS_MPI_Comm_size(MPI_Comm comm, int *size) {
  CHECK_MPI(MPI_Comm_size(comm, size));
  return MPI_SUCCESS;
}

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

int PS_MPI_Barrier(MPI_Comm comm) {
  CHECK_MPI(MPI_Barrier(comm));
  return MPI_SUCCESS;
}

int PS_MPI_Test(MPI_Request *req, int *flag, MPI_Status *status) {
  CHECK_MPI(MPI_Test(req, flag, status));
  return MPI_SUCCESS;
}

int PS_MPI_Test(MPI_Request *req, bool *flag, MPI_Status *status) {
  int i_flag;
  int r = PS_MPI_Test(req, &i_flag, status);
  *flag = i_flag ? true : false;
  return r;
}

int PS_MPI_Wait(MPI_Request *req, MPI_Status *status) {
  CHECK_MPI(MPI_Wait(req, status));
  return MPI_SUCCESS;
}

} // namespace runtime
} // namespace physis

