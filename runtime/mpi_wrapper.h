// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_MPI_WRAPPER_H_
#define PHYSIS_RUNTIME_MPI_WRAPPER_H_

#include "mpi.h"

#include "runtime/mpi_util.h"
#include "runtime/runtime_common.h"

namespace physis {
namespace runtime {

extern int PS_MPI_Init(int *argc, char ***argv);

extern int PS_MPI_Finalize();

extern int PS_MPI_Comm_rank(MPI_Comm comm, int *rank);

extern int PS_MPI_Comm_size(MPI_Comm comm, int *size);

extern int PS_MPI_Send( void *buf, int count, MPI_Datatype datatype, int dest, 
                        int tag, MPI_Comm comm );
extern int PS_MPI_Isend(void *buf, int count, MPI_Datatype datatype,
                        int dest, int tag, MPI_Comm comm, MPI_Request *request );


extern int PS_MPI_Recv( void *buf, int count, MPI_Datatype datatype,
                        int source, 
                        int tag, MPI_Comm comm, MPI_Status *status );

extern int PS_MPI_Irecv( void *buf, int count, MPI_Datatype datatype,
                         int source, 
                         int tag, MPI_Comm comm, MPI_Request *request );

extern int PS_MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                        int root, MPI_Comm comm);

extern int PS_MPI_Reduce(void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op,
                         int root, MPI_Comm comm);

extern int PS_MPI_Barrier(MPI_Comm comm);

extern int PS_MPI_Test(MPI_Request *req, int *flag, MPI_Status *status);
extern int PS_MPI_Test(MPI_Request *req, bool *flag, MPI_Status *status);

extern int PS_MPI_Wait(MPI_Request *req, MPI_Status *status);
                         

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_MPI_WRAPPER_H_ */
