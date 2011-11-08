// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_MPI_WRAPPER_H_
#define PHYSIS_RUNTIME_MPI_WRAPPER_H_

#include "mpi.h"

#include "runtime/mpi_util.h"
#include "runtime/runtime_common.h"

namespace physis {
namespace runtime {

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
                         

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_MPI_WRAPPER_H_ */
