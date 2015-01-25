// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_MPI_UTIL_H_
#define PHYSIS_RUNTIME_MPI_UTIL_H_

#include <mpi.h>

#include "physis/physis_common.h"
#include "physis/physis_util.h"

#define CHECK_MPI(c) do {                       \
  int ret = c;                                  \
  if (ret != MPI_SUCCESS) {                     \
    LOG_ERROR() << "MPI error\n";               \
    PSAbort(1);                                 \
  }                                             \
  } while (0)

static inline int MPI_GET_RANK(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

// displaying mpi rank can be achieved mpirun command-line option in OpenMPI
#if 0
#define LOG_VERBOSE_MPI()                                       \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"  \
   <<"[VERBOSE:" << __FUNC_ID__                                 \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                    \
   << "#"  << __LINE__ << "] ")
#define LOG_DEBUG_MPI()                                         \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"  \
   <<"[DEBUG:" << __FUNC_ID__                                   \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                    \
   << "#"  << __LINE__ << "] ")
#define LOG_WARNING_MPI()                                       \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"  \
   <<"[WARNING:" << __FUNC_ID__                                 \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                    \
   << "#"  << __LINE__ << "] ")

#define LOG_ERROR_MPI()                                         \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"  \
   <<"[ERROR:" << __FUNC_ID__                                   \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                    \
   << "#"  << __LINE__ << "] ")

#define LOG_INFO_MPI()                                          \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"  \
   <<"[INFO:" << __FUNC_ID__                                    \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                    \
   << "#"  << __LINE__ << "] ")
#else
#define LOG_VERBOSE_MPI LOG_VERBOSE
#define LOG_DEBUG_MPI LOG_DEBUG
#define LOG_WARNING_MPI LOG_WARNING
#define LOG_ERROR_MPI LOG_ERROR
#define LOG_INFO_MPI LOG_INFO
#endif

namespace physis {
namespace runtime {
template <class T>
MPI_Datatype GetMPIDataType() {
  PSAssert(0);
  return MPI_INT; // dummy; never reach here.
}

template <> inline
MPI_Datatype GetMPIDataType<float>() {
  return MPI_FLOAT;
}

template <> inline
MPI_Datatype GetMPIDataType<double>() {
  return MPI_DOUBLE;
}

inline
MPI_Datatype GetMPIDataType(PSType type) {
  MPI_Datatype mpi_type;
  switch (type) {
    case PS_INT:
      mpi_type = MPI_INT;
      break;
    case PS_LONG:
      mpi_type = MPI_LONG;
      break;
    case PS_FLOAT:
      mpi_type = MPI_FLOAT;
      break;
    case PS_DOUBLE:
      mpi_type = MPI_DOUBLE;
      break;
    default:
      PSAbort(1);
  }
  return mpi_type;
}

inline
MPI_Op GetMPIOp(PSReduceOp op) {
  MPI_Op mpi_op;
  switch (op) {
    case PS_MAX:
      mpi_op = MPI_MAX;
      break;
    case PS_MIN:
      mpi_op = MPI_MIN;
      break;
    case PS_SUM:
      mpi_op = MPI_SUM;
      break;
    case PS_PROD:
      mpi_op = MPI_PROD;
      break;
    default:
      PSAbort(1);
  }
  return mpi_op;
}

} // namespace runtime
} // namespace physis


#endif /* PHYSIS_RUNTIME_MPI_UTIL_H_ */

