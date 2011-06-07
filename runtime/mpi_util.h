// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

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
#define LOG_VERBOSE_MPI()                                         \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"    \
   <<"[VERBOSE:" << __FUNC_ID__                                   \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                      \
   << "#"  << __LINE__ << "] ")
#define LOG_DEBUG_MPI()                                        \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]" \
   <<"[DEBUG:" << __FUNC_ID__                                   \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                    \
   << "#"  << __LINE__ << "] ")
#define LOG_WARNING_MPI()                                         \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"   \
   <<"[WARNING:" << __FUNC_ID__                                   \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                      \
   << "#"  << __LINE__ << "] ")

#define LOG_ERROR_MPI()                                         \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"   \
   <<"[ERROR:" << __FUNC_ID__                                   \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                      \
   << "#"  << __LINE__ << "] ")

#define LOG_INFO_MPI()                                         \
  (std::cerr << "[MPI:" << MPI_GET_RANK(MPI_COMM_WORLD) << "]"   \
   <<"[INFO:" << __FUNC_ID__                                   \
   << "@" << LOGGING_FILE_BASENAME(__FILE__)                      \
   << "#"  << __LINE__ << "] ")
#else
#define LOG_VERBOSE_MPI LOG_VERBOSE
#define LOG_DEBUG_MPI LOG_DEBUG
#define LOG_WARNING_MPI LOG_WARNING
#define LOG_ERROR_MPI LOG_ERROR
#define LOG_INFO_MPI LOG_INFO
#endif


#endif /* PHYSIS_RUNTIME_MPI_UTIL_H_ */

