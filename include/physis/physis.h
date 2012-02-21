// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_PHYSIS_H_
#define PHYSIS_PHYSIS_H_

#if defined(PHYSIS_REF)
#include "physis/physis_ref.h"
#elif defined(PHYSIS_CUDA)
#include "physis/physis_cuda.h"
#elif defined(PHYSIS_MPI)
#include "physis/physis_mpi.h"
#elif defined(PHYSIS_MPI_CUDA)
#include "physis/physis_mpi_cuda.h"
#elif defined(PHYSIS_OPENCL)
#include "physis/physis_opencl.h"
#elif defined(PHYSIS_MPI_OPENCL)
#include "physis/physis_mpi_opencl.h"
#elif defined(PHYSIS_MPI_OPENMP)
#include "physis_mpi_openmp.h"
#endif

#if defined(PHYSIS_USER)
#include "physis/physis_user.h"
#endif

#include "physis/math.h"

#endif /* PHYSIS_PHYSIS_H_ */
