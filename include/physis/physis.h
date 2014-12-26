// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_PHYSIS_H_
#define PHYSIS_PHYSIS_H_

#if defined(PHYSIS_USER)
#include "physis/physis_user.h"
#endif

#if defined(PHYSIS_REF)
#include "physis/physis_ref.h"
#elif defined(PHYSIS_CUDA)
#include "physis/physis_cuda.h"
#elif defined(PHYSIS_CUDA_HM)
#include "physis/physis_cuda_hm.h"
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


#include "physis/math.h"

#endif /* PHYSIS_PHYSIS_H_ */
