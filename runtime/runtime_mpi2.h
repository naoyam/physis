// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_RUNTIME_MPI2_H_
#define PHYSIS_RUNTIME_RUNTIME_MPI2_H_

#include <stdarg.h>

#include "runtime/runtime_mpi.h"
#include "runtime/grid_mpi2.h"
#include "runtime/rpc2.h"

namespace physis {
namespace runtime {

class RuntimeMPI2: public RuntimeMPI {
 public:
  RuntimeMPI2(): RuntimeMPI() {}
  virtual ~RuntimeMPI2() {}
  virtual void Init(int *argc, char ***argv, int grid_num_dims,
                    va_list vl);
  virtual GridSpaceMPI2 *gs() {
    return static_cast<GridSpaceMPI2*>(gs_);
  }
};

} // namespace runtime
} // namespace physis



#endif /* PHYSIS_RUNTIME_RUNTIME_MPI2_H_ */
