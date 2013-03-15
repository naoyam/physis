// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_RUNTIME_RUNTIME_H_
#define PHYSIS_RUNTIME_RUNTIME_H_

#include <stdarg.h>

#include "runtime/runtime_common.h"
#include "runtime/grid.h"

namespace physis {
namespace runtime {

// Returns the number of process grid dimensions. Returns negative
// value on failure.
int GetProcessDim(int *argc, char ***argv, IntArray &proc_size);

bool ParseOption(int *argc, char ***argv, const string &opt_name,
                 int num_additional_args, vector<string> &opts);

class Runtime {
 public:
  Runtime();
  virtual ~Runtime();
  virtual void Init(int *argc, char ***argv, int grid_num_dims,
                    va_list vl);
  virtual GridSpace *gs() {
    return gs_;
  }
 protected:
  GridSpace *gs_;
  
  
};

} // namespace runtime
} // namespace physis



#endif /* PHYSIS_RUNTIME_RUNTIME_H_ */
