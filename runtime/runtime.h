// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_H_
#define PHYSIS_RUNTIME_RUNTIME_H_

#include <stdarg.h>

#include "runtime/runtime_common.h"
#include "runtime/grid.h"

namespace physis {
namespace runtime {


template <class GridSpaceType>
class Runtime {
 public:
  Runtime(): gs_(NULL) {}
  virtual ~Runtime() {}
  virtual void Init(int *argc, char ***argv, int grid_num_dims,
                    va_list vl) {
    // Set __ps_trace if physis-trace option is given
    __ps_trace = NULL;
    string opt_name = "physis-trace";
    vector<string> opts;
    if (ParseOption(argc, argv, opt_name, 0, opts)) {
      __ps_trace = stderr;
      LOG_INFO() << "Tracing enabled\n";
    }
  }
    
  virtual GridSpaceType *gs() {
    return gs_;
  }
 protected:
  GridSpaceType *gs_;
  
};

} // namespace runtime
} // namespace physis



#endif /* PHYSIS_RUNTIME_RUNTIME_H_ */
