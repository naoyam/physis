// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_TIMING_H_
#define PHYSIS_RUNTIME_TIMING_H_

#include "runtime/runtime_common.h"

namespace physis {
namespace runtime {

struct DataCopyProfile {
  double gpu_to_cpu;
  double cpu_in;    
  double cpu_out;
  double cpu_to_gpu;
  DataCopyProfile();
  std::ostream &print(std::ostream &os) const;
};

struct Stopwatch {
  __PSStopwatch st;
  void Start() {
    __PSStopwatchStart(&st);
  }
  float Stop() {
    return __PSStopwatchStop(&st);
  }
};

} // namespace runtime
} // namespace physis

inline std::ostream& operator<<(
    std::ostream &os,
    const physis::runtime::DataCopyProfile &prof) {
  return prof.print(os);
}

#endif /* PHYSIS_RUNTIME_TIMING_H_ */
