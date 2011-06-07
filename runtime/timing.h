// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_TIMING_H_
#define PHYSIS_RUNTIME_TIMING_H_

#include "runtime/runtime_common.h"

namespace physis {
namespace runtime {
namespace performance {

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

} // namespace performance
} // namespace runtime
} // namespace physis

inline std::ostream& operator<<(
    std::ostream &os,
    const physis::runtime::performance::DataCopyProfile &prof) {
  return prof.print(os);
}

#endif /* PHYSIS_RUNTIME_TIMING_H_ */
