// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/timing.h"

namespace physis {
namespace runtime {
namespace performance {

DataCopyProfile::DataCopyProfile():
    gpu_to_cpu(0.0), cpu_in(0.0), cpu_out(0.0), cpu_to_gpu(0.0) {}


std::ostream &DataCopyProfile::print(std::ostream &os) const {
  StringJoin sj;
  sj << "GPU->CPU: " << gpu_to_cpu;
  sj << "CPU->MPI: " << cpu_out;
  sj << "MPI->CPU: " << cpu_in;
  sj << "CPU->GPU: " << cpu_to_gpu;
  os << "(" << sj.str() << ")";
  return os;
}


} // namespace performance
} // namespace runtime
} // namespace physis

