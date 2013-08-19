// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "runtime/proc.h"

namespace physis {
namespace runtime {

std::ostream &Proc::print(std::ostream &os) const {
  os << "Proc {"
     << "rank: " << rank_
     << ", #procs: " << num_procs_
     << "}";
  return os;
}

} // namespace runtime
} // namespace physis
