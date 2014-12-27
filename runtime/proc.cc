// Licensed under the BSD license. See LICENSE.txt for more details.

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
