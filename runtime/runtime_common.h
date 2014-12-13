// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_COMMON_H_
#define PHYSIS_RUNTIME_RUNTIME_COMMON_H_

#define PHYSIS_RUNTIME

#include "physis/physis_util.h"
#include "physis/physis_common.h"
#include "physis/internal_common.h"
#include "common/config.h"

namespace physis {
namespace runtime {

struct Width2 {
  UnsignedArray bw;
  UnsignedArray fw;
  const UnsignedArray &operator()(bool is_fw) const {
    return is_fw ? fw : bw;
  }
  unsigned operator()(int dim, bool is_fw) const {
    return operator()(is_fw)[dim];
  }
};

typedef void (*__PSStencilRunClientFunction)(int, void **);

// Returns the number of process grid dimensions. Returns negative
// value on failure.
int GetProcessDim(int *argc, char ***argv, IntArray &proc_size);

bool ParseOption(int *argc, char ***argv, const string &opt_name,
                 int num_additional_args, vector<string> &opts);


} // namespace runtime
} // namespace physis

inline
std::ostream &operator<<(std::ostream &os, const physis::runtime::Width2 &w) {
  return os << "{bw: " << w.bw << ", fw: " << w.fw << "}";
}

#endif /* PHYSIS_RUNTIME_RUNTIME_COMMON_H_ */
