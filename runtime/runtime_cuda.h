// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_CUDA_H_
#define PHYSIS_RUNTIME_RUNTIME_CUDA_H_

#include "runtime/runtime_common.h"
#include "runtime/runtime.h"

namespace physis {
namespace runtime {

template <class GridSpaceType>
class RuntimeCUDA: public Runtime<GridSpaceType> {
 public:
  RuntimeCUDA() {}
  virtual ~RuntimeCUDA() {}
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RUNTIME_CUDA_H_ */

