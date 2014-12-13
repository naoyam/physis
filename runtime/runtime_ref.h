// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_REF_H_
#define PHYSIS_RUNTIME_RUNTIME_REF_H_

#include "runtime/runtime_common.h"
#include "runtime/runtime.h"

namespace physis {
namespace runtime {

template <class GridSpaceType>
class RuntimeRef: public Runtime<GridSpaceType> {
 public:
  RuntimeRef() {}
  virtual ~RuntimeRef() {}
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RUNTIME_REF_H_ */

