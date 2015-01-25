// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_CUDA_HM_H_
#define PHYSIS_RUNTIME_RUNTIME_CUDA_HM_H_

#include "runtime/runtime_common.h"
#include "runtime/runtime_cuda.h"

namespace physis {
namespace runtime {

class RuntimeCUDAHM: public RuntimeCUDA {
 public:
  RuntimeCUDAHM();
  virtual ~RuntimeCUDAHM();
};

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RUNTIME_CUDA_HM_H_ */

