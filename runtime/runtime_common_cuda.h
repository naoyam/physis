// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_RUNTIME_COMMON_CUDA_H_
#define PHYSIS_RUNTIME_RUNTIME_COMMON_CUDA_H_

#include "common/config.h"
#include "runtime/runtime_common.h"
#include <cuda_runtime.h>
#include <cutil.h>

namespace physis {
namespace runtime {

//! Check the CUDA SM capability.
/*!
  Check the CUDA SM capability of the current device satisify a given
  version requirement.

  Based on the cutil library of the NVIDIA CUDA SDK, but modified not
  to display device information to stdout.
  
  \param major The major version number.
  \param minorThe minor version number.
  \return True if the given version requirement is met.
 */
inline bool CheckCudaCapabilities(int major, int minor) {
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  int dev;

  CUDA_SAFE_CALL( cudaGetDevice(&dev) );
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, dev));

  if((deviceProp.major > major) ||
     (deviceProp.major == major && deviceProp.minor >= minor))
  {
    //fprintf(stderr, "> Device %d: <%16s >, Compute SM %d.%d detected\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
    return true;
  } else {
    fprintf(stderr,
            "There is no device supporting CUDA compute capability %d.%d.\n",
            major, minor);
    return false;
  }
}

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RUNTIME_COMMON_CUDA_H_ */
