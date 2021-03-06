// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_RUNTIME_COMMON_CUDA_H_
#define PHYSIS_RUNTIME_RUNTIME_COMMON_CUDA_H_

#include "common/config.h"
#include "runtime/runtime_common.h"
#include <cuda_runtime.h>

inline void wait_for_attach() {
  while (1) {
    sleep(5);                                               
  }
}

#define CUDA_SAFE_CALL(x) do {                                  \
    cudaError_t e = x;                                          \
    if (e != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA ERROR at " __FILE__ "#%d: %s\n",    \
              __LINE__, cudaGetErrorString(e));                 \
      fprintf(stderr, "PID: %d\n", getpid());                   \
      if (getenv("CUDA_SAFE_CALL_WAIT")) {                      \
        wait_for_attach();                                      \
      }                                                         \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  } while (0)

#define CUDA_CHECK_ERROR(msg) do {                                      \
    cudaError_t e = cudaGetLastError();                                 \
    if (e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA ERROR: %s at " __FILE__ "#%d: %s\n",        \
              msg, __LINE__, cudaGetErrorString(e));                    \
    }                                                                   \
    e = cudaDeviceSynchronize();                                        \
    if (e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA ERROR: %s at " __FILE__ "#%d: %s\n",        \
              msg, __LINE__, cudaGetErrorString(e));                    \
    }                                                                   \
  } while (0)

#define CUDA_DEVICE_INIT(devid) do {                            \
    cudaDeviceProp dp;                                          \
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&dp, devid));        \
    LOG_INFO() << "Using device " << devid                      \
               << ": " << dp.name << "\n";                      \
    CUDA_SAFE_CALL(cudaSetDevice(devid));                       \
  } while (0)

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
