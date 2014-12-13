// Licensed under the BSD license. See LICENSE.txt for more details.

#include "runtime/runtime_mpi_cuda.h"
#include "runtime/rpc_cuda.h"
#include "runtime/runtime_common.h"
#include "runtime/runtime_common_cuda.h"

#include <cuda_runtime.h>

namespace physis {
namespace runtime {

void InitCUDA(int my_rank, int num_local_processes) {
  // Assumes each local process has successive process rank
  int dev_id = my_rank % num_local_processes;
  cudaDeviceProp dp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&dp, dev_id));
  LOG_INFO() << "Using device " << dev_id
             << ": " << dp.name << "\n";
  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_SAFE_CALL(cudaSetDevice(dev_id));
  CUDA_CHECK_ERROR("CUDA initialization");
  if (!physis::runtime::CheckCudaCapabilities(2, 0)) {
    PSAbort(1);
  }
  CUDA_SAFE_CALL(cudaStreamCreate(&stream_inner));
  CUDA_SAFE_CALL(cudaStreamCreate(&stream_boundary_copy));
  for (int i = 0; i < num_stream_boundary_kernel; ++i) {
    CUDA_SAFE_CALL(cudaStreamCreate(&stream_boundary_kernel[i]));
  }
  return;
}

int GetNumberOfLocalProcesses(int *argc, char ***argv) {
  std::vector<std::string> opts;
  std::string option_string = "physis-nlp";
  int nlp = 1; // default
  if (physis::runtime::ParseOption(argc, argv, option_string,
                                   1, opts)) {
    LOG_VERBOSE() << option_string << ": " << opts[1] << "\n";
    nlp = physis::toInteger(opts[1]);
  }
  LOG_DEBUG() << "Number of local processes: "  << nlp << "\n";
  return nlp;
}


} // namespace runtime
} // namespace physis

