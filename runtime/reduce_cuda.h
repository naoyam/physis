// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)
#ifndef PHYSIS_RUNTIME_REDUCE_CUDA_H_
#define PHYSIS_RUNTIME_REDUCE_CUDA_H_
#include "runtime/runtime_common.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/cuda_util.h"
#include "runtime/reduce.h"
//#include "physis/physis_cuda.h"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace physis {
namespace runtime {

//! Reduce a grid with binary operation op.
template <class T>
void ReduceGridCUDA(void *buf, PSReduceOp op,
                    void *dev_grid, size_t len) {
  thrust::device_ptr<T> dev_ptr((T*)dev_grid);
  T *out = (T*)buf;
  if (op == PS_MAX) {
    *out = *thrust::max_element(dev_ptr, dev_ptr + len);
  } else if (op == PS_MIN) {
    *out = *thrust::min_element(dev_ptr, dev_ptr + len);
  } else if (op == PS_SUM) {
    *out = thrust::reduce(dev_ptr, dev_ptr + len,
                       physis::runtime::GetReductionDefaultValue<T>(op),
                       thrust::plus<T>());
  } else if (op == PS_PROD) {
    *out = thrust::reduce(dev_ptr, dev_ptr + len,
                       physis::runtime::GetReductionDefaultValue<T>(op),
                       thrust::multiplies<T>());
  } else {
    PSAbort(1);
  }
  return;
}

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_REDUCE_CUDA_H_ */
