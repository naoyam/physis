// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)


#include "runtime/runtime_common.h"
#include "runtime/runtime_common_cuda.h"
#include "runtime/cuda_util.h"
#include "runtime/reduce.h"
#include "physis/physis_cuda.h"

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace {

//! Reduce a grid with binary operation op.
template <class T>
void PSReduceGridTemplate(void *buf, PSReduceOp op,
                          __PSGrid *g) {
  thrust::device_ptr<T> dev_ptr((T*)(g->p0));
  T *out = (T*)buf;
  if (op == PS_MAX) {
    *out = *thrust::max_element(dev_ptr, dev_ptr + g->num_elms);
  } else if (op == PS_MIN) {
    *out = *thrust::min_element(dev_ptr, dev_ptr + g->num_elms);
  } else if (op == PS_SUM) {
    *out = thrust::reduce(dev_ptr, dev_ptr + g->num_elms,
                       physis::runtime::GetReductionDefaultValue<T>(op),
                       thrust::plus<T>());
  } else if (op == PS_PROD) {
    *out = thrust::reduce(dev_ptr, dev_ptr + g->num_elms,
                       physis::runtime::GetReductionDefaultValue<T>(op),
                       thrust::multiplies<T>());
  } else {
    PSAbort(1);
  }
  //LOG_DEBUG() << "Reduction: " << *out << "\n";
  return;
}

}

#ifdef __cplusplus
extern "C" {
#endif

  void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                           __PSGrid *g) {
    PSReduceGridTemplate<float>(buf, op, g);
  }

  void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                            __PSGrid *g) {
    PSReduceGridTemplate<double>(buf, op, g);    
  }

#ifdef __cplusplus
}
#endif
