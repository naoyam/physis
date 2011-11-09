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
#include "runtime/reduce_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

  void __PSReduceGridFloat(void *buf, enum PSReduceOp op,
                           __PSGrid *g) {
    physis::runtime::ReduceGridCUDA<float>(buf, op, g->p0, g->num_elms);
  }

  void __PSReduceGridDouble(void *buf, enum PSReduceOp op,
                            __PSGrid *g) {
    physis::runtime::ReduceGridCUDA<double>(buf, op, g->p0, g->num_elms);
  }

#ifdef __cplusplus
}
#endif
