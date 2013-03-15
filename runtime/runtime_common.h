// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_RUNTIME_COMMON_H_
#define PHYSIS_RUNTIME_RUNTIME_COMMON_H_

#define PHYSIS_RUNTIME

//#ifndef ENABLE_LOG_DEBUG
//#define ENABLE_LOG_DEBUG
//#endif
#ifdef ENABLE_LOG_DEBUG
#undef ENABLE_LOG_DEBUG
#endif
#ifdef ENABLE_LOG_VERBOSE
#undef ENABLE_LOG_VERBOSE
#endif
#include "physis/physis_util.h"
#include "physis/physis_common.h"
#include "physis/internal_common.h"
#include "common/config.h"

namespace physis {
namespace runtime {

struct Width2 {
  UnsignedArray bw;
  UnsignedArray fw;
};

typedef void (*__PSStencilRunClientFunction)(int, void **);



} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_RUNTIME_COMMON_H_ */
