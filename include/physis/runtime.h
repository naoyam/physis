// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_RUNTIME_H_
#define PHYSIS_RUNTIME_H_

#include <stdio.h>

#include "physis/stopwatch.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

extern FILE *__ps_trace;

static inline void __PSTraceStencilPre(const char *msg) {
  if (__ps_trace) {
    fprintf(__ps_trace, "Physis: Stencil started (%s)\n", msg);
  }
  return;
}

  static inline void __PSTraceStencilPost(float time) {
  if (__ps_trace) {
    fprintf(__ps_trace, "Physis: Stencil finished (time: %f)\n", time);
  }
  return;
}

#ifdef AUTO_TUNING  
  /**  initialize random
   * @param[in] n ... number of randomized value
   * @return    random handle
   */
  extern void *__PSRandomInit(int n);
  /** get randomized value
   * @param[in] handle ... random handle
   * @param[in] count ... index of randomized value
   * @return    randomized value
   */
  static inline int __PSRandom(void *handle, int count) {
    return ((int *)handle)[count];
  }
  /** finalize random
   * @param[in] handle ... random handle
   */
  static inline void __PSRandomFini(void *handle) {
    free(handle);
  }
#endif
  
#ifdef __cplusplus
}
#endif /* __cplusplus */
  

#endif /* PHYSIS_RUNTIME_H_ */
