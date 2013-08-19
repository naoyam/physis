// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "runtime/runtime_common.h"

#include <string>

FILE *__ps_trace;

using std::string;

namespace physis {
namespace runtime {


} // namespace runtime
} // namespace physis



#ifdef __cplusplus
extern "C" {
#endif

  //! Initialize random numbers.
  /*!  
   * \param[in] n ... number of random values
   * \return    random handle
   */
  void *__PSRandomInit(int n) {
    struct timeval tv;
    int i, *handle = (int *)calloc(n, sizeof(int));
    gettimeofday(&tv, NULL);
    srandom(tv.tv_usec);
    for (i = 0; i < n; ++i) {
      handle[i] = i;
    }
    for (i = 0; i < n; ++i) {
      int t, r = random() % n;
      t = handle[r];
      handle[r] = handle[i];
      handle[i] = t;
    }
    return (void *)handle;
  }

#ifdef __cplusplus
}
#endif
