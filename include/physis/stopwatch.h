// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_STOPWATCH_H_
#define PHYSIS_STOPWATCH_H_

#if defined(unix) || defined(__unix__) || defined(__unix) || \
  defined(__APPLE__)
#include <sys/time.h>
#include <time.h>

typedef struct {
  struct timeval tv;
} __PSStopwatch;

#else
#error "Unknown environment"
#endif

static inline void __PSStopwatchQuery(__PSStopwatch *w) {
  gettimeofday(&(w->tv), NULL);
  return;
}

// returns mili seconds
static inline float __PSStopwatchDiff(const __PSStopwatch *begin,
                                      const __PSStopwatch *end) {
  return (end->tv.tv_sec - begin->tv.tv_sec) * 1000.0f
      + (end->tv.tv_usec - begin->tv.tv_usec) / 1000.0f;
}

static __inline void __PSStopwatchStart(__PSStopwatch *w) {
  __PSStopwatchQuery(w);
  return;
}
    
static __inline float __PSStopwatchStop(__PSStopwatch *w) {
  __PSStopwatch now;
  __PSStopwatchQuery(&now);
  return __PSStopwatchDiff(w, &now);
}

#endif /* PHYSIS_STOPWATCH_H_ */
