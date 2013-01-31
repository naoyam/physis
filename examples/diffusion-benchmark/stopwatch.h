// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef BENCHMARKS_COMMON_STOPWATCH_H_
#define BENCHMARKS_COMMON_STOPWATCH_H_

#if defined(unix) || defined(__unix__) || defined(__unix) || \
  defined(__APPLE__)
#include <sys/time.h>
#include <time.h>

typedef struct {
  struct timeval tv;
} Stopwatch;

#else
#error "Unknown environment"
#endif

static inline void StopwatchQuery(Stopwatch *w) {
  gettimeofday(&(w->tv), NULL);
  return;
}

static inline float StopwatchDiff(const Stopwatch *begin,
                                  const Stopwatch *end) {
  return (end->tv.tv_sec - begin->tv.tv_sec)
      + (end->tv.tv_usec - begin->tv.tv_usec) * 1.0e-06;
}

static inline void StopwatchStart(Stopwatch *w) {
  StopwatchQuery(w);
  return;
}
    
static inline float StopwatchStop(Stopwatch *w) {
  Stopwatch now;
  StopwatchQuery(&now);
  return StopwatchDiff(w, &now);
}

#endif /* BENCHMARKS_COMMON_STOPWATCH_H_ */
