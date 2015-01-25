// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_RUNTIME_REDUCE_H_
#define PHYSIS_RUNTIME_REDUCE_H_

#include <functional>
#include <boost/function.hpp>
#include <float.h>
#include <limits.h>

namespace physis {
namespace runtime {

template <class T>
struct MaxOp: public std::binary_function<T, T, T> {
  T operator()(T x, T y) {
    return (x > y) ? x : y;
  }
};
  
template <class T>
struct MinOp: public std::binary_function<T, T, T> {
  T operator()(T x, T y) {
    return (x < y) ? x : y;
  }
};

template <class T>
boost::function<T (T, T)> GetReducer(PSReduceOp op) {
  boost::function<T (T, T)> func;  
  switch (op) {
    case PS_MAX:
      func = MaxOp<T>();
      break;
    case PS_MIN:
      func = MinOp<T>();
      break;
    case PS_SUM:
      func = std::plus<T>();
      break;
    case PS_PROD:
      func = std::multiplies<T>();
      break;
    default:
      PSAbort(1);
      break;
  }
  return func;
}

template <class T>
T GetReductionDefaultValue(PSReduceOp op) {
  T x;
  return x;
}

template <> inline
float GetReductionDefaultValue<float>(PSReduceOp op) {
  float v;
  switch (op) {
    case PS_MAX:
      v = FLT_MIN;
      break;
    case PS_MIN:
      v = FLT_MAX;
      break;
    case PS_SUM:
      v = 0.0f;
      break;
    case PS_PROD:
      v = 1.0f;
      break;
    default:
      PSAbort(1);
      break;
  }
  return v;
}

template <> inline
double GetReductionDefaultValue<double>(PSReduceOp op) {
  double v;
  switch (op) {
    case PS_MAX:
      v = DBL_MIN;
      break;
    case PS_MIN:
      v = DBL_MAX;
      break;
    case PS_SUM:
      v = 0.0;
      break;
    case PS_PROD:
      v = 1.0;
      break;
    default:
      PSAbort(1);
      break;
  }
  return v;
}

template <> inline
int GetReductionDefaultValue<int>(PSReduceOp op) {
  int v;
  switch (op) {
    case PS_MAX:
      v = INT_MIN;
      break;
    case PS_MIN:
      v = INT_MAX;
      break;
    case PS_SUM:
      v = 0;
      break;
    case PS_PROD:
      v = 1;
      break;
    default:
      PSAbort(1);
      break;
  }
  return v;
}

template <> inline
long GetReductionDefaultValue<long>(PSReduceOp op) {
  long v;
  switch (op) {
    case PS_MAX:
      v = LONG_MIN;
      break;
    case PS_MIN:
      v = LONG_MAX;
      break;
    case PS_SUM:
      v = 0L;
      break;
    case PS_PROD:
      v = 1L;
      break;
    default:
      PSAbort(1);
      break;
  }
  return v;
}
 

} // namespace runtime
} // namespace physis

#endif /* PHYSIS_RUNTIME_REDUCE_H_ */
