// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_INTERNAL_COMMON_H_
#define PHYSIS_INTERNAL_COMMON_H_

#include <assert.h>
#include <numeric>
#include <functional>

#include <boost/array.hpp>

#include "physis/physis_common.h"
#include "physis/physis_util.h"

namespace physis {
namespace util {

// REFACTORING: rename to IndexArray. Create new IntArray type with
// plain int type.
class IntArray: public boost::array<index_t, PS_MAX_DIM> {
  typedef boost::array<index_t, PS_MAX_DIM> pt;
  typedef index_t elmt;
 public:  
  IntArray() {
    assign(0);
  }
  IntArray(elmt x) {
    assign(0);
    (*this)[0] = x;
  }
  IntArray(elmt x, elmt y) {
    assign(0);
    (*this)[0] = x;
    (*this)[1] = y;
  }
  IntArray(elmt x, elmt y, elmt z) {
    assign(0);    
    (*this)[0] = x;
    (*this)[1] = y;
    (*this)[2] = z;
  }
  IntArray(const PSVectorInt v) {
    assign(0);        
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = v[i];
    }
  }
  IntArray(const PSIndexVector v) {
    assign(0);        
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = v[i];
    }
  }
  IntArray(const int *v, int len) {
    assign(0);
    assert(len < PS_MAX_DIM);
    for (int i = 0; i < len; ++i) {
      (*this)[i] = v[i];
    }
  }
  elmt accumulate(int len) const {
    elmt v = 1;
    FOREACH (it, begin(), begin() + len) {
      v *= *it;
    }
    return v;
  }
  std::ostream& print(std::ostream &os) const {
    physis::StringJoin sj;
    FOREACH(i, begin(), end()) {
      sj << *i;
    }
    os << "{" << sj << "}";
    return os;
  }
  IntArray operator+(const IntArray &x) const {
    IntArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] + x[i];
    }
    return ret;
  }
  IntArray operator+(const elmt &x) const {
    IntArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] + x;
    }
    return ret;
  }
  IntArray operator-(const IntArray &x) const {
    IntArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] - x[i];
    }
    return ret;
  }    
  IntArray operator-(const elmt &x) const {
    IntArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] - x;
    }
    return ret;
  }
  bool operator>(const elmt &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] > x)) return false;
    }
    return true;
  }
  bool operator<(const elmt &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] < x)) return false;
    }
    return true;
  }
  bool operator>=(const elmt &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] >= x)) return false;
    }
    return true;
  }
  bool operator<=(const elmt &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] <= x)) return false;
    }
    return true;
  }
  bool operator==(const elmt &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] == x)) return false;
    }
    return true;
  }
  bool operator!=(const elmt &x) const {
    return !(*this == x);
  }
  IntArray &operator+=(const IntArray &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] += x[i];
    }
    return *this;
  }
  IntArray &operator-=(const IntArray &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] -= x[i];
    }
    return *this;
  }
  void SetNoLessThan(const elmt &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::max((*this)[i], x);
    }
  }
  void SetNoLessThan(const IntArray &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::max((*this)[i], x[i]);
    }
  }
  void SetNoMoreThan(const elmt &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::min((*this)[i], x);
    }
  }
  void SetNoMoreThan(const IntArray &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::min((*this)[i], x[i]);
    }
  }
  void Set(index_t *buf) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      buf[i] = (*this)[i];
    }
  }
  bool LessThan(const IntArray &x, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
      if (!((*this)[i] < x[i])) return false;
    }
    return true;
  }
  bool GreaterThan(const IntArray &x, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
      if (!((*this)[i] > x[i])) return false;
    }
    return true;
  }
  
};

// REFACTORING: rename to IndexVector
typedef std::vector<index_t> IntVector;

} // namespace util
} // namespace physis

inline std::ostream &operator<<(std::ostream &os,
                                const physis::util::IntArray &x) {
  return x.print(os);
}

inline std::ostream &operator<<(std::ostream &os,
                                const physis::util::IntVector &x) {
  physis::StringJoin sj;
  FOREACH (i, x.begin(), x.end()) { sj << *i; }
  os << "{" << sj << "}";  
  return os;
}

#endif /* PHYSIS_INTERNAL_COMMON_H_ */
