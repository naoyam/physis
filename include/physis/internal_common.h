// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_INTERNAL_COMMON_H_
#define PHYSIS_INTERNAL_COMMON_H_

#include <assert.h>
#include <numeric>
#include <functional>

#include <boost/array.hpp>
#include <boost/foreach.hpp>

#include "physis/physis_common.h"
#include "physis/physis_util.h"

#define PS_XFREE(p) do {                        \
    if (p) {                                    \
      free(p);                                  \
      p = NULL;                                 \
    } } while (0)

#define PS_XDELETE(p) do {                        \
    delete (p);                                   \
    p = NULL;                                     \
  } while (0)

#define PS_XDELETEA(p) do {                         \
    delete[] (p);                                   \
    p = NULL;                                       \
  } while (0)

namespace physis {

template <typename ty>
class IntegerArray: public boost::array<ty, PS_MAX_DIM> {
  // TODO: necessary?
  //typedef boost::array<c, PS_MAX_DIM> pt;
  //typedef ty elmt;
 public:  
  IntegerArray() {
    this->assign(0);
  }
  explicit IntegerArray(ty x) {
    this->assign(0);
    (*this)[0] = x;
  }
  explicit IntegerArray(ty x, ty y) {
    this->assign(0);
    (*this)[0] = x;
    (*this)[1] = y;
  }
  explicit IntegerArray(ty x, ty y, ty z) {
    this->assign(0);    
    (*this)[0] = x;
    (*this)[1] = y;
    (*this)[2] = z;
  }
#if 0  
  IntegerArray(const PSVectorInt v) {
    this->assign(0);        
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = v[i];
    }
  }

  IntegerArray(const PSIndexVector v) {
    this->assign(0);        
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = v[i];
    }
  }

  IntegerArray(const int *v, int len) {
    this->assign(0);
    assert(len < PS_MAX_DIM);
    for (int i = 0; i < len; ++i) {
      (*this)[i] = v[i];
    }
  }
#endif
  template <typename ty2>
  IntegerArray(const ty2 *v) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = (ty)v[i];
    }
  }
  template <typename ty2>
  IntegerArray(const IntegerArray<ty2> &v) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = (ty)v[i];
    }
  }
  void Set(ty v) {
    this->assign(v);
  }
  ty accumulate(int len) const {
    ty v = 1;
    BOOST_FOREACH (ty i, std::make_pair(this->begin(), this->begin() + len)) {
      v *= i;
    }
    return v;
  }
  std::ostream& print(std::ostream &os) const {
    physis::StringJoin sj;
    BOOST_FOREACH (ty i, std::make_pair(this->begin(), this->end())) {
      sj << i;
    }
    os << "{" << sj << "}";
    return os;
  }
  IntegerArray<ty> operator+(const IntegerArray<ty> &x) const {
    IntegerArray<ty> ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] + x[i];
    }
    return ret;
  }
  IntegerArray<ty> operator+(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] + x;
    }
    return ret;
  }
  IntegerArray<ty> operator-(const IntegerArray<ty> &x) const {
    IntegerArray<ty> ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] - x[i];
    }
    return ret;
  }    
  IntegerArray<ty> operator-(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] - x;
    }
    return ret;
  }
  IntegerArray<ty> operator*(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] * x;
    }
    return ret;
  }
  IntegerArray<ty> operator/(const ty &x) const {
    IntegerArray ret;
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      ret[i] = (*this)[i] / x;
    }
    return ret;
  }
  bool operator>(const ty &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] > x)) return false;
    }
    return true;
  }
  bool operator<(const ty &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] < x)) return false;
    }
    return true;
  }
  bool operator>=(const ty &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] >= x)) return false;
    }
    return true;
  }
  bool operator<=(const ty &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] <= x)) return false;
    }
    return true;
  }
  bool operator==(const ty &x) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      if (!((*this)[i] == x)) return false;
    }
    return true;
  }
  bool operator!=(const ty &x) const {
    return !(*this == x);
  }
  IntegerArray<ty> &operator+=(const IntegerArray<ty> &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] += x[i];
    }
    return *this;
  }
  IntegerArray<ty> &operator-=(const IntegerArray<ty> &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] -= x[i];
    }
    return *this;
  }
  IntegerArray<ty> &operator*=(const IntegerArray<ty> &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] *= x[i];
    }
    return *this;
  }
  IntegerArray<ty> &operator/=(const IntegerArray<ty> &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] /= x[i];
    }
    return *this;
  }
  void SetNoLessThan(const ty &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::max((*this)[i], x);
    }
  }
  void SetNoLessThan(const IntegerArray<ty> &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::max((*this)[i], x[i]);
    }
  }
  void SetNoMoreThan(const ty &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::min((*this)[i], x);
    }
  }
  void SetNoMoreThan(const IntegerArray<ty> &x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = std::min((*this)[i], x[i]);
    }
  }
  void CopyTo(PSIndex *buf) const {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      buf[i] = (*this)[i];
    }
  }
  bool LessThan(const IntegerArray<ty> &x, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
      if (!((*this)[i] < x[i])) return false;
    }
    return true;
  }
  bool GreaterThan(const IntegerArray<ty> &x, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
      if (!((*this)[i] > x[i])) return false;
    }
    return true;
  }
  template <typename ty2>
  IntegerArray<ty> operator=(const IntegerArray<ty2> x) {
    for (int i = 0; i < PS_MAX_DIM; ++i) {
      (*this)[i] = (ty)x[i];
    }
    return *this;
  }
};

// REFACTORING: rename to IndexVector
typedef std::vector<PSIndex> IntVector;
typedef std::vector<size_t> SizeVector;

typedef IntegerArray<int> IntArray;
typedef IntegerArray<unsigned> UnsignedArray;
typedef IntegerArray<size_t> SizeArray;
typedef IntegerArray<ssize_t> SSizeArray;
typedef IntegerArray<PSIndex> IndexArray;

inline void CopyTypeInfo(__PSGridTypeInfo &dst,
                         const __PSGridTypeInfo src) {
  dst = src;
  if (src.members != NULL) {
    dst.members = new __PSGridTypeMemberInfo[dst.num_members];
    for (int i = 0; i < dst.num_members; ++i) {
      dst.members[i] = src.members[i];
    }
  }
}

inline void FreeTypeInfo(__PSGridTypeInfo *info) {
  delete[] info->members;
  delete info;
}

inline std::string ToString(PSType t) {
  std::string s;
  switch (t) {
    case PS_INT:
      s = "int";
      break;
    case PS_LONG:
      s = "long";
      break;
    case PS_FLOAT:
      s = "float";
      break;
    case PS_DOUBLE:
      s = "double";
      break;
    case PS_USER:
      s = "user";
    default:
      PSAbort(1);
  }
  return s;
}

} // namespace physis

template <typename ty>
inline std::ostream &operator<<(
    std::ostream &os,
    const physis::IntegerArray<ty> &x) {
  return x.print(os);
}

template <typename ty>
inline std::ostream &operator<<(std::ostream &os,
                                const std::vector<ty> &x) {
  physis::StringJoin sj;
  BOOST_FOREACH (ty i, x) { sj << i; }
  os << "{" << sj << "}";  
  return os;
}

#endif /* PHYSIS_INTERNAL_COMMON_H_ */
