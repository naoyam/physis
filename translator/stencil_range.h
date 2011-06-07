// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_STENCIL_RANGE_H_
#define PHYSIS_TRANSLATOR_STENCIL_RANGE_H_

#include <algorithm>
#include <utility>
#include "translator_common.h"

namespace physis {
namespace translator {

struct StencilIndex {
  int dim;
  ssize_t offset;
  StencilIndex(): dim(0), offset(0) {}
  StencilIndex(const StencilIndex &si): dim(si.dim), offset(si.offset) {}  
  std::ostream &print(std::ostream &os) const {
    os << "(" << dim << ", " << offset << ")";
    return os;
  }
  std::string ToString() const {
    std::stringstream ss;
    ss << "(" << dim << ", " << offset << ")";
    return ss.str();
  }
  bool operator>(const StencilIndex &x) const {
    PSAssert(dim == x.dim);
    return offset > x.offset;
  }
  bool operator>=(const StencilIndex &x) const {
    PSAssert(dim == x.dim);
    return offset >= x.offset;
  }
  bool operator<(const StencilIndex &x) const {
    PSAssert(dim == x.dim);
    return offset < x.offset;
  }
  bool operator<=(const StencilIndex &x) const {
    PSAssert(dim == x.dim);
    return offset <= x.offset;
  }
  bool operator==(const StencilIndex &x) const {
    return dim == x.dim && offset == x.offset;
  }
  bool operator!=(const StencilIndex &x) const {
    return !((*this) == x);
  }
  StencilIndex &operator=(const StencilIndex &x) {
    dim = x.dim;
    offset = x.offset;
    return *this;
  }
};


typedef std::vector<StencilIndex> StencilIndexList;

class StencilIndexAttribute: public AstAttribute {
 public:
  StencilIndexAttribute(const StencilIndexList &sil)
      : stencil_index_list_(sil) {}
  const StencilIndexList &stencil_index_list() { return stencil_index_list_; }
  static const std::string name;
  StencilIndexAttribute &operator=(const StencilIndexAttribute &sia) {
    stencil_index_list_ = sia.stencil_index_list_;
    return *this;
  }
  AstAttribute *copy() {
    return new StencilIndexAttribute(stencil_index_list_);
  }
 protected:
  StencilIndexList stencil_index_list_;
};

bool StencilIndexSelf(const StencilIndexList &sil, unsigned num_dims);
bool StencilIndexRegularOrder(const StencilIndexList &sil, unsigned num_dims);



class StencilRange {
  StencilIndexList min_indices_[PS_MAX_DIM];
  StencilIndexList max_indices_[PS_MAX_DIM];
  int num_dims_;
  bool diagonal_;
  void insert(int dim, const StencilIndex &si);
  
 public:
  explicit StencilRange(int dim) : num_dims_(dim), diagonal_(false) {
    PSAssert(num_dims_ > 0 && num_dims_ <= PS_MAX_DIM);
  }

  StencilRange(const StencilRange &sr):
      min_indices_(sr.min_indices_),
      max_indices_(sr.max_indices_),
      num_dims_(sr.num_dims_), diagonal_(sr.diagonal_)  {}

  void insert(const StencilIndexList &stencil_indices);
  void merge(const StencilRange &sr);
  std::ostream &print(std::ostream &os) const;
  int num_dims() const { return num_dims_; }
  StencilIndexList* min_indices() { return min_indices_; }
  StencilIndexList* max_indices() { return max_indices_; }  
  bool IsNeighborAccess() const;
  bool GetNeighborAccess(IntVector &forward, IntVector &backward);
  bool IsNeighborAccessDiagonalAccessed() const;
  bool IsZero() const;
  // Returns true if indices with a unique dimension is given for each
  // dimension. 
  bool IsUniqueDim() const;
  SgVariableDeclaration *BuildPSGridRange(
      std::string name, SgScopeStatement *block);
  int GetMaxWidth() const;
};

} // namespace translator
} // namespace physis

inline std::ostream &operator<<(std::ostream &os, 
                                const physis::translator::StencilIndex &si) {
  return si.print(os);
}

inline std::ostream& operator<<(std::ostream &os,
                                const physis::translator::StencilRange &sr) {
  return sr.print(os);
}

inline std::ostream& operator<<(std::ostream &os,
                                const physis::translator::StencilIndexList &sil) {
  physis::StringJoin sj;
  FOREACH (it, sil.begin(), sil.end()) {
    sj << *it;
  }
  os << "{" << sj.str() << "}";
  return os;
}


#endif /* PHYSIS_TRANSLATOR_STENCIL_RANGE_H_ */
