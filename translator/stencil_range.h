// Licensed under the BSD license. See LICENSE.txt for more details.

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
  StencilIndex(int dim, ssize_t offset): dim(dim), offset(offset) {}  
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

void StencilIndexListInitSelf(StencilIndexList &sil, unsigned num_dim);

int StencilIndexListFindDim(const StencilIndexList *sil, int dim);


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
bool StencilIndexRegularOrder(const StencilIndexList &sil);
void StencilIndexListClearOffset(StencilIndexList &sil);

class StencilRegularIndexList {
 public:
  typedef std::map<int, ssize_t> map_t;
 private:
  map_t indices_;
 public:
  StencilRegularIndexList() {}
  StencilRegularIndexList(const StencilIndexList &sil) {
    PSAssert(StencilIndexRegularOrder(sil));
    FOREACH (it, sil.begin(), sil.end()) {
      indices_[it->dim] = it->offset;
    }
  }
  ssize_t GetIndex(int dim) const;
  void SetIndex(int dim, ssize_t index);
  int GetNumDims() const { return indices_.size(); }
  virtual ~StencilRegularIndexList() {}
  const std::map<int, ssize_t>& indices() const { return indices_; }  
  bool operator<(const StencilRegularIndexList &x) const {
    FOREACH (it, indices_.begin(), indices_.end()) {
      if (it->second < x.GetIndex(it->first)) {
        return true;
      } else if (it->second > x.GetIndex(it->first)) {
        return false;
      }
    }
    return false;
  }
  bool operator==(const StencilRegularIndexList &x) const {
    return indices_ == x.indices_;
  }
  bool operator==(const StencilIndexList &x) const {
    if (!StencilIndexRegularOrder(x)) return false;
    StencilRegularIndexList xr(x);
    return operator==(xr);
  }
  bool operator!=(const StencilIndexList &x) const {
    return !operator==(x);
  }
  std::ostream &print(std::ostream &os) const {
    StringJoin sj;
    FOREACH (it, indices_.begin(), indices_.end()) {
      sj << it->first << ": " << it->second;
    }
    os << "{" << sj << "}";
    return os;
  }
};

class StencilRange {
  StencilIndexList min_indices_[PS_MAX_DIM];
  StencilIndexList max_indices_[PS_MAX_DIM];
  std::vector<StencilIndexList> all_indices_;
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
  bool GetNeighborAccess(IntVector &offset_min, IntVector &offset_max) const;
  bool IsNeighborAccessDiagonalAccessed() const;
  bool IsZero() const;
  // Returns true if indices with a unique dimension is given for each
  // dimension. 
  bool IsUniqueDim() const;
  SgVariableDeclaration *BuildPSGridRange(
      std::string name, SgScopeStatement *block);
  int GetMaxWidth() const;
  const std::vector<StencilIndexList> &all_indices() const {
    return all_indices_;
  }
  int IsEmpty() const {
    return min_indices_[0].size() == 0;
  }
};

class StencilIndexVarAttribute: public AstAttribute {
 public:
  static const std::string name;  
  StencilIndexVarAttribute(int dim): dim_(dim) {}
  StencilIndexVarAttribute(const StencilIndexVarAttribute &x): dim_(x.dim_) {}  
  virtual ~StencilIndexVarAttribute() {}
  int dim() { return dim_; }
  AstAttribute *copy() {
    return new StencilIndexVarAttribute(*this);
  }

 protected:
  int dim_;
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

inline std::ostream& operator<<(
    std::ostream &os,
    const physis::translator::StencilRegularIndexList &x) {
  return x.print(os);
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
