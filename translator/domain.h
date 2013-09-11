// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_DOMAIN_H_
#define PHYSIS_TRANSLATOR_DOMAIN_H_

#include <set>

#include "translator/translator_common.h"
#include "translator/stencil_range.h"

namespace physis {
namespace translator {

// Returns number of dimensions if t is a name of domain types.
// Otherwise, a negative value is returned.
inline int getNumDimOfDomain(const string &t) {
  if (t == PS_DOMAIN1D_TYPE_NAME) {
    return 1;
  } else if (t == PS_DOMAIN2D_TYPE_NAME) {
    return 2;
  } else if (t == PS_DOMAIN3D_TYPE_NAME) {
    return 3;
  } else {
    return -1;
  }
}

inline int getNumDimOfDomain(const SgType *type) {
  const SgNamedType *namedType = isSgNamedType(type);
  if (!namedType) return -1;
  return getNumDimOfDomain(namedType->get_name().getString());
}

class Domain;

class RegularDomain {
  friend class Domain;
  int num_dims_;
  SizeVector min_point_;
  SizeVector max_point_;
 public:  
  RegularDomain(int num_dims): num_dims_(num_dims) {}
  RegularDomain(const RegularDomain &r):
      num_dims_(r.num_dims_), min_point_(r.min_point_),
      max_point_(r.max_point_) {}
  const SizeVector &min_point() const { return min_point_; }
  const SizeVector &max_point() const { return max_point_; }
  int num_dims() const { return num_dims_; }
  bool operator==(const RegularDomain &rd) const {
    return num_dims_ == rd.num_dims_ &&
        min_point_ == rd.min_point_ &&
        max_point_ == rd.max_point_;
  }
};

class Domain: public AstAttribute {
 public:
 private:
  const int num_dims_;
  bool has_static_constant_size_;
  const RegularDomain regular_domain_;
  explicit Domain(int num_dims)
      : num_dims_(num_dims), has_static_constant_size_(false),
        regular_domain_(num_dims_) {}
  explicit Domain(const RegularDomain &r)
      : num_dims_(r.num_dims()),
        has_static_constant_size_(true), regular_domain_(r) {}
  // tatic Domain *getStaticDomain(DeclRefExpr *exp);
  Domain(const Domain &d);
 public:
  static const std::string name;
  Domain *copy() {
    return new Domain(*this);
  }
  
  static Domain *GetDomain(int numDim);
  static Domain *GetDomain(SgFunctionCallExp *callToDomNew);

  static SgExpression *GetDomainExpFromDomainFunc(SgFunctionCallExp *call);
  static bool isCallToDomainFunc(SgFunctionCallExp *call);

  static bool isDomainType(const SgType *type);

  virtual ~Domain() {}
  virtual string toString();
  virtual int num_dims() const { return num_dims_; }
  virtual string getRealFuncName(const string &funcName,
                                 const string &kernelName) const;
  virtual bool has_static_constant_size() const {
    return has_static_constant_size_;
  }
  virtual const RegularDomain &regular_domain() const { return regular_domain_; }
  //! Merges multiple domain definitions
  virtual void Merge(const Domain &dom);
};

typedef std::set<Domain*> DomainSet;

// Why this?
class Grid;

} // namespace translator
} // namespace physis



#endif /* DOMAIN_H */

