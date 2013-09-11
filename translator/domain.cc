// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/domain.h"

#include <utility>

#include "translator/rose_util.h"
#include "translator/grid.h"

namespace physis {
namespace translator  {

const std::string Domain::name = "Domain";

Domain::Domain(const Domain &d)
    : num_dims_(d.num_dims_),
      has_static_constant_size_(d.has_static_constant_size_),
      regular_domain_(d.regular_domain_) {
}

bool Domain::isDomainType(const SgType *type) {
  const SgNamedType *named_type = isSgNamedType(type);
  if (named_type) {
    string type_name = named_type->get_name().getString();
    if (type_name == PS_DOMAIN_INTERNAL_TYPE_NAME ||
        type_name == PSF_DOMAIN_TYPE_NAME) {
      return true;
    }
  }
  return getNumDimOfDomain(type) > 0;
}

Domain* Domain::GetDomain(int dim) {
  return new Domain(dim);
}

Domain* Domain::GetDomain(SgFunctionCallExp *exp) {
  LOG_DEBUG() << "finding domain object for "
              << exp->unparseToString() << "\n";

  SgType *t = exp->get_type();
  int numDim = getNumDimOfDomain(t);
  assert(numDim > 0);

  SizeVector v;
  if (rose_util::copyConstantFuncArgs<size_t>(exp, v)) {
    LOG_DEBUG() << "Constant domain detected\n";
    RegularDomain r(numDim);
    for (int i = 0; i < numDim; ++i) {
      r.min_point_.push_back(v[i*2]);
      r.max_point_.push_back(v[i*2+1]);
    }
    return new Domain(r);
  }
  return GetDomain(numDim);
}

SgExpression *Domain::GetDomainExpFromDomainFunc(SgFunctionCallExp *call) {
  assert(call);
  SgExpression *callee = call->get_function();
  // LOG_DEBUG() << "callee: " << callee->class_name() << "\n";
  SgPointerDerefExp *deref = isSgPointerDerefExp(callee);
  if (!deref) return NULL;
  LOG_DEBUG() << deref->get_type()->unparseToString() << "\n";
  SgDotExp *memberRef = isSgDotExp(deref->get_operand());
  if (!memberRef) return NULL;
  SgExpression *base = memberRef->get_lhs_operand();
  if (!isDomainType(base->get_type())) return NULL;
  return base;
}

bool Domain::isCallToDomainFunc(SgFunctionCallExp *call) {
  SgExpression *base = Domain::GetDomainExpFromDomainFunc(call);
  return base != NULL;
}

string Domain::toString() {
  ostringstream ss;
  ss << num_dims_ << "-D domain";
  if (has_static_constant_size()) {
    StringJoin sj(", ");
    for (int i = 0; i < num_dims_; ++i) {
      sj << regular_domain_.min_point_[i] << ":"
         << regular_domain_.max_point_[i];
    }
    ss << "(" << sj << ")";
  }

  return ss.str();
}

string Domain::getRealFuncName(const string &funcName,
                               const string &kernelName) const {
  ostringstream ss;
  ss << "grid_" << funcName
     << num_dims_ << "d_"
     << kernelName;
  return ss.str();
}

void Domain::Merge(const Domain &dom) {
  PSAssert(num_dims_ == dom.num_dims_);
  if (has_static_constant_size_ && dom.has_static_constant_size_ &&
      regular_domain_ == dom.regular_domain_) {
    LOG_DEBUG() << "Exactly same static constant domain\n";
  } else if (has_static_constant_size_) {
    LOG_DEBUG() << "Conservatively assuming non-static domain\n";    
    has_static_constant_size_ = false;
  }
}
} // namespace translator
} // namespace physis

