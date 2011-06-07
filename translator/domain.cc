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

bool Domain::isDomainType(const SgType *type) {
  const SgNamedType *named_type = isSgNamedType(type);
  if (named_type) {
    if (named_type->get_name().getString() == PSDOMAIN_TYPE_NAME) {
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

  vector<size_t> v;
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

} // namespace translator
} // namespace physis

