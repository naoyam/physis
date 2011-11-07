// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/reduce.h"
#include "translator/rose_util.h"
#include "translator/grid.h"
#include "translator/translation_context.h"

namespace physis {
namespace translator {

const std::string Reduce::name = "Reduce";

//! Tells the kind  of a reduce call.
/*!
  The return value can be either GRID or KERNEL. It can be determined
  by looking at the 3rd parameter, which is a function name if it's
  kernel reduction. Otherwise, it's a grid reduction.
  
  \param call A call to PSReduce.
  \return A kind indicator value.
 */
static Reduce::KIND get_reduction_kind(SgFunctionCallExp *call) {
  int kernel_param_offset = 2;
  SgExpressionPtrList &args = call->get_args()->get_expressions();
  SgExpression *kernel_arg = args[kernel_param_offset];
  if (isSgFunctionRefExp(kernel_arg)) {
    return Reduce::KERNEL;
  } else {
    return Reduce::GRID;
  }
}

Reduce::Reduce(SgFunctionCallExp *call)
    : reduce_call_(call) {
  kind_ = get_reduction_kind(call);
}

Reduce::~Reduce() {
}

AstAttribute *Reduce::copy() {
  return new Reduce(reduce_call_);
}

bool Reduce::IsGrid() const {
  return kind_ == GRID;
}

bool Reduce::IsKernel() const {
  return kind_ == KERNEL;
}

bool Reduce::IsReduce(SgFunctionCallExp *call) {
  SgFunctionRefExp *f = isSgFunctionRefExp(call->get_function());
  if (!f) return false;
  SgName name = f->get_symbol()->get_name();
  return name == REDUCE_NAME;
}

SgVarRefExp *Reduce::GetGrid() const {
  if (IsKernel()) return NULL;
  SgExprListExp *args = reduce_call()->get_args();
  SgExpression *ge = *(args->get_expressions().begin() + 2);
  return isSgVarRefExp(ge);
}

} // namespace translator
} // namespace physis

