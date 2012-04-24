// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/rose_ast_attribute.h"

namespace physis {
namespace translator {

const std::string GridCallAttribute::name = "GridCall";

GridCallAttribute::GridCallAttribute(SgInitializedName *grid_var,
                                     KIND k):
    grid_var_(grid_var), kind_(k) {
}

GridCallAttribute::~GridCallAttribute() {}

AstAttribute *GridCallAttribute::copy() {
  return new GridCallAttribute(grid_var_, kind_);
}

bool GridCallAttribute::IsGet() {
  return kind_ == GET;
}

bool GridCallAttribute::IsGetPeriodic() {
  return kind_ == GET_PERIODIC;
}

bool GridCallAttribute::IsEmit() {
  return kind_ == EMIT;
}

} // namespace translator
} // namespace physis
