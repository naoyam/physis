// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_ROSE_AST_ATTRIBUTE_H_
#define PHYSIS_TRANSLATOR_ROSE_AST_ATTRIBUTE_H_

#include "translator/translator_common.h"

namespace physis {
namespace translator {

class GridCallAttribute: public AstAttribute {
 public:
  enum KIND {GET, GET_PERIODIC, EMIT};  
  GridCallAttribute(SgInitializedName *grid_var,
                    KIND k);
  virtual ~GridCallAttribute();
  static const std::string name;
  AstAttribute *copy();
  SgInitializedName *grid_var() { return grid_var_; };
  //! Returns true if the node is get.  
  bool IsGet();
  //! Returns true if the node is get_periodic.
  bool IsGetPeriodic();  
  bool IsEmit();  
 protected:
  SgInitializedName *grid_var_;
  KIND kind_;
};

void CopyAllAttributes(SgNode *dst, SgNode *src);

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_ROSE_AST_ATTRIBUTE_H_ */
