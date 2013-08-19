// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_TRANSLATOR_AST_TRAVERSAL_H_
#define PHYSIS_TRANSLATOR_AST_TRAVERSAL_H_

#include "translator/translator_common.h"
#include "physis/internal_common.h"

namespace physis {
namespace translator {
namespace rose_util {

template <class ASTNodeType>
ASTNodeType *FindClosestAncestor(SgNode *node) {
  SgNode *p = node->get_parent();
  while (p) {
    if (p->variantT() == (VariantT)ASTNodeType::static_variant) {
      return dynamic_cast<ASTNodeType*>(p);
    }
    p = p->get_parent();
  }
  return NULL;
}

}  // namespace rose_util
}  // namespace translator
}  // namespace physis

#endif /* PHYSIS_TRANSLATOR_AST_TRAVERSAL_H_ */


