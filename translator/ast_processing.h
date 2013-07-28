// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_TRANSLATOR_AST_PROCESSING_H_
#define PHYSIS_TRANSLATOR_AST_PROCESSING_H_

#include "translator/translator_common.h"
#include "physis/internal_common.h"

namespace physis {
namespace translator {
namespace rose_util {

int RemoveRedundantVariableCopy(SgNode *scope);
int RemoveUnusedFunction(SgNode *scope);

}  // namespace rose_util
}  // namespace translator
}  // namespace physis

#endif /* PHYSIS_TRANSLATOR_AST_PROCESSING_H_ */


