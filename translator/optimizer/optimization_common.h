// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/translator_common.h"
#include "translator/translation_context.h"

namespace physis {
namespace translator {
namespace optimizer {

//! Fix references to index variables.
extern void FixGridOffsetAttribute(SgExpression *offset_exp);

//! Fix references to the grid and offset expression.
extern void FixGridGetAttribute(SgExpression *get_exp);


//! Fix references to the offset expression.
extern void FixGridAttributes(
    SgNode *node);

//! Find innermost kernel loops
extern vector<SgForStatement*> FindInnermostLoops(SgNode *proj);

//! Find expressions that are assigned to variable v
extern void GetVariableSrc(SgInitializedName *v,
                           vector<SgExpression*> &src_exprs);

//! Simple dead code elimination
extern bool EliminateDeadCode(SgStatement *stmt);

//! Returns a single source expression for a variable if statically determined
SgExpression *GetDeterministicDefinition(SgInitializedName *var);

} // namespace optimizer
} // namespace translator
} // namespace physis
