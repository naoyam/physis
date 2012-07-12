// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/optimization_common.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {

static void FixGridOffsetAttributeFuncCall(SgFunctionCallExp *offset_exp,
                                           GridOffsetAttribute *goa) {
  SgExpressionVector &indices = goa->indices();
  SgExpressionPtrList &args = offset_exp->get_args()->get_expressions();
  indices.clear();
  FOREACH (it, ++args.begin(), args.end()) {
    indices.push_back(*it);
  }
  SgVarRefExp *grid_ref =
      isSgVarRefExp(offset_exp->get_args()->get_expressions()[0]);
  PSAssert(grid_ref);
  goa->gvexpr() = grid_ref;
}

static void FixGridOffsetAttributeInlined(SgBinaryOp *offset_exp,
                                          GridOffsetAttribute *goa) {
  // This turnes out to be rather difficult. It would be rather
  // simpler to change the code generation method to use the function
  // call format.
  LOG_ERROR() << "Unsupported: "
              << offset_exp->unparseToString()
              << "\n";
  PSAbort(1);
  return;
}

void FixGridOffsetAttribute(SgExpression *offset_exp) {
  GridOffsetAttribute *goa =
      rose_util::GetASTAttribute<GridOffsetAttribute>(offset_exp);
  if (!goa) {
    LOG_ERROR() << "No GridOffsetAttribute found: "
                << offset_exp->unparseToString()
                << "\n";
    PSAssert(0);
  }

  offset_exp = rose_util::removeCasts(offset_exp);
  if (isSgFunctionCallExp(offset_exp)) {
    FixGridOffsetAttributeFuncCall(isSgFunctionCallExp(offset_exp), goa);
  } else if (isSgBinaryOp(offset_exp)) {
    FixGridOffsetAttributeInlined(isSgBinaryOp(offset_exp), goa);
  } else {
    LOG_ERROR() << "Unsupported offset expression: "
                << offset_exp->unparseToString() << "\n";
    PSAbort(1);
  }
}

void FixGridGetAttribute(SgExpression *get_exp) {
  GridGetAttribute *gga =
      rose_util::GetASTAttribute<GridGetAttribute>(get_exp);
  
  // Extract the correct offset expression
  SgExpression *new_offset = NULL;
  SgVarRefExp *new_grid = NULL;
  get_exp = rose_util::removeCasts(get_exp);
  if (isSgBinaryOp(get_exp)) {
    new_offset = isSgBinaryOp(get_exp)->get_rhs_operand();
    new_grid = isSgVarRefExp(isSgBinaryOp(
        rose_util::removeCasts(isSgBinaryOp(get_exp)->get_lhs_operand()))
                             ->get_lhs_operand());
  } else {
    LOG_ERROR() << "Unsupported grid get: "
                << get_exp->unparseToString() << "\n";
    PSAbort(1);
  }

  gga->offset() = new_offset;
  gga->gv() = new_grid->get_symbol()->get_declaration();
  
  // NOTE: new_offset does not have offset attribute if it is, for
  // example, a reference to a variable.
  if (rose_util::GetASTAttribute<GridOffsetAttribute>(new_offset)) {
    FixGridOffsetAttribute(new_offset);
  }
  //LOG_DEBUG() << "Fixed GridGetAttribute\n";
  return;
}

void FixGridAttributes(
    SgNode *node) {
  Rose_STL_Container<SgNode *> exps =
      NodeQuery::querySubTree(node, V_SgExpression);
  FOREACH(it, exps.begin(), exps.end()) {
    SgExpression *exp = isSgExpression(*it);
    PSAssert(exp);
    
    // Traverse only GridGet
    GridGetAttribute *gga =
        rose_util::GetASTAttribute<GridGetAttribute>(exp);
    if (!gga) continue;
    FixGridGetAttribute(exp);
  }
 
  return;
}

SgForStatement *FindInnermostLoop(SgNode *proj) {
  std::vector<SgNode*> run_kernel_loops =
      rose_util::QuerySubTreeAttribute<RunKernelLoopAttribute>(proj);
  SgForStatement *target_loop = NULL;
  FOREACH (run_kernel_loops_it, run_kernel_loops.rbegin(),
           run_kernel_loops.rend()) {
    target_loop = isSgForStatement(*run_kernel_loops_it);    
    RunKernelLoopAttribute *loop_attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(target_loop);
    if (!loop_attr) continue;
    if (!loop_attr->IsMain()) continue;
    break;
  }
  if (!target_loop) {
    LOG_DEBUG() << "No target loop for offset spatial CSE found\n";
  }

  return target_loop;
}

} // namespace optimizer
} // namespace translator
} // namespace physis
