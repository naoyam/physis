// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include <algorithm>
#include <boost/foreach.hpp>

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/ast_processing.h"
#include "translator/runtime_builder.h"
#include "translator/translation_util.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

static void RemoveRedundantAddressOfOp(SgNode *node) {
  vector<SgAddressOfOp*> ops = si::querySubTree<SgAddressOfOp>(node);
  BOOST_FOREACH (SgAddressOfOp *op, ops) {
    SgArrowExp *p = isSgArrowExp(op->get_parent());
    if (!p) continue;
    SgExpression *lhs = op->get_operand();
    SgExpression *rhs = p->get_rhs_operand();
    SgExpression *replacement =
        sb::buildDotExp(si::copyExpression(lhs),
                        si::copyExpression(rhs));
    si::replaceExpression(p, replacement);
  }
}

void kernel_inlining(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

  //si::fixVariableReferences(proj);
  
  Rose_STL_Container<SgNode *> funcs =
      NodeQuery::querySubTree(proj, V_SgFunctionDeclaration);
  FOREACH(it, funcs.begin(), funcs.end()) {
    SgFunctionDeclaration *func = isSgFunctionDeclaration(*it);
    RunKernelAttribute *run_kernel_attr
        = rose_util::GetASTAttribute<RunKernelAttribute>(func);
    // Filter non RunKernel function
    if (!run_kernel_attr) continue;
    
    Rose_STL_Container<SgNode *> calls =
        NodeQuery::querySubTree(proj, V_SgFunctionCallExp);
    FOREACH (calls_it, calls.begin(), calls.end()) {
      SgFunctionCallExp *call_exp =
          isSgFunctionCallExp(*calls_it);
      SgFunctionRefExp *callee_ref
          = isSgFunctionRefExp(call_exp->get_function());
      if (!callee_ref) continue;
      SgFunctionDeclaration *callee_decl
          = rose_util::getFuncDeclFromFuncRef(callee_ref);
      if (!callee_decl) continue;
      if (!tx->isKernel(callee_decl)) continue;
      LOG_DEBUG() << "Inline a call to kernel found: "
                  << call_exp->unparseToString() << "\n";
      // Kernel call found
      //SgNode *t = call_exp->get_parent();
      // while (t) {
      //   LOG_DEBUG() << "parent: ";
      //   LOG_DEBUG() << t->unparseToString() << "\n";
      //   t = t->get_parent();
      // }
      SgScopeStatement *scope = si::getScope(call_exp);
      if (!doInline(call_exp)) {
        LOG_ERROR() << "Kernel inlining failed.\n";
        LOG_ERROR() << "Failed call: "
                    << call_exp->unparseToString() << "\n";
        PSAbort(1);
      }
      // Fix the grid attributes
      FixGridAttributes(proj);

      LOG_DEBUG() << "Removed " <<
          rose_util::RemoveRedundantVariableCopy(scope)
                  << " variables\n";

    }
  }
  
  RemoveRedundantAddressOfOp(proj);
  
  si::removeUnusedLabels(proj);
  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

