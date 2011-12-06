// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/optimization_passes.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/translation_util.h"

#include <algorithm>

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

static void RemoveUnusedLabel(SgProject *proj) {
  // Find used labels
  Rose_STL_Container<SgNode *> gotos =
      NodeQuery::querySubTree(proj, V_SgGotoStatement);
  std::vector<SgLabelStatement*> labels;
  FOREACH (it, gotos.begin(), gotos.end()) {
    SgGotoStatement *goto_stmt = isSgGotoStatement(*it);
    SgLabelStatement *label = goto_stmt->get_label();
    labels.push_back(label);
  }
  
  // Iterate over all labels and remove ones not included in the used
  // set.
  Rose_STL_Container<SgNode *> label_stmts =
      NodeQuery::querySubTree(proj, V_SgLabelStatement);
  FOREACH (it, label_stmts.begin(), label_stmts.end()) {
    SgLabelStatement *label = isSgLabelStatement(*it);
    if (isContained(labels, label)) continue;
    si::removeStatement(label);
  }
}

void kernel_inlining(
    SgProject *proj,
    physis::translator::TranslationContext *tx) {
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
      // Kernel call found
      LOG_DEBUG() << "Inline a call to kernel found: "
                  << call_exp->unparseToString() << "\n";
      //SgNode *t = call_exp->get_parent();
      // while (t) {
      //   LOG_DEBUG() << "parent: ";
      //   LOG_DEBUG() << t->unparseToString() << "\n";
      //   t = t->get_parent();
      // }
      if (!doInline(call_exp)) {
        LOG_ERROR() << "Kernel inlining failed.\n";
        LOG_ERROR() << "Failed call: "
                    << call_exp->unparseToString() << "\n";
        PSAbort(1);
      }
    }
  }

  // Remove unused lables created by doInline.
  RemoveUnusedLabel(proj);

  // TODO: Does not work probably because the AST node linkage is
  //partially broken. 
  //cleanupInlinedCode(proj);
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

