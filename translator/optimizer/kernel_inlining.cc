// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
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

// SageInterface has an function exactly for this purpose.
#if 0
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
#endif

/*! Replace a variable reference in an offset expression.

  \param vref Variable reference
  \param loop_body Target loop 
 */
static SgVarRefExp *replace_var_ref(SgVarRefExp *vref, SgBasicBlock *loop_body) {
  LOG_DEBUG() << "Replacing " << vref->unparseToString() << "\n";
  SgVariableDeclaration *vdecl = isSgVariableDeclaration(
      vref->get_symbol()->get_declaration()->get_declaration());
  vector<SgVariableDeclaration*> decls =
      si::querySubTree<SgVariableDeclaration>(loop_body, V_SgVariableDeclaration);
  if (!isContained(decls, vdecl)) return NULL;
  if (!vdecl) return NULL;
  SgAssignInitializer *init =
      isSgAssignInitializer(
          vdecl->get_definition()->get_vardefn()->get_initializer());
  if (!init) return NULL;
  SgExpression *rhs = init->get_operand();
  LOG_DEBUG() << "RHS: " << rhs->unparseToString() << "\n";
  PSAssert(isSgVarRefExp(rhs));
  std::vector<SgVarRefExp*> vref_exprs =
      si::querySubTree<SgVarRefExp>(loop_body, V_SgVarRefExp);
  FOREACH (it, vref_exprs.begin(), vref_exprs.end()) {
    if (vdecl == isSgVariableDeclaration(
            (*it)->get_symbol()->get_declaration()->get_declaration())) {
      si::replaceExpression(*it, si::copyExpression(rhs));
    }
  }
  si::removeStatement(vdecl);
  return isSgVarRefExp(rhs);
}

/*! Fix variable references in offset expression.

  Offset expressions may use a variable reference to grids and loop
  indices that are defined inside the target loop. They need to be
  replaced with their original value when moved out of the loop.

  \param offset_expr Offset expression
  \param loop_body Target loop body
*/
static void replace_arg_defined_in_loop(SgExpression *offset_expr,
                                        SgBasicBlock *loop_body,
                                        set<SgName> &original_names) {
  LOG_DEBUG() << "Replacing undef var in "
              << offset_expr->unparseToString() << "\n";
  PSAssert(offset_expr != NULL && loop_body != NULL);
  SgFunctionCallExp *offset_call = isSgFunctionCallExp(offset_expr);
  PSAssert(offset_call);
  SgExpressionPtrList &args = offset_call->get_args()->get_expressions();

  //if (isSgVarRefExp(args[0])) {
  //replace_var_ref(isSgVarRefExp(args[0]), loop_body);
  //}
  FOREACH (argit, args.begin()+1, args.end()) {
    SgExpression *arg = *argit;
    Rose_STL_Container<SgNode*> vref_list =
        NodeQuery::querySubTree(arg, V_SgVarRefExp);
    if (vref_list.size() == 0) continue;
    PSAssert(vref_list.size() == 1);
    if (isContained(
            original_names,
            rose_util::GetName(isSgVarRefExp(vref_list[0])))) {
      // Already replaced. This is a reference to a variable that is
      // defined in the loop instead of the kernel function.
      continue;
    }
    SgVarRefExp *vr = replace_var_ref(isSgVarRefExp(vref_list[0]), loop_body);
    if (vr) {
      LOG_DEBUG() << "Replaced with " << vr->unparseToString() << "\n";
      original_names.insert(rose_util::GetName(vr));
    }
  }
  LOG_DEBUG() << "Result: " << offset_expr->unparseToString() << "\n";
  return;
}

static SgForStatement *FindInnermostMapLoop(SgFunctionCallExp *kernel_call) {
  for (SgNode *node = kernel_call->get_parent(); node != NULL;
       node = node->get_parent()) {
    RunKernelLoopAttribute *loop_attr
        = rose_util::GetASTAttribute<RunKernelLoopAttribute>(node);
    if (loop_attr) return isSgForStatement(node);
  }
  return NULL;
}

static void cleanup(SgFunctionCallExp *call_exp,
                    SgFunctionRefExp *callee_ref,
                    SgForStatement *loop) {
  std::vector<SgNode*> offset_exprs =
      rose_util::QuerySubTreeAttribute<GridOffsetAttribute>(
          loop);
  set<SgName> original_names;  
  FOREACH (it, offset_exprs.begin(), offset_exprs.end()) {
    SgExpression *offset_expr = isSgExpression(*it);
    replace_arg_defined_in_loop(offset_expr,
                                isSgBasicBlock(loop->get_loop_body()),
                                original_names);
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
      SgForStatement *map_loop = FindInnermostMapLoop(call_exp);
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
      // Fix the grid attributes
      FixGridAttributes(proj);
      cleanup(call_exp, callee_ref, map_loop);
    }
  }

  // Remove unused lables created by doInline.
  //RemoveUnusedLabel(proj);
  si::removeUnusedLabels(proj);

  // Remove original kernels
  // NOTE: Does not work (segmentation fault) even if AST consistency is
  // kept. 
  // cleanupInlinedCode(proj);
  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

