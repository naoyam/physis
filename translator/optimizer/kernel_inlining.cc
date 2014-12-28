// Licensed under the BSD license. See LICENSE.txt for more details.

#include <algorithm>
#include <boost/foreach.hpp>

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/ast_processing.h"
#include "translator/builder_interface.h"
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

static bool RemoveRedundantBlock(SgScopeStatement *&s) {
  if (!isSgBasicBlock(s)) return false;
  SgBasicBlock *child = isSgBasicBlock(si::getFirstStatement(s, true));
  if (!child) return false;
  // Test if child is the only statement; 
  if (child != si::getLastStatement(s)) return false;
  // Replace s with child
  SgBasicBlock *child_copy = isSgBasicBlock(si::copyStatement(child));
  si::replaceStatement(s, child_copy);
  s = child_copy;
  return true;
}

static void MoveUpKernelParam(SgFunctionDeclaration *func,
                              SgScopeStatement *inlined_body,
                              int num_args) {
  int i;
  SgStatement *stmt;
  int num_dim = rose_util::GetASTAttribute<RunKernelAttribute>(func)
      ->stencil_map()->getNumDim();
  for (i = 0, stmt = si::getFirstStatement(inlined_body, true);
       i < num_args; ++i) {
    // Don't move index variables
    if (i < num_dim) {
      stmt = si::getNextStatement(stmt);
      continue;
    }      
    LOG_DEBUG() << "kernel param: " << stmt->unparseToString() << "\n";
    SgVariableDeclaration *vdecl = isSgVariableDeclaration(stmt);
    PSAssert(vdecl);
    PSAssert(vdecl->get_variables().size() == 1);    
    SgInitializedName *in = vdecl->get_variables()[0];
    SgAssignInitializer *init = isSgAssignInitializer(in->get_initializer());
    PSAssert(init);
    SgExpression *rhs = init->get_operand();
    // make sure no loop var is used
    vector<SgVarRefExp*> vars = si::querySubTree<SgVarRefExp>(rhs);
    bool is_loop_var = false;
    BOOST_FOREACH (SgVarRefExp *v, vars) {
      if (rose_util::GetASTAttribute<RunKernelIndexVarAttribute>
          (rose_util::GetDecl(v))) {
        is_loop_var = true;
        break;
      }
    }
    if (is_loop_var) {
      stmt = si::getNextStatement(stmt);
      continue;
    }
    // now it's safe to move up
    SgNode *insert_target = inlined_body;
    while (insert_target->get_parent() !=
           func->get_definition()->get_body()) {
      insert_target = insert_target->get_parent();
    }
    PSAssert(isSgStatement(insert_target));
    SgStatement *next_stmt = si::getNextStatement(stmt);
    si::removeStatement(stmt);
    LOG_DEBUG() << stmt->unparseToString() << "\n";
    si::insertStatementBefore(isSgStatement(insert_target),
                              stmt);
    
    stmt = next_stmt;
  }
}

SgForStatement *IsInLoop(SgFunctionCallExp *c) {
  SgNode *stmt = si::getEnclosingStatement(c)->get_parent();
  while (true) {
    if (isSgForStatement(stmt)) {
      PSAssert(rose_util::GetASTAttribute<RunKernelLoopAttribute>(stmt));
      return isSgForStatement(stmt);
    }
    if (isSgBasicBlock(stmt)) {
      stmt = stmt->get_parent();
    } else {
      break;
    }
  }
  return false;
}

static void AttachStencilIndexVarAttribute(SgFunctionDeclaration *run_kernel) {

  vector<SgNode *> indices =
      rose_util::QuerySubTreeAttribute<RunKernelIndexVarAttribute>(run_kernel);
  BOOST_FOREACH(SgNode *n, indices) {
    SgVariableDeclaration *index = isSgVariableDeclaration(n);
    RunKernelIndexVarAttribute *attr =
        rose_util::GetASTAttribute<RunKernelIndexVarAttribute>(index);
    StencilIndexVarAttribute *sva = new StencilIndexVarAttribute(attr->dim());
    BOOST_FOREACH(SgVarRefExp *vr, si::querySubTree<SgVarRefExp>(run_kernel)) {
      SgInitializedName *vrn = si::convertRefToInitializedName(vr);
      if (si::getFirstInitializedName(index) == vrn) {
        rose_util::AddASTAttribute<StencilIndexVarAttribute>(vr, sva);
      }
    }
  }
}

void kernel_inlining(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::BuilderInterface *builder) {
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
    
    vector<SgFunctionCallExp*> calls =
        si::querySubTree<SgFunctionCallExp>(func);
    FOREACH (calls_it, calls.begin(), calls.end()) {
      SgFunctionCallExp *call_exp = *calls_it;
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
      int num_args = call_exp->get_args()->get_expressions().size();
      SgForStatement *parent_loop = IsInLoop(call_exp);
      SgStatement *call_stmt = si::getEnclosingStatement(call_exp);
      SgStatement *prev_stmt = si::getPreviousStatement(call_stmt);

      // si::getPreviousStatement returns the parent loop if call stmt
      // is the only statement, which is not what is needed here.
      if (parent_loop == prev_stmt) {
        prev_stmt = NULL;
      }

      if (prev_stmt) {
        LOG_DEBUG() << "PREV: " << prev_stmt->unparseToString() << "\n";
      } else {
        LOG_DEBUG() << "NO PREV\n";
      }
      
      if (!doInline(call_exp)) {
        LOG_ERROR() << "Kernel inlining failed.\n";
        LOG_ERROR() << "Failed call: "
                    << call_exp->unparseToString() << "\n";
        PSAbort(1);
      }

      vector<SgNode*> body_vec =
          rose_util::QuerySubTreeAttribute<KernelBody>(func);
      PSAssert(body_vec.size() == 1);
      SgScopeStatement *inlined_body = isSgScopeStatement(body_vec.front());
      PSAssert(inlined_body);

      AttachStencilIndexVarAttribute(func);
      
      //while (RemoveRedundantBlock(scope)) {};

      // Don't move if no loop exists between the function body and
      // this call site
      if (parent_loop) {
        MoveUpKernelParam(func, inlined_body, num_args);
      }

      LOG_DEBUG() << "Removed " <<
          rose_util::RemoveRedundantVariableCopy(func->get_definition()->get_body())
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

