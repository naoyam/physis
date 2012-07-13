// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/translation_util.h"

#include <algorithm>
#include <stack>

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

static bool IsEnclosed(SgNode *node, SgScopeStatement *scope) {
  while (node) {
    if (scope == node) return true;
    node = node->get_parent();
  }
  return false;
}

static SgVariableDeclaration *GetVariableDeclaration(SgVarRefExp *e) {
  return isSgVariableDeclaration(
      e->get_symbol()->get_declaration()->get_declaration());
}

static SgInitializedName *GetVariable(SgVarRefExp *e) {
  return e->get_symbol()->get_declaration();
}

//typedef std::list<SgVariableDeclaration*> VarStack;
typedef std::list<SgInitializedName*> VarStack;

static bool IsLoopInvariant(SgExpression *e, SgForStatement *loop,
                            VarStack &stack);

static bool IsAddressTaken(SgInitializedName *vs) {
  vector<SgAddressOfOp*> addrof_exprs =
      si::querySubTree<SgAddressOfOp>(si::getScope(vs), V_SgAddressOfOp);
  // No & op allowed    
  FOREACH (it, addrof_exprs.begin(), addrof_exprs.end()) {
    SgExpression *addr_exp  = (*it)->get_operand();
    SgInitializedName *decl
        = GetVariable(isSgVarRefExp(addr_exp));
    if (isSgVarRefExp(addr_exp) && (vs == decl)) {
      LOG_DEBUG() << "Assumes non-invariant when address is taken\n";
      return true;
    }
  }
  return false;
}

static bool IsLoopInvariant(SgInitializedName *v,
                            SgForStatement *loop,
                            VarStack &stack) {
  
  LOG_DEBUG() << "Declaration, "
              << v->unparseToString() << ", is invariant?\n";
  
  if (IsAddressTaken(v)) {
      LOG_DEBUG() << "Declaration, "
                  << v->unparseToString() << ", is invariant?: 0\n";
    return false;
  }
  
  SgNode *v_scope = si::getScope(v);
  // Assignments are allowed only outside the loop
  vector<SgNode*> assign_exprs =
      NodeQuery::querySubTree(v_scope, V_SgAssignOp);
  // Assignments are allowed only outside the loop
  vector<SgNode*> tmp_exprs =
      NodeQuery::querySubTree(v_scope,  V_SgCompoundAssignOp);
  assign_exprs.insert(assign_exprs.end(), tmp_exprs.begin(),
                      tmp_exprs.end());
  tmp_exprs = NodeQuery::querySubTree(v_scope, V_SgPlusPlusOp);
  assign_exprs.insert(assign_exprs.end(), tmp_exprs.begin(),
                      tmp_exprs.end());
  tmp_exprs = NodeQuery::querySubTree(v_scope, V_SgMinusMinusOp);
  assign_exprs.insert(assign_exprs.end(), tmp_exprs.begin(),
                      tmp_exprs.end());
  tmp_exprs = NodeQuery::querySubTree(v_scope, V_SgVariableDeclaration);
  assign_exprs.insert(assign_exprs.end(), tmp_exprs.begin(),
                      tmp_exprs.end());
  tmp_exprs = NodeQuery::querySubTree(v_scope, V_SgParameterStatement);
  assign_exprs.insert(assign_exprs.end(), tmp_exprs.begin(),
                      tmp_exprs.end());

  bool assigned = false;
  bool invariant = true;

  if (isSgFunctionParameterList(v->get_declaration())) {
    assigned = true;
  }
  FOREACH (it, assign_exprs.begin(), assign_exprs.end()) {
    SgNode *node = *it;
    if (!IsEnclosed(node, loop)) continue;
    SgExpression *operand_expr = NULL;
    if (isSgExpression(node)) {
      SgExpression *modified_expr = NULL;
      if (isSgBinaryOp(node)) {
        modified_expr = isSgBinaryOp(node)->get_lhs_operand();
        operand_expr = isSgBinaryOp(node)->get_rhs_operand();
      } else if (isSgUnaryOp(node)) {
        modified_expr = isSgUnaryOp(node)->get_operand();
        operand_expr = isSgUnaryOp(node)->get_operand();
      }
      PSAssert(modified_expr);
      if (!(isSgVarRefExp(modified_expr) &&
            (v == GetVariable(isSgVarRefExp(modified_expr))))) {
        continue;
      }
    } else if (isSgVariableDeclaration(node)) {
      SgVariableDeclaration *decl = isSgVariableDeclaration(node);
      if (v->get_declaration() != decl) continue;
      SgVariableDefinition *vdef = decl->get_definition();
      if (!vdef) continue;
      SgAssignInitializer *init =
          isSgAssignInitializer(vdef->get_vardefn()->get_initializer());
      if (!init) continue;
      operand_expr = init->get_operand();
    }
    PSAssert(operand_expr);
    if (!IsEnclosed(node, loop)) {
      assigned = true;
      continue;
    }
    if (assigned) {
      LOG_DEBUG() << "Assigned multiple times within the loop\n";
      invariant = false;
      break;
    }
    assigned = true;
    stack.push_back(v);
    if (!IsLoopInvariant(operand_expr, loop, stack)) {
      LOG_DEBUG() << operand_expr->unparseToString() <<
          " is non invariant\n";
      invariant = false;
      break;
    }
    stack.pop_back();
  }
  LOG_DEBUG() << "Declaration, "
              << v->unparseToString() << ", is invariant?: "
              << invariant << "\n";
  return invariant;
}

static bool IsLoopInvariant(SgFunctionCallExp *e, SgForStatement *loop,
                            VarStack &stack) {
  std::string func_name = rose_util::getFuncName(e);
  if (func_name == "PSGridDim") {
    SgExpressionPtrList &args = e->get_args()->get_expressions();
    FOREACH (it, ++(args.begin()), args.end()) {
      SgExpression *arg_expr = *it;
      if (!IsLoopInvariant(arg_expr, loop, stack)) return false;
    }
    LOG_DEBUG() << "Call to PSGridDim is invariant\n";
    return true;
  }
  return false;
}

static bool IsLoopInvariant(SgExpression *e, SgForStatement *loop,
                            VarStack &stack) {
  LOG_DEBUG() << "Is expression invariant?: "
              << e->unparseToString() << "\n";
  bool invariant = false;
  if (isSgValueExp(e)) {
    invariant = true;
  } else if (isSgUnaryOp(e)) {
    invariant = IsLoopInvariant(isSgUnaryOp(e)->get_operand(), loop,
                                stack);
  } else if (isSgBinaryOp(e)) {
    invariant = IsLoopInvariant(isSgBinaryOp(e)->get_lhs_operand(),
                                loop, stack) &&
        IsLoopInvariant(isSgBinaryOp(e)->get_rhs_operand(),
                        loop, stack);
  } else if (isSgVarRefExp(e)) {
    SgInitializedName *vs = GetVariable(isSgVarRefExp(e));
    // If decl is not found, assumes non-invariant
    if (!vs) {
      invariant = false;
      LOG_DEBUG() << "Declaration of " << e->unparseToString()
                  << " not found\n";
    } else if (isContained(stack, vs)) {
      LOG_DEBUG() << "Recursive definition: "
                  << vs->unparseToString() << "\n";
      // Recursive definition if this variable already appears
      invariant = false;
    } else {
      invariant = IsLoopInvariant(vs, loop, stack);
    }
  } else if (isSgFunctionCallExp(e)) {
    invariant = IsLoopInvariant(isSgFunctionCallExp(e), loop,
                                stack);
  }
  
  LOG_DEBUG() << "Expr, " << e->unparseToString() << ", is invariant?: "
              << invariant << "\n";
  return invariant;
}


#define IS_EQUIVALENT_VALUE(e1, e2, type)                               \
  (is ## type (e1) && is##type(e2) &&                                   \
   (is ## type (e1)->get_value() == is ## type(e2)->get_value()))


static bool IsEquivalent(SgExpression *e1, SgExpression *e2) {
  if (isSgVarRefExp(e1) && isSgVarRefExp(e2)) {
    return isSgVarRefExp(e1)->get_symbol() ==
        isSgVarRefExp(e2)->get_symbol();
  }

  return IS_EQUIVALENT_VALUE(e1, e2, SgIntVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgLongIntVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgBoolValExp) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgCharVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgDoubleVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgFloatVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgLongLongIntVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgShortVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgUnsignedIntVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgUnsignedLongVal) ||
      IS_EQUIVALENT_VALUE(e1, e2, SgUnsignedLongLongIntVal);
}

static bool ProcessAssignment(SgIfStmt *if_stmt,
                              SgStatement *true_body,
                              SgStatement *false_body,
                              SgExprStatement *true_assign,
                              SgExprStatement *false_assign) {
  LOG_DEBUG() << "Optimizing " << if_stmt->unparseToString() << "\n";
  SgBinaryOp *t_assign_expr = isSgBinaryOp(true_assign->get_expression());
  SgBinaryOp *f_assign_expr = isSgBinaryOp(false_assign->get_expression());
  // Process only variable references for simplicity of analysis  
  SgVarRefExp *t_lhs = isSgVarRefExp(t_assign_expr->get_lhs_operand());
  SgExpression *t_rhs = t_assign_expr->get_rhs_operand();
  SgVarRefExp *f_lhs = isSgVarRefExp(f_assign_expr->get_lhs_operand());
  SgExpression *f_rhs = f_assign_expr->get_rhs_operand();

  // Ensure both expressions assign to the same variable
  if (t_lhs->get_symbol() != f_lhs->get_symbol()) return false;

  std::stack<SgExpression*> expr_stack;
  SgExpression *t_rhs_reduced = t_rhs;
  SgExpression *f_rhs_reduced = f_rhs;
  while (true) {
    LOG_DEBUG()
        << "Reducing expressions: "
        << (t_rhs_reduced ? t_rhs_reduced->unparseToString() : "NULL")
        << ", "
        << (f_rhs_reduced ? f_rhs_reduced->unparseToString() : "NULL")
        << "\n";
    if (t_rhs_reduced == NULL || f_rhs_reduced == NULL) break;
    SgExpression *te[2] = {t_rhs_reduced, NULL};
    SgExpression *fe[2] = {f_rhs_reduced, NULL};
    if (isSgBinaryOp(t_rhs_reduced)) {
      te[0] = isSgBinaryOp(t_rhs_reduced)->get_lhs_operand();
      te[1] = isSgBinaryOp(t_rhs_reduced)->get_rhs_operand();
      // Only plus op is assumed for simplicity
      if (!isSgAddOp(t_rhs_reduced)) return false;
    }
    if (isSgBinaryOp(f_rhs_reduced)) {
      fe[0] = isSgBinaryOp(f_rhs_reduced)->get_lhs_operand();
      fe[1] = isSgBinaryOp(f_rhs_reduced)->get_rhs_operand();
      // Only plus op is assumed for simplicity
      if (!isSgAddOp(f_rhs_reduced)) return false;
    }
    bool modified = false;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (te[i] != NULL && fe[j] != NULL) {
          if (IsEquivalent(te[i], fe[j])) {
            expr_stack.push(si::copyExpression(te[i]));
            t_rhs_reduced = te[i ^ 1];
            f_rhs_reduced = fe[i ^ 1];
            te[i] = NULL;
            fe[j] = NULL;
            modified = true;
          }
        }
      }
    }
    if (!modified) break;
  }

  // Returns if no common expression is extracted
  if (expr_stack.empty()) return false;
  
  SgExpression *common_expr = NULL;
  while (!expr_stack.empty()) {
    SgExpression *e = expr_stack.top();
    expr_stack.pop();
    if (common_expr)
      common_expr = sb::buildAddOp(common_expr, e);
    else
      common_expr = e;
  }

  // build a new assignment to the same variable with common_expr and
  // insert it before the if statement
  SgExprStatement *common_expr_stmt =
      sb::buildExprStatement(
          sb::buildAssignOp(si::copyExpression(t_lhs), common_expr));
  si::insertStatementBefore(if_stmt, common_expr_stmt);
  LOG_DEBUG() << "Inserting common expression statement: "
              << common_expr_stmt->unparseToString() << "\n";

  if (t_rhs_reduced) {
    si::replaceExpression(
        t_rhs,
        sb::buildAddOp(si::copyExpression(t_lhs),
                       si::copyExpression(t_rhs_reduced)));
  } else {
    LOG_DEBUG() << "Removing " << true_assign->unparseToString() << "\n";
    // Simply removing doesn't work
    //si::removeStatement(true_assign);
    si::replaceStatement(true_assign, sb::buildNullStatement());
  }
  if (f_rhs_reduced) {
    si::replaceExpression(
        f_rhs,
        sb::buildAddOp(si::copyExpression(t_lhs),
                       si::copyExpression(f_rhs_reduced)));
  } else {
    LOG_DEBUG() << "Removing " << false_assign->unparseToString() << "\n";
    // Simply removing doesn't work
    //si::removeStatement(false_assign);
    si::replaceStatement(false_assign, sb::buildNullStatement());    
  }

  return true;
}

static void MoveInvariantOutOfBranch(SgIfStmt *if_stmt) {
  SgStatement *true_body = if_stmt->get_true_body();
  SgStatement *false_body = if_stmt->get_false_body();
  
  if (true_body == NULL || false_body == NULL) return;
  
  std::vector<SgExprStatement*> t_assignments =
      si::querySubTree<SgExprStatement>(true_body);
  FOREACH (t_it, t_assignments.begin(), t_assignments.end()) {
    if (!isSgAssignOp((*t_it)->get_expression())) continue;
    std::vector<SgExprStatement*> f_assignments =
        si::querySubTree<SgExprStatement>(false_body);
    FOREACH (f_it, f_assignments.begin(), f_assignments.end()) {
      if (!isSgAssignOp((*f_it)->get_expression())) continue;
      ProcessAssignment(if_stmt, true_body, false_body,
                        *t_it, *f_it);
    }
  }
}

static void ProcessLoopBody(SgForStatement *loop) {
  SgBasicBlock *body = isSgBasicBlock(loop->get_loop_body());
  std::vector<SgIfStmt*> if_stmts = si::querySubTree<SgIfStmt>(body);
  FOREACH (it, if_stmts.begin(), if_stmts.end()) {
    SgIfStmt *if_stmt = *it;
    SgStatement *cond = if_stmt->get_conditional();
    VarStack stack;
    if (isSgExprStatement(cond) &&
        IsLoopInvariant(
            isSgExprStatement(cond)->get_expression(),
            loop, stack)) {
      MoveInvariantOutOfBranch(if_stmt);
    }
  }
  
  return;
}

void loop_opt(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

  
  vector<SgForStatement*> target_loops = FindInnermostLoops(proj);
  FOREACH (it, target_loops.begin(), target_loops.end()) {
    SgForStatement *loop = *it;
    RunKernelLoopAttribute *attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
    if (!attr->IsMain()) continue;
    LOG_DEBUG() << "Optimizing "
                << loop->unparseToString() << "\n";
    ProcessLoopBody(loop);
  }
  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

