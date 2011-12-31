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

static void QueryGetInKernel(
    SgNode *top,
    Rose_STL_Container<SgPntrArrRefExp*> &gets) {
  Rose_STL_Container<SgNode *> exps =
      NodeQuery::querySubTree(top, V_SgPntrArrRefExp);
  FOREACH(it, exps.begin(), exps.end()) {
    SgPntrArrRefExp *exp = isSgPntrArrRefExp(*it);
    PSAssert(exp);
    
    // Traverse only GridGet
    GridGetAttribute *gga =
        rose_util::GetASTAttribute<GridGetAttribute>(exp);
    if (!gga) continue;

    // Optimization applied only to in-kernel get
    if (!gga->in_kernel()) continue;
    gets.push_back(exp);
  }
  return;
}

static SgInitializedName *GetGridFromGet(SgPntrArrRefExp *get) {
  GridGetAttribute *gga =
      rose_util::GetASTAttribute<GridGetAttribute>(get);
  PSAssert(gga);
  return gga->gv();
}

static bool IsInTrueBlock(SgIfStmt *if_stmt,
                          SgNode *node) {
  if (si::isAncestor(if_stmt->get_true_body(), node)) {
    return true;
  }
  if (si::isAncestor(if_stmt->get_false_body(), node)) {
    return false;
  }
  // node must be a child of either blocks, so this should not be
  // reached.
  PSAssert(0);
  return false;
}

static bool IsOffsetCenter(SgFunctionCallExp *get_offset) {
  SgExpressionPtrList &args = get_offset->get_args()->get_expressions();
  FOREACH (it, args.begin() + 1, args.end()) {
    if (isSgBinaryOp(rose_util::removeCasts(*it))) return false;
  }
  return true;
}

static SgExpression *BuildGetOffsetCenter(SgFunctionCallExp *get_offset) {
  PSAssert(get_offset);
  get_offset = isSgFunctionCallExp(si::copyExpression(get_offset));
  SgExpressionPtrList &args = get_offset->get_args()->get_expressions();
  FOREACH (it, args.begin() + 1, args.end()) {
    SgBinaryOp *binop = isSgBinaryOp(rose_util::removeCasts(*it));
    if (binop) {
      SgVarRefExp *vref = NULL;
      if (isSgVarRefExp(binop->get_rhs_operand())) {
        vref = isSgVarRefExp(binop->get_rhs_operand());
      } else {
        vref = isSgVarRefExp(binop->get_lhs_operand());
      }
      PSAssert(vref);
      si::replaceExpression(binop, si::copyExpression(vref),
                            false);
    }
  }
  return get_offset;
}

static void ProcessIfStmt(SgPntrArrRefExp *get_exp,
                          SgIfStmt *if_stmt) {
  /*
    Step 1: Simplify the if block by moving the get expression out of 
    the loop to a separate new loop.
    E.g.,
    if (t) {
      x = get(PSGetOffset(...)) + a
    }
    This block should be transformed to:
    float f = 0.0;
    if (t) {
      f = get(PSGetOffset(...));
    }
    if (t) {
      x = f + a
    }

    If multiple gets are used in both true and false parts, move all
    of them to the prepended loop.
    E.g.,
    if (t) {
      x = get(PSGetOffset(a1));
    } else {
      x = get(PSGetOffset(a3));
    }
    This block should be transformed to:
    float f1 = 0.0;
    if (t) {
      f1 = get(PSGetOffset(a1));
    } else {
      f3 = get(PSGetOffset(a3));
    }
    if (t) {
      x = f1;
    } else {
      x = f3;
    }

    Step 2: Move forward the get out of loop.
    PSIndexType idx = PSGetOffset(x, y, z);
    if (t) {
      idx = PSGetOffset(...);
    }  
    float f = get(idx);
    // No change hereafter
  */



  SgScopeStatement *outer_scope
      = si::getScope(if_stmt->get_parent());
  SgInitializedName *gv = GetGridFromGet(get_exp);

  // Find a get_exp that can be paired with this.
  Rose_STL_Container<SgPntrArrRefExp *> gets;
  QueryGetInKernel(if_stmt, gets);
  SgPntrArrRefExp *paired_get_exp = NULL;
  FOREACH (it, gets.begin(), gets.end()) {
    LOG_DEBUG() << "Conditional get: "
                << (*it)->unparseToString() << "\n";
    if (get_exp == *it) continue;
    SgNode *parent = rose_util::FindCommonParent(get_exp, *it);
    LOG_DEBUG() << "Common parent: "
                << parent->unparseToString() << "\n";
    if (parent != if_stmt) {
      // located in the same conditional case
      continue;
    }
    // They must point to the same grid
    SgInitializedName *gv_peer = GetGridFromGet(*it);
    if (gv != gv_peer) continue;
    paired_get_exp = *it;
  }
  
  if (paired_get_exp) {
    LOG_DEBUG() << "Peer of get found: "
                << paired_get_exp->unparseToString()
                << "\n";
  } else {
    LOG_DEBUG() << "No peer found\n";
  }

  // Make a bool var of the if conditional so that it can be reused
  // for the new if block with the same conditional.
  SgStatement *cond = if_stmt->get_conditional();
  // Note: what class can the conditional be? Examining several ASTs
  // indicates it is always SgExprStatement, so we assume it is always
  // the case.
  if (!isSgExprStatement(cond)) {
    LOG_ERROR() << "Unexpected condition type: "
                << cond->class_name()
                << ": " << cond->unparseToString()
                << "\n";
    PSAbort(1);
  }
  SgVariableDeclaration *cond_var
      = sb::buildVariableDeclaration(
          rose_util::generateUniqueName(outer_scope),
          sb::buildIntType(),
          sb::buildAssignInitializer(
              isSgExprStatement(cond)->get_expression()),
          outer_scope);
  si::insertStatementBefore(if_stmt, cond_var);
  // Replace the condition with the bool variable
  if_stmt->set_conditional(
      sb::buildExprStatement(sb::buildVarRefExp(cond_var)));

  // Declare the index variable
  SgVariableDeclaration *index_var = NULL;
  // If the access is to the center point, no guard by condition is
  // necessary and just assign the offset
  if (paired_get_exp == NULL &&
      IsOffsetCenter(isSgFunctionCallExp(get_exp->get_rhs_operand()))) {
    index_var = sb::buildVariableDeclaration(
        rose_util::generateUniqueName(outer_scope),
        physis::translator::BuildIndexType2(outer_scope),
        sb::buildAssignInitializer(
            si::copyExpression(get_exp->get_rhs_operand())),
        outer_scope);
    si::insertStatementBefore(if_stmt, index_var);    
  } else {
    index_var = sb::buildVariableDeclaration(
        rose_util::generateUniqueName(outer_scope),
        physis::translator::BuildIndexType2(outer_scope),
        sb::buildAssignInitializer(sb::buildIntVal(0)),
        outer_scope);
    si::insertStatementBefore(if_stmt, index_var);
    // Make a conditional block to assign the correct index to index_var 
    SgBasicBlock *get_exp_index_assign =
        sb::buildBasicBlock(
            sb::buildAssignStatement(
                sb::buildVarRefExp(index_var),
                si::copyExpression(get_exp->get_rhs_operand())));
    SgBasicBlock *paired_get_exp_index_assign = NULL;
    if (paired_get_exp) {
      paired_get_exp_index_assign =
          sb::buildBasicBlock(
              sb::buildAssignStatement(
                  sb::buildVarRefExp(index_var),
                  si::copyExpression(
                      paired_get_exp->get_rhs_operand())));
    } else {
      // GetOffset of the center point, i.e., GetOffset(g, x, y, z)
      paired_get_exp_index_assign =
          sb::buildBasicBlock(
              sb::buildAssignStatement(
                  sb::buildVarRefExp(index_var),
                  BuildGetOffsetCenter(
                      isSgFunctionCallExp(get_exp->get_rhs_operand()))));
    }

    SgBasicBlock *true_block = get_exp_index_assign;
    SgBasicBlock *false_block = paired_get_exp_index_assign;
    if (!IsInTrueBlock(if_stmt, get_exp)) {
      std::swap(true_block, false_block);
    }
    SgIfStmt *index_assign_block = sb::buildIfStmt(
        sb::buildVarRefExp(cond_var),
        true_block, false_block);
    si::insertStatementBefore(if_stmt, index_assign_block);
  }
    
  // Declare a new variable of the same type as grid.
  SgVariableDeclaration *v
      = sb::buildVariableDeclaration(
          rose_util::generateUniqueName(outer_scope),
          get_exp->get_type(),
          NULL,
          outer_scope);
  // Replace the get expression with v
  si::replaceExpression(get_exp, sb::buildVarRefExp(v),
                        true);
  if (paired_get_exp)
    // Remove the get in this case since it is no longer used. 
    si::replaceExpression(paired_get_exp,
                          sb::buildVarRefExp(v), true);
  get_exp->set_rhs_operand(sb::buildVarRefExp(index_var));
  v->reset_initializer(sb::buildAssignInitializer(get_exp));
  si::insertStatementBefore(if_stmt, v);
}

static void ProcessConditionalExp(SgPntrArrRefExp *get_exp,
                                  const SgConditionalExp *cond_exp) {
  LOG_WARNING() << "Select not supported.\n";
  return;
}


void unconditional_get(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

  // Traverse grid gets in stencil kernels
  Rose_STL_Container<SgPntrArrRefExp *> gets;
  QueryGetInKernel(proj, gets);
  FOREACH(it, gets.begin(), gets.end()) {
    SgPntrArrRefExp *exp = *it;
    LOG_DEBUG() << "Get: "
                << exp->unparseToString() << "\n";
    SgNode *cond = rose_util::IsConditional(exp);
    if (!cond) {
      LOG_DEBUG() << "Not conditional\n";
      continue;
    }
    LOG_DEBUG() << "Conditional get to optimize: "
                << exp->unparseToString() << "\n";
    if (isSgIfStmt(cond)) {
      ProcessIfStmt(exp, isSgIfStmt(cond));
    } else if (isSgConditionalExp(cond)) {
      ProcessConditionalExp(exp, isSgConditionalExp(cond));
    }
  }
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

