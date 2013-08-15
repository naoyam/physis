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

namespace si = SageInterface;
namespace sb = SageBuilder;

using std::vector;
using std::map;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {


static void RemoveDeadConditional(SgForStatement *loop,
                                  int dim, int peel_size_first,
                                  int peel_size_last,
                                  RunKernelLoopAttribute::Kind type);

//! Return true if a given get MAY be guarded by a conditional node.
/*!
  This is a conservative analysis. It is SAFE to always return true.
  
  \param get_exp A GridGet expression.
  \param if_stmt An if node that is a parent of the get expression.
  \param dim The dimension with off-region access.
  \return true if the conditional node may guard the get.
 */
static bool MayGuardGridAccess(const SgExpression *get_exp,
                               const SgIfStmt *if_stmt,
                               int dim) {
  return true;
}

//! Return true if a given get MAY be guarded by a conditional node.
/*!
  This is a conservative analysis. It is SAFE to always return true.
  
  \param get_exp A GridGet expression.
  \param cond_exp A conditional expression that is a parent of the
  get. 
  \param dim The dimension with off-region access.
  \return true if the conditional node may guard the get.
 */
static bool MayGuardGridAccess(const SgExpression *get_exp,
                               const SgConditionalExp *cond_exp,
                               int dim) {
  return true;
}

//! Return true if a given get MAY be guarded by a conditional node.
/*!
  This is a conservative analysis. It is SAFE to always return true.
  
  \param get_exp A GridGet expression.
  \param switch_stmt A switch statment that is a parent of the get. 
  \param dim The dimension with off-region access.
  \return true if the conditional node may guard the get.
 */
static bool MayGuardGridAccess(const SgExpression *get_exp,
                               const SgSwitchStatement *switch_stmt,
                               int dim) {
  return true;
}

static int FindProfitablePeelSize(const SgExpression *grid_get,
                                  const StencilIndexList &index_list,
                                  int loop_dim,
                                  SgStatement *loop_body) {
  if (!StencilIndexRegularOrder(index_list)) return 0;
  ssize_t offset = index_list[loop_dim-1].offset;
  if (offset == 0) {
    LOG_DEBUG() << "Center access\n";
    return 0;
  } else {
    LOG_DEBUG() << "Non-center access\n";
    // Find target peel size for first iterations
    // grid_get that potentially accesses a smaller offset point is 
    // found. 
    // If it is guarded by a condition with the loop induction
    // variable, this can be optimized by peeling the first loop
    // offset iterations.
    SgNode *parent = grid_get->get_parent();
    while (parent != loop_body) {
      PSAssert(parent);
      if ((isSgIfStmt(parent) &&
           MayGuardGridAccess(grid_get, isSgIfStmt(parent), loop_dim)) ||
          (isSgConditionalExp(parent) &&
           MayGuardGridAccess(grid_get, isSgConditionalExp(parent),
                              loop_dim)) ||
          (isSgSwitchStatement(parent) &&
           MayGuardGridAccess(grid_get, isSgSwitchStatement(parent),
                              loop_dim))) {
        LOG_DEBUG() << "Profitable access found: "
                    << grid_get->unparseToString() << "\n";
        return offset;
      }
      parent = parent->get_parent();
    }
    LOG_DEBUG() << "Found to be unconditional\n"; 
    return 0;
  }
}

static void RenameLastStatementLabel(SgForStatement *loop) {
  std::vector<SgNode*> labels =
      NodeQuery::querySubTree(loop, V_SgLabelStatement);
  FOREACH (labels_it, labels.begin(), labels.end()) {
    SgLabelStatement *label_stmt = isSgLabelStatement(*labels_it);
    SgStatement *last_stmt = si::getLastStatement(si::getScope(label_stmt));
    // Note: This doesn't work.
    //if (!si::isLastStatement(label_stmt)) {
    if (label_stmt != label_stmt) {
      LOG_DEBUG() << "Last: " << last_stmt->unparseToString() << "\n";
      continue;
    }
    SgName unique_name(rose_util::generateUniqueName(
        si::getGlobalScope(loop), "__label"));
    label_stmt->set_label(unique_name);
  }
}

static SgForStatement* PeelFirstIterations(SgForStatement *loop,
                                           int peel_size) {
  LOG_DEBUG() << "Peeling the first " << peel_size << " iteration(s)\n";
  PSAssert(peel_size > 0);
  SgVarRefExp *loop_var =
      KernelLoopAnalysis::GetLoopVar(loop);
  PSAssert(loop_var);
  LOG_INFO() << "Copying original loop (Warnings on AST copy may be issued. Seems safe to ignore.)\n";
  SgForStatement *peeled_iterations =
      isSgForStatement(si::copyStatement(loop));
  LOG_DEBUG() << "Copying of original loop done.\n";  
  RenameLastStatementLabel(peeled_iterations);
  LOG_DEBUG() << "Label renaming done\n";
  
  // Modify the termination condition to i < peel_size
  SgStatement *original_cond = peeled_iterations->get_test();
  SgExpression *loop_end = sb::buildIntVal(peel_size);
  SgStatement *cond =
      sb::buildExprStatement(
          sb::buildLessThanOp(si::copyExpression(loop_var),
                              si::copyExpression(loop_end)));
  RunKernelLoopAttribute *peeled_iter_attr
      = rose_util::GetASTAttribute<RunKernelLoopAttribute>(
          peeled_iterations);
#if 0  
  peeled_iter_attr->end() = si::copyExpression(loop_end);
#endif  
  peeled_iter_attr->SetFirst();
  si::replaceStatement(original_cond, cond);
  si::insertStatementBefore(loop, peeled_iterations);
  si::removeStatement(original_cond);
  //si::deleteAST(original_cond);

  LOG_DEBUG() << "Condition changed\n";
  
  // Remove the original loop's initialization since the loop
  // induction variable is reused over from the peeled iteration
  // block.
  SgStatementPtrList empty_statement;
  SgForInitStatement *empty_init =
      sb::buildForInitStatement(empty_statement);
  SgForInitStatement *original_init = loop->get_for_init_stmt();
  si::replaceStatement(original_init, empty_init);
  si::removeStatement(original_init);
  // This seems to make tree untraversable.
  //si::deleteAST(original_init);
  LOG_DEBUG() << "Init modified\n";

  LOG_DEBUG() << "Peeling of the first iterations done.\n";
  // Only copies of loop_end is used, so the original is not needed.
  si::deleteAST(loop_end);
  return peeled_iterations;
}

static SgExpression *FindGridRefInLoop(SgInitializedName *grid) {
  SgAssignInitializer *init = isSgAssignInitializer(grid->get_initptr());
  PSAssert(init);
  SgExpression *gr = init->get_operand();
  LOG_DEBUG() << "Grid var init: " << gr->unparseToString() << "\n";  
  return gr;
}

static SgForStatement* PeelLastIterations(SgForStatement *loop,
                                          int peel_size,
                                          SgInitializedName *peel_grid,
                                          RuntimeBuilder *builder) {
  LOG_DEBUG() << "Peeling the last " << peel_size
              << " iteration(s)\n";  
  PSAssert(peel_size > 0);
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  SgVarRefExp *loop_var = KernelLoopAnalysis::GetLoopVar(loop);
  PSAssert(loop_var);
  LOG_INFO() << "Copying original loop (Warnings on AST copy may be issued. Seems safe to ignore.)\n";  
  SgForStatement *peeled_iterations =
      isSgForStatement(si::copyStatement(loop));
  LOG_DEBUG() << "Copying of original loop done.\n";
  RunKernelLoopAttribute *peel_iter_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(
          peeled_iterations);

  RenameLastStatementLabel(peeled_iterations);

  // Modify the termination condition of the original loop
  SgStatement *original_cond = loop->get_test();
  // loop_end is min(original_end, g->dim - peel_size)
  SgExpression *original_loop_end =
      KernelLoopAnalysis::GetLoopEnd(loop);
  SgExpression *grid_ref_in_loop =
      sb::buildVarRefExp(peel_grid);
  SgExpression *grid_end_offset =
      sb::buildSubtractOp(
          builder->BuildGridDim(grid_ref_in_loop,
                                loop_attr->dim()),
          sb::buildIntVal(peel_size));
  SgExpression *loop_end = rose_util::BuildMin(
      si::copyExpression(original_loop_end), grid_end_offset);
  SgStatement *cond =
      sb::buildExprStatement(
          sb::buildLessThanOp(si::copyExpression(loop_var),
                              loop_end));
  si::replaceStatement(original_cond, cond);
  // Remove old condtional
  si::removeStatement(original_cond);
  //si::deleteAST(original_cond);
#if 0  
  loop_attr->end() = si::copyExpression(loop_end);
#endif  
  
  // Prepend the peeled iterations
  si::insertStatementAfter(loop, peeled_iterations);
  
  // Set the loop begin and end of the peeled iterations.
  // The end is not affected; only the beginning needs to be
  // corrected.
#if 0  
  peel_iter_attr->begin() = si::copyExpression(loop_end);
#endif  
  peel_iter_attr->SetLast();
  return peeled_iterations;
}

static void PeelLoop(
    SgFunctionDeclaration *run_kernel_func,
    SgForStatement *loop,
    RuntimeBuilder *builder) {
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  int dim = loop_attr->dim();
  SgStatement *loop_body = loop->get_loop_body();
  std::vector<SgNode*> grid_gets =
      rose_util::QuerySubTreeAttribute<GridGetAttribute>(loop);
  int peel_size_first = 0;
  int peel_size_last = 0;
  SgInitializedName *peel_last_grid = NULL;
  FOREACH (grid_gets_it, grid_gets.begin(), grid_gets.end()) {
    SgExpression *grid_get = isSgExpression(*grid_gets_it);
    GridGetAttribute *grid_get_attr =
        rose_util::GetASTAttribute<GridGetAttribute>(grid_get);
    const StencilIndexList &sil =
        *grid_get_attr->GetStencilIndexList();
    int peel_size = FindProfitablePeelSize(grid_get, sil, dim,
                                           loop_body);
    if (peel_size == 0) continue;
    // Find target peel size for first iterations
    if (peel_size < 0) {
      // grid_get that potentially accesses a smaller offset point is 
      // found. 
      // If it is guarded by a condition with the loop induction
      // variable, this can be optimized by peeling the first loop
      // offset iterations.
      peel_size_first = std::max(peel_size_first, std::abs(peel_size));
    } else {
      // Peel last iterations
      if (peel_last_grid == NULL) {
        peel_last_grid = GridGetAnalysis::GetGridVar(grid_get);
        peel_size_last = peel_size;
      } else if (peel_last_grid == GridGetAnalysis::GetGridVar(grid_get)) {
        peel_size_last = std::max(peel_size, peel_size_last);
      } else {
        // Peel only the grid found first.
        LOG_DEBUG() << "Ignoring peeling of grid: "
                    << GridGetAnalysis::GetGridVar(grid_get)->get_name()
                    << "\n";
      }
    }
  }
  if (peel_size_first > 0) {
    SgForStatement *peeled_loop = PeelFirstIterations(loop, peel_size_first);
    RemoveDeadConditional(
        peeled_loop,
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop)->dim(),
        peel_size_first, peel_size_last, RunKernelLoopAttribute::FIRST);
  } else {
    LOG_DEBUG() << "No profitable iteration found at the loop beginning.\n";
  }
  if (peel_size_last > 0) {
    SgForStatement *peeled_loop = PeelLastIterations(loop, peel_size_last,
                                                     peel_last_grid, builder);
    RemoveDeadConditional(
        peeled_loop,
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop)->dim(),
        peel_size_first, peel_size_last, RunKernelLoopAttribute::LAST);
    
  } else {
    LOG_DEBUG() << "No profitable iteration found at the loop end.\n";
  }

  if (peel_size_first > 0 || peel_size_last > 0) {
    RemoveDeadConditional(
        loop,
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop)->dim(),
        peel_size_first, peel_size_last, RunKernelLoopAttribute::MAIN);
  }
  
}

// Return 0 on success
static int FindExresssionReferencingLoopVar(SgBinaryOp *bop,
                                            SgInitializedName *loop_var,
                                            SgExpression *&v_expr,
                                            SgExpression *&c_expr) {
  SgExpression *lhs = bop->get_lhs_operand();
  SgExpression *rhs = bop->get_rhs_operand();  
  vector<SgVarRefExp*> vrefs = si::querySubTree<SgVarRefExp>(lhs);
  bool ref = false;
  FOREACH (it, vrefs.begin(), vrefs.end()) {
    if ((*it)->get_symbol()->get_declaration() == loop_var) {
      v_expr = lhs;
      c_expr = rhs;
      return 0;
    }
  }
  vrefs = si::querySubTree<SgVarRefExp>(rhs);
  ref = false;
  FOREACH (it, vrefs.begin(), vrefs.end()) {
    if ((*it)->get_symbol()->get_declaration() == loop_var) {
      v_expr = rhs;
      c_expr = lhs;
      return 0;
    }
  }

  LOG_DEBUG() << "No reference to " << loop_var->unparseToString() <<
      " found in " << bop->unparseToString() << "\n";
  return -1;
}

static bool IsRefToVar(SgVarRefExp *vref, SgInitializedName *var) {
  return vref->get_symbol()->get_declaration() == var;
}

static SgBinaryOp *FlipComparatorOp(SgBinaryOp *bop) {
  SgBinaryOp *flipped_op = NULL;
  SgExpression *lhs = si::copyExpression(bop->get_lhs_operand());
  SgExpression *rhs = si::copyExpression(bop->get_rhs_operand());
  if (isSgGreaterThanOp(bop)) {
    flipped_op = sb::buildLessOrEqualOp(lhs, rhs);
  } else if (isSgGreaterOrEqualOp(bop)) {
    flipped_op = sb::buildLessThanOp(lhs, rhs);
  } else if (isSgLessThanOp(bop)) {
    flipped_op = sb::buildGreaterOrEqualOp(lhs, rhs);
  } else if (isSgLessOrEqualOp(bop)) {
    flipped_op = sb::buildGreaterThanOp(lhs, rhs);
  }
  LOG_DEBUG() << bop->unparseToString() << " flipped to "
              << flipped_op->unparseToString() << "\n";
  return flipped_op;
}

static SgExpression *BuildTrue() {
  return sb::buildIntVal(1);
}

static SgExpression *BuildFalse() {
  return sb::buildIntVal(0);
}

//! Return true if expression is like "PSGridDim() - i", where i is
//! integer constant.
static int InvolveGridSize(SgExpression *exp, int dim) {
  SgSubtractOp *sop = isSgSubtractOp(exp);
  if (!sop) return -1;
  SgFunctionCallExp *lhs = isSgFunctionCallExp(sop->get_lhs_operand());
  SgValueExp *rhs = isSgValueExp(sop->get_rhs_operand());
  if (lhs == NULL || rhs == NULL) return -1;
  if (!si::isStrictIntegerType(rhs->get_type())) return -1;
  string func_name = rose_util::getFuncName(lhs);
  if (func_name != "PSGridDim") return -1;
  SgValueExp *dim_arg =
      isSgValueExp(lhs->get_args()->get_expressions()[1]);
  PSAssert(dim_arg);
  //LOG_DEBUG() << "Dim arg: " << dim_arg->unparseToString() << "\n";
  if ((int)si::getIntegerConstantValue(dim_arg) != dim-1) return -1;
  return si::getIntegerConstantValue(rhs);
}

static int Evaluate(SgBinaryOp *op,
                    SgExpression *rhs,
                    int peel_size_first, int peel_size_last,
                    RunKernelLoopAttribute::Kind kind,
                    int dim) {
  int minus_offset = InvolveGridSize(rhs, dim);
  if (minus_offset > 0) {
    if (isSgEqualityOp(op)) {
      switch (kind) {
        case RunKernelLoopAttribute::FIRST:
        case RunKernelLoopAttribute::MAIN:
          if (minus_offset <= peel_size_last) {
            return 0;
          } else {
            return -1;
          }
        case RunKernelLoopAttribute::LAST:
          if (minus_offset > peel_size_last) {
            return 0;
          } else {
            return -1;
          }
      }
    }
  } else if (isSgValueExp(rhs)) {
    int val = si::getIntegerConstantValue(isSgValueExp(rhs));
    if (isSgEqualityOp(op)) {
      switch (kind) {
        case RunKernelLoopAttribute::FIRST:
          if (val < peel_size_first) {
            return -1;
          } else {
            return 0;
          }
        case RunKernelLoopAttribute::MAIN:
        case RunKernelLoopAttribute::LAST:
          if (val < peel_size_first) {
            return 0;
          }
          return -1;
      }
    }
  }

  
  LOG_DEBUG() << "No static evaluation done for "
              << rhs->unparseToString() << "\n";
  return -1;    
}

static SgIfStmt *IsIfConditional(SgExpression *expr) {
  SgExprStatement *cond = isSgExprStatement(expr->get_parent());
  if (!cond) return false;
  SgIfStmt *if_stmt = isSgIfStmt(cond->get_parent());
  if (!if_stmt) return false;
  if (if_stmt->get_conditional() == cond) {
    return if_stmt;
  } else {
    return NULL;
  }
}

static void RemoveDeadConditional(SgForStatement *loop,
                                  int dim,
                                  int peel_size_first,
                                  int peel_size_last,
                                  RunKernelLoopAttribute::Kind kind) {
  vector<SgBinaryOp*> bin_ops = si::querySubTree<SgBinaryOp>(loop);
  SgInitializedName *loop_var =
      KernelLoopAnalysis::GetLoopVar(loop)->get_symbol()->get_declaration();
  FOREACH (it, bin_ops.begin(), bin_ops.end()) {
    SgBinaryOp *bin_op = isSgBinaryOp(si::copyExpression(*it));
    if (!(isSgEqualityOp(bin_op))) continue;
    LOG_DEBUG() << "Optimizing " << bin_op->unparseToString() << "?\n";    
    vector<SgExpression*> nested_exprs = si::querySubTree<SgExpression>(bin_op);
    SgVarRefExp *loop_var_ref = NULL;
    bool safe_to_opt = true;
    FOREACH (nested_exprs_it, nested_exprs.begin(), nested_exprs.end()) {
      SgExpression *nested_expr = *nested_exprs_it;
      if (nested_expr == bin_op) continue;
      if (isSgVarRefExp(nested_expr)) {
        SgVarRefExp *vref = isSgVarRefExp(nested_expr);
        if (!IsRefToVar(isSgVarRefExp(nested_expr), loop_var)) {
          // non loop var ref found; assumes non safe conditional
          LOG_DEBUG() << "Non loop var found\n";
          SgExpression *vdef =
              GetDeterministicDefinition(vref->get_symbol()->get_declaration());
          if (vdef) {
            LOG_DEBUG() << "Deterministic def found\n";
            si::replaceExpression(vref, si::copyExpression(vdef));
          } else {
            safe_to_opt = false;            
            break;
          }
        } else {
          if (loop_var_ref) {
            // multiple references to loop var found; not assumed;
            LOG_DEBUG() << "multiple var ref found\n";
            safe_to_opt = false;
            break;
          } else {
            loop_var_ref = isSgVarRefExp(nested_expr);
          }
        }
      } else if (isSgValueExp(nested_expr) ||
                 isSgAddOp(nested_expr) || isSgSubtractOp(nested_expr) ||
                 isSgUnaryAddOp(nested_expr)) {
        LOG_DEBUG() << "Valid expression: " <<
            nested_expr->unparseToString() << "\n";
        continue;
      } else {
        // not assumed expression
        LOG_DEBUG() << "Having " << nested_expr->unparseToString()
                    << " not assumed.\n";
        break;
      }
    }
    if (!safe_to_opt) {
      LOG_DEBUG() << "Not possible to optimize\n";
      continue;
    }
    // If no ref found, removing not possible
    if (!loop_var_ref) {
      LOG_DEBUG() << "No ref to loop var found; not possible to optimize\n";
      continue;
    }
    SgExpression *v_expr = NULL, *c_expr = NULL;
    PSAssert(FindExresssionReferencingLoopVar(bin_op, loop_var,
                                              v_expr, c_expr) == 0);
    LOG_DEBUG() << "v_expr: " << v_expr->unparseToString()
                << ", c_expr: " << c_expr->unparseToString() << "\n";

    //v_expr = si::copyExpression(v_expr);
    //c_expr = si::copyExpression(c_expr);
    SgBinaryOp *top_expr = bin_op;
    // Move the terms other than loop var ref to RHS
    while (true) {
      if (isSgVarRefExp(v_expr) &&
          IsRefToVar(isSgVarRefExp(v_expr), loop_var)) {
        break;
      } else if (isSgBinaryOp(v_expr)) {
        SgExpression *v_expr_child = NULL;
        SgExpression *c_expr_child = NULL;
        PSAssert(FindExresssionReferencingLoopVar(isSgBinaryOp(v_expr), loop_var,
                                                  v_expr_child, c_expr_child) == 0);
        if (isSgAddOp(v_expr)) {
          c_expr = sb::buildSubtractOp(c_expr, si::copyExpression(c_expr_child));
        } else {
          // subtract
          if (v_expr_child == isSgSubtractOp(v_expr)->get_lhs_operand()) {
            c_expr = sb::buildAddOp(c_expr, si::copyExpression(c_expr_child));
          } else {
            c_expr = sb::buildSubtractOp(c_expr, si::copyExpression(c_expr_child));
            top_expr = FlipComparatorOp(top_expr);
          }
        }
        v_expr = v_expr_child;
      } else if (isSgUnaryAddOp(v_expr)) {
        LOG_DEBUG() << "UnaryAdd: " << v_expr->unparseToString() << "\n";
        v_expr = isSgUnaryAddOp(v_expr)->get_operand();
        continue;
      } else {
        LOG_ERROR() << "Invalid AST node: " << v_expr->unparseToString() << "\n";
        PSAbort(1);
      }
    }
    si::constantFolding(c_expr);
    LOG_DEBUG() << "RHS: " << c_expr->unparseToString() << "\n";
    LOG_DEBUG() << "OPERATOR: " << top_expr->unparseToString() << "\n";
    int evaluated_result = Evaluate(top_expr, c_expr,
                                    peel_size_first, peel_size_last,
                                    kind, dim);
    if (evaluated_result >= 0) {
      LOG_DEBUG() << "Static evaluation done: " << evaluated_result << "\n";
      LOG_DEBUG() << "Replacing " << bin_op->unparseToString()
                  << " with " << evaluated_result << "\n";
      SgIfStmt *if_stmt = IsIfConditional(*it);
      if (if_stmt) {
        if (evaluated_result) {
          si::replaceStatement(if_stmt,
                               si::copyStatement(if_stmt->get_true_body()),
                               true);
        } else {
          si::replaceStatement(if_stmt,
                               si::copyStatement(if_stmt->get_false_body()),
                               true);
        }
      } else {
        si::replaceExpression(*it, evaluated_result? BuildTrue() : BuildFalse());
      }
    } else {
      LOG_DEBUG() << "Static evaluation not possible\n";
    }
  }
}

void loop_peeling(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

  vector<SgForStatement*> target_loops = FindInnermostLoops(proj);
  
  FOREACH (it, target_loops.begin(), target_loops.end()) {
    SgForStatement *target_loop = *it;
    RunKernelLoopAttribute *loop_attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(target_loop);
    LOG_DEBUG() << "Loop dimension: " << loop_attr->dim() << "\n";
    SgFunctionDeclaration *run_kernel_func =
        si::getEnclosingFunctionDeclaration(target_loop);
    PeelLoop(run_kernel_func, target_loop, builder);    
  }
  
  
  post_process(proj, tx, __FUNCTION__);
}


} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

