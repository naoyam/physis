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

static void PeelFirstIterations(SgForStatement *loop,
                                int peel_size) {
  LOG_DEBUG() << "Peeling the first " << peel_size << " iteration(s)\n";
  PSAssert(peel_size > 0);
  SgInitializedName *loop_var = 
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop)->var();
  PSAssert(loop_var);
  SgForStatement *peeled_iterations =
      isSgForStatement(si::copyStatement(loop));
  LOG_DEBUG() << "Copying of original loop done.\n";  
  RenameLastStatementLabel(peeled_iterations);
  LOG_DEBUG() << "Label renaming done\n";
  FixGridAttributes(peeled_iterations);
  
  // Modify the termination condition to i < peel_size
  SgStatement *original_cond = peeled_iterations->get_test();
  SgExpression *loop_end = sb::buildIntVal(peel_size);
  SgStatement *cond =
      sb::buildExprStatement(
          sb::buildLessThanOp(sb::buildVarRefExp(loop_var),
                              si::copyExpression(loop_end)));
  RunKernelLoopAttribute *peeled_iter_attr
      = rose_util::GetASTAttribute<RunKernelLoopAttribute>(
          peeled_iterations);
  peeled_iter_attr->end() = si::copyExpression(loop_end);
  peeled_iter_attr->SetFirst();
  si::replaceStatement(original_cond, cond);
  si::insertStatementBefore(loop, peeled_iterations);
  si::removeStatement(original_cond);
  //si::deleteAST(original_cond);

  LOG_DEBUG() << "Condition changed\n";
  
  // Remove the original loop's initialization since the loop
  // induction variable is reused over from the peeled iteration
  // block.
#if 0
  // This doesn't work (no file_info assertion)
  SgForInitStatement *empty_init = sb::buildForInitStatement();
#else
  SgStatementPtrList empty_statement;
  SgForInitStatement *empty_init =
      sb::buildForInitStatement(empty_statement);
#endif  
  SgForInitStatement *original_init = loop->get_for_init_stmt();
  si::replaceStatement(original_init, empty_init);
  si::removeStatement(original_init);
  // This seems to make tree untraversable.
  //si::deleteAST(original_init);
  LOG_DEBUG() << "Init modified\n";

  RunKernelLoopAttribute *loop_attr
      = rose_util::GetASTAttribute<RunKernelLoopAttribute>(
          loop);
  loop_attr->begin() = rose_util::BuildMax(
      si::copyExpression(loop_attr->begin()),
      si::copyExpression(loop_end));
  LOG_DEBUG() << "Peeling of the first iterations done.\n";
  // Only copies of loop_end is used, so the original is not needed.
  si::deleteAST(loop_end);
  return;  
}

static SgExpression *FindGridRefInLoop(SgInitializedName *grid) {
  SgAssignInitializer *init = isSgAssignInitializer(grid->get_initptr());
  PSAssert(init);
  SgExpression *gr = init->get_operand();
  LOG_DEBUG() << "Grid var init: " << gr->unparseToString() << "\n";  
  return gr;
}

static void PeelLastIterations(SgForStatement *loop,
                               int peel_size,
                               SgInitializedName *peel_grid,
                               RuntimeBuilder *builder) {
  LOG_DEBUG() << "Peeling the last " << peel_size
              << " iteration(s)\n";  
  PSAssert(peel_size > 0);
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  SgInitializedName *loop_var = loop_attr->var();
  PSAssert(loop_var);
  SgForStatement *peeled_iterations =
      isSgForStatement(si::copyStatement(loop));
  LOG_DEBUG() << "Copying of original loop done.\n";
  RunKernelLoopAttribute *peel_iter_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(
          peeled_iterations);

  RenameLastStatementLabel(peeled_iterations);

  FixGridAttributes(peeled_iterations);
  
  // Modify the termination condition of the original loop
  SgStatement *original_cond = loop->get_test();
  // loop_end is min(original_end, g->dim - peel_size)
  SgExpression *original_loop_end = loop_attr->end();
  SgExpression *grid_ref_in_loop =
      si::copyExpression(FindGridRefInLoop(peel_grid));
  SgExpression *grid_end_offset =
      sb::buildSubtractOp(
          builder->BuildGridDim(grid_ref_in_loop,
                                loop_attr->dim()),
          sb::buildIntVal(peel_size));
  SgExpression *loop_end = rose_util::BuildMin(
      si::copyExpression(original_loop_end), grid_end_offset);
  SgStatement *cond =
      sb::buildExprStatement(
          sb::buildLessThanOp(sb::buildVarRefExp(loop_var),
                              loop_end));
  si::replaceStatement(original_cond, cond);
  // Remove old condtional
  si::removeStatement(original_cond);
  //si::deleteAST(original_cond);
  loop_attr->end() = si::copyExpression(loop_end);
  
  // Prepend the peeled iterations
  si::insertStatementAfter(loop, peeled_iterations);
  
  // Set the loop begin and end of the peeled iterations.
  // The end is not affected; only the beginning needs to be
  // corrected.
  peel_iter_attr->begin() = si::copyExpression(loop_end);
  peel_iter_attr->SetLast();
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
        peel_last_grid = grid_get_attr->gv();
        peel_size_last = peel_size;
      } else if (peel_last_grid == grid_get_attr->gv()) {
        peel_size_last = std::max(peel_size, peel_size_last);
      } else {
        // Peel only the grid found first.
        LOG_DEBUG() << "Ignoring peeling of grid: "
                    << grid_get_attr->gv()->get_name()
                    << "\n";
      }
    }
  }
  if (peel_size_first > 0) {
    PeelFirstIterations(loop, peel_size_first);
  } else {
    LOG_DEBUG() << "No profitable iteration found at the loop beginning.\n";
  }
  if (peel_size_last > 0) {
    PeelLastIterations(loop, peel_size_last,
                       peel_last_grid, builder);
  } else {
    LOG_DEBUG() << "No profitable iteration found at the loop end.\n";
  }
}

void loop_peeling(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

  std::vector<SgNode*> run_kernels =
      rose_util::QuerySubTreeAttribute<RunKernelAttribute>(proj);
  FOREACH (it, run_kernels.begin(), run_kernels.end()) {
    std::vector<SgNode*> run_kernel_loops =
        rose_util::QuerySubTreeAttribute<RunKernelLoopAttribute>(*it);
    if (run_kernel_loops.size() == 0) continue;
    SgForStatement *loop =
        isSgForStatement(run_kernel_loops.back());
    PSAssert(loop);
    RunKernelLoopAttribute *loop_attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
    LOG_DEBUG() << "Loop dimension: " << loop_attr->dim() << "\n";
    PeelLoop(isSgFunctionDeclaration(*it), loop, builder);
  }
  
  post_process(proj, tx, __FUNCTION__);
}


} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

