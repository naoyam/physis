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
#include <stack>

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

typedef map<string, SgExpressionVector> GridOffsetMap;

static SgExpression *extract_index_var(SgExpression *offset_dim_exp) {
  SgExpression *off =
      isSgVarRefExp(
          *(NodeQuery::querySubTree(offset_dim_exp,
                                    V_SgVarRefExp).begin()));
  return off;
}

static void build_index_var_list_from_offset_exp(
    SgExpression *offset_expr,
    SgExpressionPtrList *index_va_list) {
  PSAssert(isSgFunctionCallExp(offset_expr));
  SgExpressionPtrList &args =
      isSgFunctionCallExp(offset_expr)->get_args()->get_expressions();
  FOREACH (it, args.begin() + 1, args.end()) {
    index_va_list->push_back(si::copyExpression(
        extract_index_var(*it)));
    LOG_DEBUG() << "Offset expr: "
                << index_va_list->back()->unparseToString()
                << "\n";
  }
}

static bool IsASTNodeDescendant(SgNode *top, SgNode *x) {
  PSAssert(top);
  PSAssert(x);
  do {
    if (top == x) {
      return true;
    }
    x = x->get_parent();
  } while (x != NULL);
  return false;
}

/*!
  Insert the base offset decl, which needs to be after all the used
  variables. The kernel is inlined into the loop, so there is a
  sequence of declarations that are kernel parameters. Find all
  parameter variable delcarations, and append the base offset decl.
*/
static void insert_base_offset(SgBasicBlock *loop_body,
                               SgVariableDeclaration *base_offset_decl,
                               SgInitializedName *gv,
                               SgExpressionPtrList &base_offset_indices) {
  SgDeclarationStatementPtrList decls;
  if (gv) decls.push_back(gv->get_declaration());
  FOREACH (it, base_offset_indices.begin(), base_offset_indices.end()) {
    SgVarRefExp *v = isSgVarRefExp(*it);
    PSAssert(v);
    SgDeclarationStatement *d =
        v->get_symbol()->get_declaration()->get_declaration();
    PSAssert(d);
    if (IsASTNodeDescendant(loop_body, d)) {
      decls.push_back(d);
    }
  }
  
  // Advance all variables are found
  SgStatement *loop_body_stmt = loop_body->get_statements()[0];
  LOG_DEBUG() << "loop_body_stmt: " << loop_body_stmt->unparseToString() << "\n";
  std::stack<SgStatement*> stack;
  while (decls.size() > 0) {
    if (isSgBasicBlock(loop_body_stmt)) {
      if (si::getNextStatement(loop_body_stmt))
        stack.push(si::getNextStatement(loop_body_stmt));
      loop_body_stmt = si::getFirstStatement(isSgBasicBlock(loop_body_stmt));
    }
    SgDeclarationStatementPtrList::iterator it =
      std::find(decls.begin(), decls.end(), loop_body_stmt);
    if (it != decls.end()) {
      decls.erase(it);
    }
    loop_body_stmt = si::getNextStatement(loop_body_stmt);
    if (!loop_body_stmt) {
      if (stack.size() > 0) {
        loop_body_stmt = stack.top();
        stack.pop();
      } else {
        LOG_ERROR() << "Not all index var declrations found\n";
        PSAssert(0);
      }
    }
    LOG_DEBUG() << "loop_body_stmt: " << loop_body_stmt->unparseToString() << "\n";    
  }
  si::insertStatementBefore(loop_body_stmt, base_offset_decl);
}

/*!
  \param gvref Require copying
 */
static SgExpression *build_offset_periodic(SgExpression *offset_exp,
                                           int dim, int index_offset,
                                           SgExpression *gvref,
                                           RuntimeBuilder *builder) {
  /*
    if (index_offset > 0)
    then
       index_offset - dim
    else
       index_offset + dim
       
  or
  
    (((offset + dim) % dim) - x)
    
  */
  SgExpression *offset_d =
      si::copyExpression(
          isSgFunctionCallExp(offset_exp)->get_args()->get_expressions()[dim]);
  SgExpression *offset_periodic = NULL;
#ifdef OFFSET_USE_CONDITIONAL
  SgExpression *wrap_around_exp =
      (index_offset > 0) ? 
      (SgExpression*)sb::buildSubtractOp(
          sb::buildIntVal(index_offset),
          builder->BuildGridDim(si::copyExpression(gvref), dim)):
      (SgExpression*)sb::buildAddOp(
          sb::buildIntVal(index_offset),
          builder->BuildGridDim(si::copyExpression(gvref), dim));
  SgExpression *nowrap_around_exp = sb::buildIntVal(index_offset);
  SgExpression *offset_d =
      si::copyExpression(
          isSgFunctionCallExp(offset_exp)->get_args()->get_expressions()[dim]);
  SgExpression *cond_wrap =
      (index_offset > 0) ?
      (SgExpression*)sb::buildGreaterOrEqualOp(
          offset_d,
          builder->BuildGridDim(si::copyExpression(gvref), dim)) :
      (SgExpression*)sb::buildLessThanOp(
          offset_d, sb::buildIntVal(0));
  SgExpression *offset_periodic =
      sb::buildConditionalExp(
          cond_wrap, wrap_around_exp, nowrap_around_exp);
  LOG_DEBUG() << "Offset exp periodic: "
              << offset_periodic->unparseToString() << "\n";
#else
  // Use modulus
  SgExpression *x = sb::buildModOp(
      sb::buildAddOp(offset_d,
                     builder->BuildGridDim(si::copyExpression(gvref), dim)),
      builder->BuildGridDim(si::copyExpression(gvref), dim));
  offset_periodic = sb::buildSubtractOp(
      x,
      si::copyExpression(extract_index_var(offset_d)));
#endif
  LOG_DEBUG() << "Offset exp periodic: "
              << offset_periodic->unparseToString() << "\n";
  return offset_periodic;
}

static void replace_offset(SgVariableDeclaration *base_offset,
                           SgExpressionPtrList &offset_exprs,
                           RuntimeBuilder *builder) {
  FOREACH (it, offset_exprs.begin(), offset_exprs.end()) {
    SgExpression *original_offset_expr = *it;
    GridOffsetAttribute *offset_attr =
        rose_util::GetASTAttribute<GridOffsetAttribute>(
            original_offset_expr);
    SgExpression *gvexpr = offset_attr->gvexpr();
    const StencilIndexList sil = *offset_attr->GetStencilIndexList();
    PSAssert(StencilIndexRegularOrder(sil));
    StencilRegularIndexList sril(sil);
    LOG_DEBUG() << "SRIL: " << sril << "\n";
    int num_dims = sril.GetNumDims();
    SgExpression *dim_offset = sb::buildIntVal(1);
    SgExpression *new_offset_expr = sb::buildVarRefExp(base_offset);
    for (int i = 1; i <= num_dims; ++i) {
      int index_offset = sril.GetIndex(i);
      LOG_DEBUG() << "index-offset: " << index_offset << "\n";
      if (index_offset != 0) {
        SgExpression *offset_term = NULL;
        if (!offset_attr->periodic()) {
          offset_term = sb::buildIntVal(index_offset);
        } else {
          offset_term = build_offset_periodic(
              original_offset_expr, i,index_offset, gvexpr, builder);
        }
        new_offset_expr = sb::buildAddOp(
            new_offset_expr,            
            sb::buildMultiplyOp(dim_offset, offset_term));
      }
      dim_offset = sb::buildMultiplyOp(
          dim_offset,
          builder->BuildGridDim(si::copyExpression(gvexpr), i));
    }
    LOG_DEBUG() << "new offset expression: "
                << new_offset_expr->unparseToString() << "\n";
    si::replaceExpression(original_offset_expr, new_offset_expr);
  }
}

static void build_center_stencil_index_list(StencilIndexList &sil) {
  return;
}

static void do_offset_cse(RuntimeBuilder *builder,
                          SgForStatement *loop,
                          SgExpressionVector &offset_exprs) {
  SgBasicBlock *loop_body = isSgBasicBlock(loop->get_loop_body());
  PSAssert(loop_body);
  SgExpression *offset_expr = offset_exprs[0];
  GridOffsetAttribute *attr =
      rose_util::GetASTAttribute<GridOffsetAttribute>(offset_expr);
  SgExpressionPtrList base_offset_list;
  build_index_var_list_from_offset_exp(offset_expr, &base_offset_list);

  StencilIndexList sil;
  StencilIndexListInitSelf(sil, attr->num_dim());
  PSAssert(attr->gvexpr());
  SgExpression *base_offset = builder->BuildGridOffset(
      si::copyExpression(attr->gvexpr()),
      attr->num_dim(), &base_offset_list,
      true, false, &sil);
  LOG_DEBUG() << "base_offset: " << base_offset->unparseToString() << "\n";
  SgScopeStatement *func_scope =
      si::getEnclosingFunctionDefinition(loop);
  SgVariableDeclaration *base_offset_var
      = sb::buildVariableDeclaration(
          rose_util::generateUniqueName(func_scope),
          BuildIndexType2(rose_util::GetGlobalScope()),
          sb::buildAssignInitializer(
              base_offset, BuildIndexType2(rose_util::GetGlobalScope())),
          func_scope);
  LOG_DEBUG() << "base_offset_var: "
              << base_offset_var->unparseToString() << "\n";

  // NOTE: If the grid reference expression is a SgVarRefExp, it is
  // assumed to be a kernel parameter. In that case, kernel inlining
  // replaces the parameter with a new variable, and the base offset
  // declaration must be placed after that.
  SgInitializedName *gv = NULL;
  if (isSgVarRefExp(attr->gvexpr())) {
    gv = isSgVarRefExp(attr->gvexpr())->get_symbol()->get_declaration();
  }
  insert_base_offset(loop_body, base_offset_var, gv, base_offset_list);
  replace_offset(base_offset_var, offset_exprs, builder);
  return;
}

/*!

  Candidates are grid offset calculation that appears multiple
  times. Those expressions that have the same grid expression are
  considered accessing the same grid. This is a conservative
  assumption, and can be relaxed by more advanced code analysis.
 */
static GridOffsetMap find_candidates(SgForStatement *loop) {
  GridOffsetMap ggm;
  std::vector<SgNode*> offset_exprs =
      rose_util::QuerySubTreeAttribute<GridOffsetAttribute>(loop);
  FOREACH (it, offset_exprs.begin(), offset_exprs.end()) {
    SgExpression *offset_expr = isSgExpression(*it);
    LOG_DEBUG() << "offset: " << offset_expr->unparseToString() << "\n";
    GridOffsetAttribute *attr =
        rose_util::GetASTAttribute<GridOffsetAttribute>(offset_expr);
    PSAssert(attr);
    string gv = attr->gvexpr()->unparseToString();
    if (!isContained(ggm, gv)) {
      ggm.insert(std::make_pair(gv, SgExpressionVector()));
    }
    SgExpressionVector &v = ggm[gv];
    v.push_back(offset_expr);
  }

  GridOffsetMap::iterator ggm_it = ggm.begin();
  while (ggm_it != ggm.end()) {
    string grid_str = ggm_it->first;
    SgExpressionVector &get_exprs = ggm_it->second;
    // Needs multiple gets to do CSE
    if (get_exprs.size() <= 1) {
      GridOffsetMap::iterator ggm_it_next = ggm_it;
      ++ggm_it_next;
      ggm.erase(ggm_it);
      ggm_it = ggm_it_next;
      continue;
    }
    LOG_DEBUG() << grid_str
                << " has multiple gets\n";
    ++ggm_it;
  }
  return ggm;
}

/*!
  Assumes kernels are inlined.
*/
void offset_cse(
    SgProject *proj,
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

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
    LOG_DEBUG() << "No target loop for index CSE found\n";
    post_process(proj, tx, __FUNCTION__);
    return;
  }
  
  GridOffsetMap ggm = find_candidates(target_loop);
  FOREACH (it, ggm.begin(), ggm.end()) {
    SgExpressionVector &offset_exprs = it->second;
    do_offset_cse(builder, target_loop, offset_exprs);
  }
  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

