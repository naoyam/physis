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

typedef map<SgInitializedName*, SgExpressionVector> GridGetMap;

static SgExpression *extract_index_var(SgExpression *offset_dim_exp) {
  SgExpression *off =
      isSgVarRefExp(
          *(NodeQuery::querySubTree(offset_dim_exp,
                                    V_SgVarRefExp).begin()));
  return off;
}

static void build_index_var_list_from_offset_exp(
    SgExpression *offset_exp,
    SgExpressionPtrList *index_va_list) {
  PSAssert(isSgFunctionCallExp(offset_exp));
  SgExpressionPtrList &args =
      isSgFunctionCallExp(offset_exp)->get_args()->get_expressions();
  FOREACH (it, args.begin() + 1, args.end()) {
    index_va_list->push_back(si::copyExpression(
        extract_index_var(*it)));
    LOG_DEBUG() << "Offset exp: "
                << index_va_list->back()->unparseToString()
                << "\n";
  }
}

static SgExpression *extract_offset_exp(SgExpression *get_exp) {
  SgExpression *offset_exp = NULL;
  if (isSgFunctionCallExp(get_exp)) {
    LOG_ERROR() << "Not supproted\n";
  } else if (isSgBinaryOp(get_exp)) {
    offset_exp = isSgBinaryOp(get_exp)->get_rhs_operand();
  } else {
    LOG_ERROR() << "get_exp of type " << get_exp->class_name()
                << " is not supproted\n";
  }
  return offset_exp;
}

static void build_base_offset_var_list(
    SgExpression *get_exp,
    SgExpressionPtrList *offset_list) {
  SgExpression *offset_exp = extract_offset_exp(get_exp);
  assert (offset_exp);
  LOG_DEBUG() << "offset: " << offset_exp->unparseToString() << "\n";
  build_index_var_list_from_offset_exp(offset_exp,
                                       offset_list);
  return;
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
  decls.push_back(gv->get_declaration());
  FOREACH (it, base_offset_indices.begin(), base_offset_indices.end()) {
    SgVarRefExp *v = isSgVarRefExp(*it);
    PSAssert(v);
    SgDeclarationStatement *d =
        v->get_symbol()->get_declaration()->get_declaration();
    PSAssert(d);
    decls.push_back(d);
  }
  
  // Advance all variables are found
  SgStatement *loop_body_stmt = si::getFirstStatement(loop_body);
  std::stack<SgStatement*> stack;
  while (decls.size() > 0) {
    if (isSgBasicBlock(loop_body_stmt)) {
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
  }
  si::insertStatementBefore(loop_body_stmt, base_offset_decl);
}

static SgExpression *build_offset_periodic(SgExpression *offset_exp,
                                           int dim, int index_offset,
                                           SgInitializedName *gv,
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
          builder->BuildGridDim(sb::buildVarRefExp(gv), dim)):
      (SgExpression*)sb::buildAddOp(
          sb::buildIntVal(index_offset),
          builder->BuildGridDim(sb::buildVarRefExp(gv), dim));
  SgExpression *nowrap_around_exp = sb::buildIntVal(index_offset);
  SgExpression *offset_d =
      si::copyExpression(
          isSgFunctionCallExp(offset_exp)->get_args()->get_expressions()[dim]);
  SgExpression *cond_wrap =
      (index_offset > 0) ?
      (SgExpression*)sb::buildGreaterOrEqualOp(
          offset_d,
          builder->BuildGridDim(sb::buildVarRefExp(gv), dim)) :
      (SgExpression*)sb::buildLessThanOp(
          offset_d, sb::buildIntVal(0));
  SgExpression *offset_periodic =
      sb::buildConditionalExp(
          cond_wrap, wrap_around_exp, nowrap_around_exp);
  LOG_DEBUG() << "Offset exp periodic: "
              << offset_periodic->unparseToString() << "\n";
  return offset_periodic;
#else
  // Use modulus
  SgExpression *x = sb::buildModOp(
      sb::buildAddOp(offset_d,
                     builder->BuildGridDim(sb::buildVarRefExp(gv), dim)),
      builder->BuildGridDim(sb::buildVarRefExp(gv), dim));
  offset_periodic = sb::buildSubtractOp(
      x,
      si::copyExpression(extract_index_var(offset_d)));
#endif
  LOG_DEBUG() << "Offset exp periodic: "
              << offset_periodic->unparseToString() << "\n";
  return offset_periodic;
}

static void replace_offset(SgVariableDeclaration *base_offset,
                           SgExpressionPtrList &get_exprs,
                           SgInitializedName *gv,
                           RuntimeBuilder *builder) {
  FOREACH (it, get_exprs.begin(), get_exprs.end()) {
    SgExpression *get_expr = *it;
    GridGetAttribute *get_attr =
        rose_util::GetASTAttribute<GridGetAttribute>(get_expr);
    PSAssert(get_attr);
    GridOffsetAttribute *offset_attr =
        rose_util::GetASTAttribute<GridOffsetAttribute>(
            extract_offset_exp(get_expr));
    const StencilIndexList &sil = get_attr->GetStencilIndexList();
    PSAssert(StencilIndexRegularOrder(sil));
    StencilRegularIndexList sril(sil);
    LOG_DEBUG() << "SRIL: " << sril << "\n";
    int num_dims = sril.GetNumDims();
    SgExpression *dim_offset = sb::buildIntVal(1);
    SgExpression *offset_expr = sb::buildVarRefExp(base_offset);
    for (int i = 1; i <= num_dims; ++i) {
      int index_offset = sril.GetIndex(i);
      LOG_DEBUG() << "index-offset: " << index_offset << "\n";
      if (index_offset != 0) {
        SgExpression *offset_term = NULL;
        if (!offset_attr->periodic()) {
          offset_term = sb::buildIntVal(index_offset);
        } else {
          offset_term = build_offset_periodic(
              extract_offset_exp(get_expr), i,
              index_offset, gv, builder);
        }
        offset_expr = sb::buildAddOp(
            offset_expr,            
            sb::buildMultiplyOp(dim_offset, offset_term));
      }
      dim_offset = sb::buildMultiplyOp(
          dim_offset,
          builder->BuildGridDim(sb::buildVarRefExp(gv), i));
    }
    LOG_DEBUG() << "offset expression: "
                << offset_expr->unparseToString() << "\n";
    si::replaceExpression(extract_offset_exp(get_expr), offset_expr);
  }
}

static void do_offset_cse(RuntimeBuilder *builder,
                          SgForStatement *loop,
                          SgExpressionVector &get_exprs) {
  SgBasicBlock *loop_body = isSgBasicBlock(loop->get_loop_body());
  PSAssert(loop_body);
  SgExpression *get_expr = get_exprs[0];
  GridGetAttribute *attr =
      rose_util::GetASTAttribute<GridGetAttribute>(get_expr);
  SgExpressionPtrList base_offset_list;
  build_base_offset_var_list(get_expr, &base_offset_list);

  SgExpression *base_offset = builder->BuildGridOffset(
      sb::buildVarRefExp(attr->gv(), loop_body),
      attr->num_dim(), &base_offset_list,
      true, false);
  LOG_DEBUG() << "base_offset: " << base_offset->unparseToString() << "\n";
  SgVariableDeclaration *base_offset_var
      = sb::buildVariableDeclaration(
          rose_util::generateUniqueName(NULL),
          BuildIndexType2(loop_body),
          sb::buildAssignInitializer(
              base_offset, BuildIndexType2(loop_body)),
          loop);
  LOG_DEBUG() << "base_offset_var: "
              << base_offset_var->unparseToString() << "\n";

  insert_base_offset(loop_body, base_offset_var, attr->gv(),
                     base_offset_list);
  
  replace_offset(base_offset_var, get_exprs, attr->gv(), builder);
  return;
}

static GridGetMap find_candidates(SgForStatement *loop) {
  GridGetMap ggm;
  std::vector<SgNode*> get_exprs =
      rose_util::QuerySubTreeAttribute<GridGetAttribute>(loop);
  FOREACH (it, get_exprs.begin(), get_exprs.end()) {
    SgExpression *get_expr = isSgExpression(*it);
    LOG_DEBUG() << "get: " << get_expr->unparseToString() << "\n";
    GridGetAttribute *attr =
        rose_util::GetASTAttribute<GridGetAttribute>(get_expr);
    SgInitializedName *gv = attr->gv();
    if (!gv) {
      LOG_DEBUG() << "gv is null. Ignored.\n";
      continue;
    }
    // if the get expression is replaced with a local var, it is done
    // so by register blocking optimization. Those expresssions are
    // not handled in this optimization.
    if (isSgVarRefExp(get_expr)) {
      continue;
    }
    if (!isContained(ggm, gv)) {
      ggm.insert(std::make_pair(gv, SgExpressionVector()));
    }
    SgExpressionVector &v = ggm[gv];
    v.push_back(get_expr);
  }

  GridGetMap::iterator ggm_it = ggm.begin();
  while (ggm_it != ggm.end()) {
    SgInitializedName *gv = ggm_it->first;
    SgExpressionVector &get_exprs = ggm_it->second;
    // Needs multiple gets to do CSE
    if (get_exprs.size() <= 1) {
      GridGetMap::iterator ggm_it_next = ggm_it;
      ++ggm_it_next;
      ggm.erase(ggm_it);
      ggm_it = ggm_it_next;
      continue;
    }
    LOG_DEBUG() << gv->unparseToString()
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
  
  LOG_DEBUG() << "loop: " << target_loop->unparseToString() << "\n";
  GridGetMap ggm = find_candidates(target_loop);
  FOREACH (it, ggm.begin(), ggm.end()) {
    SgExpressionVector &get_exprs = it->second;
    do_offset_cse(builder, target_loop, get_exprs);
  }

  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

