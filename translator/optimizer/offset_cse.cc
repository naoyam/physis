// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/builder_interface.h"
#include "translator/translation_util.h"

#include <algorithm>
#include <stack>

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

typedef map<SgVariableSymbol*, SgExpressionVector> GridOffsetMap;

static SgExpression *extract_index_var(SgExpression *offset_dim_exp) {
  SgNodePtrList v = NodeQuery::querySubTree(offset_dim_exp,
                                            V_SgVarRefExp);
  if (v.size() == 0) return NULL;
  SgExpression *off = isSgVarRefExp(*(v.begin()));
  return off;
}

SgFunctionCallExp *ExtractOffsetCall(SgExpression *offset) {
  if (isSgAddOp(offset)) {
    SgAddOp *aop = isSgAddOp(offset);
    if (isSgFunctionCallExp(aop->get_lhs_operand())) {
      return isSgFunctionCallExp(aop->get_lhs_operand());
    }
  }
  PSAssert(isSgFunctionCallExp(offset));
  return isSgFunctionCallExp(offset);
}

static int build_index_var_list_from_offset_exp(
    SgExpression *offset_expr,
    SgExpressionPtrList *index_va_list) {
  offset_expr = ExtractOffsetCall(offset_expr);
  PSAssert(isSgFunctionCallExp(offset_expr));
  SgExpressionPtrList &args =
      isSgFunctionCallExp(offset_expr)->get_args()->get_expressions();
  FOREACH (it, args.begin() + 1, args.end()) {
    SgExpression *index_var = extract_index_var(*it);
    if (index_var == NULL) return EXIT_FAILURE;
    index_va_list->push_back(si::copyExpression(index_var));
    LOG_DEBUG() << "Offset expr: "
                << index_va_list->back()->unparseToString()
                << "\n";
  }
  return EXIT_SUCCESS;
}

/*!
  \param gvref Require copying
 */
static SgExpression *build_offset_periodic(SgExpression *offset_exp,
                                           int dim, int index_offset,
                                           SgExpression *gvref,
                                           BuilderInterface *builder) {
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
          builder->BuildGridDim(
              si::copyExpression(gvref), dim)):
      (SgExpression*)sb::buildAddOp(
          sb::buildIntVal(index_offset),
          builder->BuildGridDim(
              si::copyExpression(gvref), dim));
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
                     builder->BuildGridDim(
                         si::copyExpression(gvref), dim)),
      builder->BuildGridDim(
          si::copyExpression(gvref), dim));
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
                           BuilderInterface *builder) {
  FOREACH (it, offset_exprs.begin(), offset_exprs.end()) {
    SgExpression *original_offset_expr = *it;
    GridOffsetAttribute *offset_attr =
        rose_util::GetASTAttribute<GridOffsetAttribute>(
            original_offset_expr);
    original_offset_expr = ExtractOffsetCall(original_offset_expr);
    SgVarRefExp *gvref = GridOffsetAnalysis::GetGridVar(original_offset_expr);
    const StencilIndexList sil = *offset_attr->GetStencilIndexList();
    PSAssert(StencilIndexRegularOrder(sil));
    StencilRegularIndexList sril(sil);
    LOG_DEBUG() << "SRIL: " << sril << "\n";
    SgExpression *dim_offset = sb::buildIntVal(1);
    SgExpression *new_offset_expr = sb::buildVarRefExp(base_offset);
    StencilRegularIndexList::map_t indices = sril.indices();
    FOREACH (it, indices.begin(), indices.end()) {
      int i = it->first;
      int index_offset = it->second;
      LOG_DEBUG() << "index-offset: " << index_offset << "\n";
      if (index_offset != 0) {
        SgExpression *offset_term = NULL;
        if (!offset_attr->periodic()) {
          offset_term = sb::buildIntVal(index_offset);
        } else {
          offset_term = build_offset_periodic(
              original_offset_expr, i,index_offset, gvref, builder);
        }
        new_offset_expr = sb::buildAddOp(
            new_offset_expr,            
            sb::buildMultiplyOp(dim_offset, offset_term));
      }
      dim_offset = sb::buildMultiplyOp(
          dim_offset,
          builder->BuildGridDim(
              si::copyExpression(gvref), i));
    }
    si::constantFolding(new_offset_expr);
    LOG_DEBUG() << "new offset expression: "
                << new_offset_expr->unparseToString() << "\n";
    si::replaceExpression(original_offset_expr, new_offset_expr);
  }
}

static void build_center_stencil_index_list(StencilIndexList &sil) {
  return;
}

static int do_offset_cse(BuilderInterface *builder,
                          SgForStatement *loop,
                          SgExpressionVector &offset_exprs) {
  SgBasicBlock *loop_body = isSgBasicBlock(loop->get_loop_body());
  PSAssert(loop_body);
  SgExpression *offset_expr = offset_exprs[0];
  GridOffsetAttribute *attr =
      rose_util::GetASTAttribute<GridOffsetAttribute>(offset_expr);
  SgExpressionPtrList base_offset_list;
  if (build_index_var_list_from_offset_exp(offset_expr, &base_offset_list)) {
    return EXIT_FAILURE;
  }

  StencilIndexList sil = *attr->GetStencilIndexList();
  StencilIndexListClearOffset(sil);
  SgExpression *gref = si::copyExpression(GridOffsetAnalysis::GetGridVar(offset_expr));
  if (!isSgPointerType(gref->get_type())) {
    gref = sb::buildAddressOfOp(gref);
  }
  SgExpression *base_offset = builder->BuildGridOffset(
      gref,
      attr->rank(), &base_offset_list,
      &sil, true, false);
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

  SgScopeStatement *kernel =
      isSgScopeStatement(
          rose_util::QuerySubTreeAttribute<KernelBody>(loop).front());
  PSAssert(kernel);
  
  si::prependStatement(base_offset_var, kernel);
                       
  replace_offset(base_offset_var, offset_exprs, builder);
  return EXIT_SUCCESS;
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
    SgVarRefExp *gref = GridOffsetAnalysis::GetGridVar(offset_expr);
    LOG_DEBUG() << "gref: " << gref->unparseToString() << "\n";
    SgVariableSymbol *vs = gref->get_symbol();
    if (!isContained(ggm, vs)) {
      ggm.insert(std::make_pair(vs, SgExpressionVector()));
    }
    SgExpressionVector &v = ggm[vs];
    //v.push_back(ExtractCSETarget(offset_expr));
    v.push_back(offset_expr);
  }

  GridOffsetMap::iterator ggm_it = ggm.begin();
  while (ggm_it != ggm.end()) {
    SgVariableSymbol *vs = ggm_it->first;
    SgExpressionVector &get_exprs = ggm_it->second;
    // Needs multiple gets to do CSE
    if (get_exprs.size() <= 1) {
      GridOffsetMap::iterator ggm_it_next = ggm_it;
      ++ggm_it_next;
      ggm.erase(ggm_it);
      ggm_it = ggm_it_next;
      continue;
    }
    LOG_DEBUG() << vs->unparseToString() << " has multiple gets\n";
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
    physis::translator::BuilderInterface *builder) {
  pre_process(proj, tx, __FUNCTION__);

  vector<SgForStatement*> target_loops = FindInnermostLoops(proj);

  FOREACH (it, target_loops.begin(), target_loops.end()) {
    SgForStatement *target_loop = *it;
    RunKernelLoopAttribute *attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(target_loop);
    if (!attr->IsMain()) continue;
    GridOffsetMap ggm = find_candidates(target_loop);
    FOREACH (it, ggm.begin(), ggm.end()) {
      SgExpressionVector &offset_exprs = it->second;
      if (do_offset_cse(builder, target_loop, offset_exprs)) {
        LOG_DEBUG() << "Failed to apply offset CSE\n";
      }
    }
  }
  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

