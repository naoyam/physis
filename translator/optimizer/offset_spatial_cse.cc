// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include <boost/foreach.hpp>

#include "translator/optimizer/optimization_passes.h"
#include "translator/optimizer/optimization_common.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/translation_util.h"

#include <algorithm>
#include <stack>

namespace si = SageInterface;
namespace sb = SageBuilder;

using namespace std;
using namespace physis;
using namespace physis::translator;

namespace {

SgFunctionCallExp *ExtractOffsetCall(SgExpression *offset) {
  if (isSgAddOp(offset)) {
    SgAddOp *aop = isSgAddOp(offset);
    if (isSgFunctionCallExp(aop->get_lhs_operand())) {
      return isSgFunctionCallExp(aop->get_lhs_operand());
    }
  }
  //PSAssert(isSgFunctionCallExp(offset));
  return isSgFunctionCallExp(offset);
}

/*! Replace a variable reference in an offset expression.

  \param vref Variable reference
  \param loop_body Target loop 
 */
static void replace_var_ref(SgVarRefExp *vref, SgBasicBlock *loop_body) {
  SgVariableDeclaration *vdecl = isSgVariableDeclaration(
      vref->get_symbol()->get_declaration()->get_declaration());
  vector<SgVariableDeclaration*> decls =
      si::querySubTree<SgVariableDeclaration>(loop_body, V_SgVariableDeclaration);
  if (!isContained(decls, vdecl)) return;
  if (!vdecl) return;
  SgAssignInitializer *init =
      isSgAssignInitializer(
          vdecl->get_definition()->get_vardefn()->get_initializer());
  if (!init) return;
  SgExpression *rhs = init->get_operand();
  LOG_DEBUG() << "RHS: " << rhs->unparseToString() << "\n";
  si::replaceExpression(vref, si::copyExpression(rhs));
  return;
}

/*! Fix variable references in offset expression.

  Offset expressions may use a variable reference to grids and loop
  indices that are defined inside the target loop. They need to be
  replaced with their original value when moved out of the loop.

  \param offset_expr Offset expression
  \param loop_body Target loop body
*/
static void replace_arg_defined_in_loop(SgExpression *offset_expr,
                                        SgBasicBlock *loop_body) {
  LOG_DEBUG() << "Replacing undef var in "
              << offset_expr->unparseToString() << "\n";
  PSAssert(offset_expr != NULL && loop_body != NULL);
  SgFunctionCallExp *offset_call = ExtractOffsetCall(offset_expr);
  PSAssert(offset_call);
  SgExpressionPtrList &args = offset_call->get_args()->get_expressions();
  if (isSgVarRefExp(args[0])) {
    replace_var_ref(isSgVarRefExp(args[0]), loop_body);
  }
  LOG_DEBUG() << "Replaced: " << offset_expr->unparseToString() << "\n";
  return;
}

/*! Set the initial offset index.

  The offset expression is copied from an expression within the
  loop. The expression uses the loop index variables so that it
  can be used for any loop iteration. It needs to be adjusted to get
  the offset for the first index variable.
  
  \param offset_expr Target offset expression
  \param loop Target innermost loop
 */
static void replace_initial_index(SgExpression *offset_expr,
                                  SgForStatement *loop,
                                  GridOffsetAttribute *offset_attr) {
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  int dim = loop_attr->dim();
  const StencilIndexList *sil = offset_attr->GetStencilIndexList();  
  int target_index_order = StencilIndexListFindDim(sil, dim);
  LOG_DEBUG() << "target index order: " << target_index_order
              << "\n";
  if (target_index_order < 0) {
    LOG_DEBUG() << "No need for replacement since target index not used.\n";
    return;
  }
  SgExpression *target_index =
      ExtractOffsetCall(offset_expr)->get_args()->get_expressions()[
          target_index_order+1]; // +1 to ignore the first arg 
  vector<SgVarRefExp*> vref =
      si::querySubTree<SgVarRefExp>(target_index, V_SgVarRefExp);
  PSAssert(vref.size() == 1);
  SgExpression *begin_exp = KernelLoopAnalysis::GetLoopBegin(loop);
  if (begin_exp)
    si::replaceExpression(vref[0], si::copyExpression(begin_exp));
  return;
}


/*! Insert an increment statement at the end of the loop body.
  
  \param vdecl Offset variable declaration
  \param loop Target innermost loop
  \param builder RuntimeBuilder
 */
static void insert_offset_increment_stmt(SgVariableDeclaration *vdecl,
                                         SgForStatement *loop,
                                         RuntimeBuilder *builder,
                                         GridOffsetAttribute *offset_attr) {
  SgExpression *rhs = rose_util::GetVariableDefinitionRHS(vdecl);
  SgExpression *grid_ref =
      ExtractOffsetCall(rhs)->get_args()->get_expressions()[0];
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  int dim = loop_attr->dim();
  LOG_DEBUG() << "dim: " << dim << "\n";
  SgFunctionDeclaration *func = si::getEnclosingFunctionDeclaration(loop);
  RunKernelAttribute *rk_attr =
      rose_util::GetASTAttribute<RunKernelAttribute>(func);
  PSAssert(rk_attr);
  bool rb = rk_attr->stencil_map()->IsRedBlackVariant();
  const StencilIndexList *sil = offset_attr->GetStencilIndexList();
  PSAssert(sil);

  if (StencilIndexListFindDim(sil, dim) < 0) {
    LOG_DEBUG() << "No offset increment since the grid is not associated with the dimension.\n";
    LOG_DEBUG() << "rhs: " << rhs->unparseToString() << "\n";    
    LOG_DEBUG() << "sil->size(): " << sil->size() << "\n";
    return;
  }

  SgExpression *increment = NULL;

  ENUMERATE (i, it, sil->begin(), sil->end()) {
    const StencilIndex &si = *it;
    if (dim == si.dim) break;
    SgExpression *d = builder->BuildGridDim(si::copyExpression(grid_ref), i+1);
    increment = increment ? sb::buildMultiplyOp(increment, d) : d;
  }
  
  // if the access is periodic, the offset is:
  // (i + 1 + n) % n - ((i + n)% n)
  if (offset_attr->periodic()) {
    LOG_DEBUG() << "Periodic access\n";
    SgVarRefExp *loop_var = KernelLoopAnalysis::GetLoopVar(loop);
    int offset = (*sil)[loop_attr->dim()-1].offset;
    SgExpression *offset_expr = offset != 0 ?
        (SgExpression*)sb::buildAddOp(si::copyExpression(loop_var),
                                      sb::buildIntVal(offset)):
        (SgExpression*)si::copyExpression(loop_var);
    SgExpression *e =
        sb::buildModOp(
            sb::buildAddOp(
                sb::buildAddOp(offset_expr,
                               sb::buildIntVal(rb? 2 : 1)),
                builder->BuildGridDim(
                    si::copyExpression(grid_ref), loop_attr->dim())),
            builder->BuildGridDim(
                si::copyExpression(grid_ref), loop_attr->dim()));
    e = sb::buildSubtractOp(
        e,
        sb::buildModOp(
            sb::buildAddOp(
                si::copyExpression(offset_expr),
                builder->BuildGridDim(
                    si::copyExpression(grid_ref), loop_attr->dim())),
            builder->BuildGridDim(
                si::copyExpression(grid_ref), loop_attr->dim())));
    increment = increment? sb::buildMultiplyOp(increment, e) : e;
  }
  // loop over the unit-stride dimension
  if (!increment) {
    increment = sb::buildIntVal(rb ? 2: 1);
  }
  PSAssert(offset_attr);
  SgStatement *stmt =
      sb::buildExprStatement(
          sb::buildPlusAssignOp(sb::buildVarRefExp(vdecl), increment));
  si::appendStatement(stmt, isSgScopeStatement(loop->get_loop_body()));
  LOG_DEBUG() << "Inserting " << stmt->unparseToString() << "\n";
  return;
}

static SgExpression *ExtractInvariantPart(SgExpression *offset_expr,
                                          SgForStatement *loop) {
  if (!isSgAddOp(offset_expr)) {
    return offset_expr;
  }
  // Test if member offset is invariant
  SgExpression *offset_lhs = isSgAddOp(offset_expr)->get_lhs_operand();  
  SgExpression *member_offset = isSgAddOp(offset_expr)->get_rhs_operand();
  vector<SgVarRefExp*> var_refs = si::querySubTree<SgVarRefExp>(member_offset);
  BOOST_FOREACH (SgVarRefExp *v, var_refs) {
    SgDeclarationStatement *decl =
        si::convertRefToInitializedName(v)->get_declaration();
    if (si::isAncestor(loop, decl)) {
      LOG_DEBUG() << "Variable " << decl->unparseToString()
                  << " is declared inside the loop, disallowing offset spatial CSE.\n";
      return offset_lhs;
    }
  }
  return offset_expr;
}

/*! Perform the optimization on an offset computation expression.
  
  \param offset_expr Offset expression
  \param loop Innermost loop
  \param builder RuntimeBuilder
 */
static void do_offset_spatial_cse(SgExpression *offset_expr,
                                  SgForStatement *loop,
                                  RuntimeBuilder *builder) {
  LOG_DEBUG() << "Offset expr: "
              << offset_expr->unparseToString() << "\n";
  SgStatement *containing_stmt =
      si::getEnclosingStatement(offset_expr);
  LOG_DEBUG() << "Statement: "
              << containing_stmt->unparseToString()
              << ", "
              << containing_stmt->class_name() 
              << "\n";

  GridOffsetAttribute *offset_attr =
      rose_util::GetASTAttribute<GridOffsetAttribute>(offset_expr);
  offset_expr = ExtractInvariantPart(offset_expr, loop);
  LOG_DEBUG() << "CSE target: " << offset_expr->unparseToString() << "\n";

  SgVariableDeclaration *vdecl = NULL;
  if (isSgVariableDeclaration(containing_stmt)) {
    SgVariableDeclaration *t = isSgVariableDeclaration(containing_stmt);
    SgVariableDefinition *var_def = t->get_definition();
    SgAssignInitializer *init = isSgAssignInitializer(
        var_def->get_vardefn()->get_initializer());
    if (init) {
      LOG_DEBUG() << "init: " << init->unparseToString() << "\n";
      SgExpression *rhs = init->get_operand();
      if (rhs == offset_expr) {
        vdecl = isSgVariableDeclaration(si::copyStatement(t));
        si::removeStatement(t);
      }
    }
  }

  SgScopeStatement *func_scope =
      si::getEnclosingFunctionDefinition(loop);
  if (!vdecl) {
    LOG_DEBUG() << "original statement is not a vdecl\n";
    SgExpression *new_offset_expr = si::copyExpression(offset_expr);
    vdecl =  sb::buildVariableDeclaration(
        rose_util::generateUniqueName(func_scope),
        BuildIndexType2(rose_util::GetGlobalScope()),
        sb::buildAssignInitializer(new_offset_expr),
        func_scope);
    SgExpression *vref = sb::buildVarRefExp(vdecl);
    si::replaceExpression(offset_expr, vref, true);
    offset_expr = new_offset_expr;
  }

  si::insertStatementBefore(loop, vdecl);
  LOG_DEBUG() << "vdecl: " << vdecl->unparseToString() << "\n";

  replace_arg_defined_in_loop(rose_util::GetVariableDefinitionRHS(vdecl),
                              isSgBasicBlock(loop->get_loop_body()));
  replace_initial_index(rose_util::GetVariableDefinitionRHS(vdecl),
                        loop, offset_attr);
  insert_offset_increment_stmt(vdecl, loop, builder,
                               offset_attr);
  return;
}

} // namespace

namespace physis {
namespace translator {
namespace optimizer {
namespace pass {

/*! Optimizing offset computation by CSE over the innermost loop.

  Move offset computation out of the innermost loop.

  E.g.,
  Translate this:
  \code
  for
    offset = PSGridOffset();
    computation
  end
  \endcode
  to:
  \code
  offset = PSGridOffset();
  for
     computation
     offset += increment
  end
  \endcode
  
  Assumes kernels are inlined.
*/
void offset_spatial_cse(
    SgProject *proj,
    ::physis::translator::TranslationContext *tx,
    ::physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

  vector<SgForStatement*> target_loops = FindInnermostLoops(proj);

  FOREACH (it, target_loops.begin(), target_loops.end()) {
    SgForStatement *target_loop = *it;
    RunKernelLoopAttribute *attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(target_loop);
    if (!attr->IsMain()) continue;
    SgFunctionDeclaration *func = si::getEnclosingFunctionDeclaration(target_loop);
    RunKernelAttribute *rk_attr =
        rose_util::GetASTAttribute<RunKernelAttribute>(func);
    // NOTE: Not implemented for red-black stencils with
    // non-unit-stride looping caases. Would be very complicated, but
    // not clear how much it would be actually effective
    if (rk_attr->stencil_map()->IsRedBlackVariant() &&
        attr->dim() != 1) {
      LOG_WARNING() <<
          "Spatial offset CSE not applied because it is not implemented for this type of stencil.\n";
      continue;
    }
        
    std::vector<SgNode*> offset_exprs =
        rose_util::QuerySubTreeAttribute<GridOffsetAttribute>(
            target_loop);
    FOREACH (it, offset_exprs.begin(), offset_exprs.end()) {
      SgExpression *offset_expr = isSgExpression(*it);
      if (ExtractOffsetCall(offset_expr) == NULL) {
        LOG_DEBUG() << "Ignoring offset with no call in "
                    << offset_expr->unparseToString() << "\n";
        continue;
      }
      PSAssert(offset_expr);
      do_offset_spatial_cse(offset_expr, target_loop, builder);
    }
  }

  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

