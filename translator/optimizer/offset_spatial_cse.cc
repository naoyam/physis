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

#if 0
static bool is_constant(SgVarRefExp *e) {
  SgDeclarationStatement *decl =
      e->get_symbol()->get_declaration()->get_declaration();
  LOG_DEBUG() << "decl: " << decl->unparseToString() << "\n";
  return false;
}

static bool is_constant(SgExpression *e) {
  bool t = false;
  if (isSgVarRefExp(e)) {
    t = is_constant(isSgVarRefExp(e));
  } else if (isSgArrowExp(e)) {
    SgArrowExp *ae = isSgArrowExp(e);
    if (si::isConstType(ae->get_lhs_operand()->get_type())) {
      SgType *elmty = si::getElementType(
          isSgModifierType(ae->get_lhs_operand()->get_type())
          ->get_base_type());
      LOG_DEBUG() << "elmty: " << elmty->unparseToString() << "\n";
      t = si::isConstType(elmty);
    }
  }
  LOG_DEBUG() << e->unparseToString() << " constant?: "
              << t << "\n";
  return t;
}

static bool is_loop_var_ref(SgExpression *e) {
  SgVarRefExp *vref = isSgVarRefExp(e);
  if (!vref) return false;
  for (SgForStatement *loop = si::getEnclosingNode<SgForStatement>(vref);
       loop != NULL; loop = si::getEnclosingNode<SgForStatement>(loop, false)) {
    RunKernelLoopAttribute *loop_attr =
        rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
    PSAssert(loop_attr);
    SgInitializedName *loop_var = loop_attr->var();
    if (loop_var == vref->get_symbol()->get_declaration()) {
      LOG_DEBUG() << e->unparseToString() << " is a loop variable\n";
      return true;
    }
  }
  return false;
}

static void remove_redundant_variable_copy(SgForStatement *loop) {
  Rose_STL_Container<SgNode*> vdecls =
      NodeQuery::querySubTree(loop, V_SgVariableDeclaration);
  FOREACH (it, vdecls.begin(), vdecls.end()) {
    SgVariableDeclaration *decl = isSgVariableDeclaration(*it);
    SgVariableDefinition *var_def = decl->get_definition();
    if (!var_def) continue;
    SgAssignInitializer *init = isSgAssignInitializer(
        var_def->get_vardefn()->get_initializer());
    if (!init) continue;
    SgExpression *rhs = init->get_operand();
    SgType *type = rhs->get_type();
    LOG_DEBUG() << "RHS: " << rhs->unparseToString()
                << "\n";
    if (is_constant(rhs) || is_loop_var_ref(rhs)) {
      // replace decl with rhs
    }
  }
}

#endif

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
  SgFunctionCallExp *offset_call = isSgFunctionCallExp(offset_expr);
  PSAssert(offset_call);
  SgExpressionPtrList &args = offset_call->get_args()->get_expressions();
  if (isSgVarRefExp(args[0])) {
    replace_var_ref(isSgVarRefExp(args[0]), loop_body);
  }
  FOREACH (argit, args.begin()+1, args.end()) {
    SgExpression *arg = *argit;
    Rose_STL_Container<SgNode*> vref_list =
        NodeQuery::querySubTree(arg, V_SgVarRefExp);
    if (vref_list.size() == 0) continue;
    PSAssert(vref_list.size() == 1);
    replace_var_ref(isSgVarRefExp(vref_list[0]), loop_body);
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
                                  SgForStatement *loop) {
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  int dim = loop_attr->dim();
  SgExpression *target_index =
      isSgFunctionCallExp(offset_expr)->get_args()->get_expressions()[dim];
  vector<SgVarRefExp*> vref =
      si::querySubTree<SgVarRefExp>(target_index, V_SgVarRefExp);
  PSAssert(vref.size() == 1);
  si::replaceExpression(vref[0], si::copyExpression(loop_attr->begin()));
  return;
}

/*! Insert an increment statement at the end of the loop body.
  
  \param vdecl Offset variable declaration
  \param loop Target innermost loop
  \param builder RuntimeBuilder
 */
static void insert_offset_increment_stmt(SgVariableDeclaration *vdecl,
                                         SgForStatement *loop,
                                         RuntimeBuilder *builder) {
  SgExpression *rhs = rose_util::GetVariableDefinitionRHS(vdecl);
  SgExpression *grid_ref =
      isSgFunctionCallExp(rhs)->get_args()->get_expressions()[0];
  RunKernelLoopAttribute *loop_attr =
      rose_util::GetASTAttribute<RunKernelLoopAttribute>(loop);
  LOG_DEBUG() << "dim: " << loop_attr->dim() << "\n";
  SgExpression *increment = NULL;
  for (int i = 1; i < loop_attr->dim(); ++i) {
    SgExpression *d = builder->BuildGridDim(si::copyExpression(grid_ref), i);
    increment = increment ? sb::buildMultiplyOp(increment, d) : d;
  }
  GridOffsetAttribute *offset_attr =
      rose_util::GetASTAttribute<GridOffsetAttribute>(rhs);
  const StencilIndexList *sil = offset_attr->GetStencilIndexList();
  PSAssert(sil);
  // if the access is periodic, the offset is:
  // (i + 1) % n - i
  if (offset_attr->periodic()) {
    LOG_DEBUG() << "Periodic access\n";
    SgVariableDeclaration *loop_var_decl =
        isSgVariableDeclaration(loop_attr->var()->get_declaration());
    int offset = (*sil)[loop_attr->dim()-1].offset;
    SgExpression *offset_expr = offset != 0 ?
        (SgExpression*)sb::buildAddOp(sb::buildVarRefExp(loop_var_decl),
                                      sb::buildIntVal(offset)):
        (SgExpression*)sb::buildVarRefExp(loop_var_decl);
    SgExpression *e =
        sb::buildModOp(
            sb::buildAddOp(offset_expr, sb::buildIntVal(1)),
            builder->BuildGridDim(si::copyExpression(grid_ref), loop_attr->dim()));
    e = sb::buildSubtractOp(e, si::copyExpression(offset_expr));
    increment = increment? sb::buildMultiplyOp(increment, e) : e;
  }
  if (!increment) increment = sb::buildIntVal(1);  
  PSAssert(offset_attr);
  SgStatement *stmt =
      sb::buildExprStatement(
          sb::buildPlusAssignOp(sb::buildVarRefExp(vdecl), increment));
  si::appendStatement(stmt, isSgScopeStatement(loop->get_loop_body()));
  LOG_DEBUG() << "Inserting " << stmt->unparseToString() << "\n";
  return;
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
                        loop);
  insert_offset_increment_stmt(vdecl, loop, builder);
  return;
}

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
    physis::translator::TranslationContext *tx,
    physis::translator::RuntimeBuilder *builder) {
  pre_process(proj, tx, __FUNCTION__);

  SgForStatement *target_loop = FindInnermostLoop(proj);

  std::vector<SgNode*> offset_exprs =
      rose_util::QuerySubTreeAttribute<GridOffsetAttribute>(
          target_loop);
  FOREACH (it, offset_exprs.begin(), offset_exprs.end()) {
    SgExpression *offset_expr = isSgExpression(*it);
    PSAssert(offset_expr);
    do_offset_spatial_cse(offset_expr, target_loop, builder);
  }
  
  post_process(proj, tx, __FUNCTION__);  
}

} // namespace pass
} // namespace optimizer
} // namespace translator
} // namespace physis

