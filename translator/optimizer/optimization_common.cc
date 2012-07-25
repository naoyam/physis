// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/optimizer/optimization_common.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {

static void FixGridOffsetAttributeFuncCall(SgFunctionCallExp *offset_exp,
                                           GridOffsetAttribute *goa) {
  SgExpressionVector &indices = goa->indices();
  SgExpressionPtrList &args = offset_exp->get_args()->get_expressions();
  indices.clear();
  FOREACH (it, ++args.begin(), args.end()) {
    indices.push_back(*it);
  }
  SgVarRefExp *grid_ref =
      isSgVarRefExp(offset_exp->get_args()->get_expressions()[0]);
  PSAssert(grid_ref);
  goa->gvexpr() = grid_ref;
}

static void FixGridOffsetAttributeInlined(SgBinaryOp *offset_exp,
                                          GridOffsetAttribute *goa) {
  // This turnes out to be rather difficult. It would be rather
  // simpler to change the code generation method to use the function
  // call format.
  LOG_ERROR() << "Unsupported: "
              << offset_exp->unparseToString()
              << "\n";
  PSAbort(1);
  return;
}

void FixGridOffsetAttribute(SgExpression *offset_exp) {
  GridOffsetAttribute *goa =
      rose_util::GetASTAttribute<GridOffsetAttribute>(offset_exp);
  if (!goa) {
    LOG_ERROR() << "No GridOffsetAttribute found: "
                << offset_exp->unparseToString()
                << "\n";
    PSAssert(0);
  }

  offset_exp = rose_util::removeCasts(offset_exp);
  if (isSgFunctionCallExp(offset_exp)) {
    FixGridOffsetAttributeFuncCall(isSgFunctionCallExp(offset_exp), goa);
  } else if (isSgBinaryOp(offset_exp)) {
    FixGridOffsetAttributeInlined(isSgBinaryOp(offset_exp), goa);
  } else {
    LOG_ERROR() << "Unsupported offset expression: "
                << offset_exp->unparseToString() << "\n";
    PSAbort(1);
  }
}

void FixGridGetAttribute(SgExpression *get_exp) {
  GridGetAttribute *gga =
      rose_util::GetASTAttribute<GridGetAttribute>(get_exp);
  
  // Extract the correct offset expression
  SgExpression *new_offset = NULL;
  SgVarRefExp *new_grid = NULL;
  if (isSgPointerDerefExp(get_exp)) {
    get_exp = isSgPointerDerefExp(get_exp)->get_operand();
  }
  get_exp = rose_util::removeCasts(get_exp);
  if (isSgBinaryOp(get_exp)) {
    new_offset = isSgBinaryOp(get_exp)->get_rhs_operand();
    new_grid = isSgVarRefExp(isSgBinaryOp(
        rose_util::removeCasts(isSgBinaryOp(get_exp)->get_lhs_operand()))
                             ->get_lhs_operand());
    PSAssert(new_offset);    
    PSAssert(new_grid);
  } else if (isSgFunctionCallExp(get_exp)) {
    // TODO: offset is not a single expression in mpi and mpi-cuda
    // yet.
    new_offset = NULL;
    new_grid = isSgVarRefExp(
        isSgFunctionCallExp(get_exp)->get_args()->get_expressions()[0]);
    PSAssert(new_grid);
  } else {
    
    LOG_ERROR() << "Unsupported grid get: "
                << get_exp->unparseToString() << "\n";
    PSAbort(1);
  }

  gga->offset() = new_offset;
  gga->gv() = new_grid->get_symbol()->get_declaration();
  
  // NOTE: new_offset does not have offset attribute if it is, for
  // example, a reference to a variable.
  if (new_offset != NULL &&
      rose_util::GetASTAttribute<GridOffsetAttribute>(new_offset)) {
    FixGridOffsetAttribute(new_offset);
  }
  //LOG_DEBUG() << "Fixed GridGetAttribute\n";
  return;
}

void FixGridAttributes(
    SgNode *node) {
  Rose_STL_Container<SgNode *> exps =
      NodeQuery::querySubTree(node, V_SgExpression);
  FOREACH(it, exps.begin(), exps.end()) {
    SgExpression *exp = isSgExpression(*it);
    PSAssert(exp);
    
    // Traverse only GridGet
    GridGetAttribute *gga =
        rose_util::GetASTAttribute<GridGetAttribute>(exp);
    if (!gga) continue;
    FixGridGetAttribute(exp);
  }
 
  return;
}

vector<SgForStatement*> FindInnermostLoops(SgNode *proj) {
  std::vector<SgNode*> run_kernel_loops =
      rose_util::QuerySubTreeAttribute<RunKernelLoopAttribute>(proj);
  vector<SgForStatement*> target_loops;
  FOREACH (run_kernel_loops_it, run_kernel_loops.begin(),
           run_kernel_loops.end()) {
    SgForStatement *loop = isSgForStatement(*run_kernel_loops_it);
    PSAssert(loop);
    std::vector<SgNode*> nested_loops =
        rose_util::QuerySubTreeAttribute<RunKernelLoopAttribute>(loop);
    bool has_nested_loop = false;
    FOREACH (it, nested_loops.begin(), nested_loops.end()) {
      if (loop != *it) {
        has_nested_loop = true;
        break;
      }
    }
    // This is not an innermost loop    
    if (has_nested_loop) continue;

    target_loops.push_back(loop);
  }
  if (target_loops.size() == 0) {
    LOG_DEBUG() << "No target loop found\n";
  }

  return target_loops;
}

static SgInitializedName *GetVariable(SgVarRefExp *e) {
  return e->get_symbol()->get_declaration();
}

void GetVariableSrc(SgInitializedName *v,
                    vector<SgExpression*> &src_exprs) {
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

  
  FOREACH (it, assign_exprs.begin(), assign_exprs.end()) {
    SgNode *node = *it;
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
    src_exprs.push_back(operand_expr);
  }
}


static bool IsSafeToEliminate(SgExpression *exp) {
  LOG_DEBUG() << "Safe to eliminate?: " << exp->unparseToString() << "\n";
  
  // Conservatively assumes func call except for PSGridDim is unsafe
  const vector<SgFunctionCallExp*> &exprs
      = si::querySubTree<SgFunctionCallExp>(exp);
  FOREACH (it, exprs.begin(), exprs.end()) {
    SgFunctionCallExp *call = *it;
    std::string func_name = rose_util::getFuncName(call);
    if (func_name != "PSGridDim") {
      return false;
    }
  }

  if (si::querySubTree<SgAssignOp>(exp).size() > 0) {
    return false;
  }
  
  if (si::querySubTree<SgCompoundAssignOp>(exp).size() > 0) {
    return false;
  }
  
  if (si::querySubTree<SgPlusPlusOp>(exp).size() > 0) {
    return false;
  }
  
  if (si::querySubTree<SgMinusMinusOp>(exp).size() > 0) {
    return false;
  }

  LOG_DEBUG() << "Safe\n";
  return true;
}

static bool EliminateDeadCode(SgExprStatement *stmt) {
  if (IsSafeToEliminate(stmt->get_expression())) {
    LOG_DEBUG() << "Eliminating " << stmt->unparseToString() << "\n";
    //si::removeStatement(stmt);
    // NOTE: To preserve preprocessing directives
    if (stmt->get_attachedPreprocessingInfoPtr()) {
      si::replaceStatement(stmt, sb::buildNullStatement(), true);
    } else {
      si::removeStatement(stmt, true);
    }
    return true;
  }
  return false;
}

static bool EliminateDeadCode(SgVariableDeclaration *vdecl) {
  SgInitializedNamePtrList &variables = vdecl->get_variables();
  FOREACH (it, variables.begin(), variables.end()) {
    SgInitializedName *in = *it;
    vector<SgVarRefExp*> vrefs =
        si::querySubTree<SgVarRefExp>(si::getScope(vdecl));
    FOREACH (vrefs_it, vrefs.begin(), vrefs.end()) {
      SgVarRefExp *vref = *vrefs_it;
      if (in == vref->get_symbol()->get_declaration()) {
        SgNode *parent = vref->get_parent();
        if (isSgAssignOp(parent) &&
            isSgAssignOp(parent)->get_lhs_operand() == vref) {
          continue;
        }
        LOG_DEBUG() << "Reference to " << in->unparseToString()
                    << " found; not eliminated\n";
        return false;
      }
    }
  }
  bool vdecl_removed = false;
  FOREACH (it, variables.begin(), variables.end()) {
    SgInitializedName *in = *it;
    vector<SgExpression*> src_exprs;
    GetVariableSrc(in, src_exprs);
    FOREACH (src_exprs_it, src_exprs.begin(), src_exprs.end()) {
      SgExpression *src_expr = *src_exprs_it;
      vdecl_removed |=
          vdecl == si::getEnclosingStatement(src_expr);
      SgStatement *enclosing_stmt =
          si::getEnclosingStatement(src_expr);
      LOG_DEBUG() << "Replacing: "
                  << enclosing_stmt->unparseToString() << "\n";
      si::replaceStatement(
          enclosing_stmt,
          sb::buildExprStatement(si::copyExpression(src_expr)),
          true);
    }
  }
  if (!vdecl_removed) {
    LOG_DEBUG() << "Eliminating " << vdecl->unparseToString() << "\n";
    si::removeStatement(vdecl);
  }
  return true;
}

bool EliminateDeadCode(SgStatement *stmt) {
  bool changed = false;
  bool changed_once = false;
  do {
    changed = false;
    if (!stmt) break;    
    vector<SgStatement*> stmts =  si::querySubTree<SgStatement>(stmt);
    FOREACH (it, stmts.begin(), stmts.end()) {
      SgStatement *st = *it;
      //LOG_DEBUG() << "stmt: " << st->unparseToString() <<"\n";
      if (isSgVariableDeclaration(st)) {
        if (EliminateDeadCode(isSgVariableDeclaration(st))) {
          changed = true;
          if (st == stmt) stmt = NULL;
        }
      } else if (isSgExprStatement(st)) {
        SgNode *parent = st->get_parent();
        if (isSgForStatement(parent) || isSgIfStmt(parent))
          continue;
        if (EliminateDeadCode(isSgExprStatement(st))) {
          changed = true;
          if (st == stmt) stmt = NULL;
        }
      }
    }
    changed_once |= changed;
  } while (changed);
  return changed_once;
}

SgExpression *GetDeterministicDefinition(SgInitializedName *var) {
  vector<SgExpression*> var_srcs;
  GetVariableSrc(var, var_srcs);
  if (var_srcs.size() == 1) {
    LOG_DEBUG() << "Var, " << var->unparseToString()
                << " has a deterministic source, "
                << var_srcs[0]->unparseToString() << "\n";
    return var_srcs[0];
  } else {
    return NULL;
  }
}

} // namespace optimizer
} // namespace translator
} // namespace physis
