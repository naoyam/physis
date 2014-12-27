// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/optimizer/optimization_common.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {
namespace optimizer {

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
