// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "translator/ast_processing.h"

#include <boost/foreach.hpp>

#include "DefUseAnalysis.h"

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {
namespace rose_util {

static bool IsAssignedOnce(SgInitializedName *var,
                           SgNode *node,
                           DefUseAnalysis *dfa) {
  vector<SgVarRefExp*> vref = si::querySubTree<SgVarRefExp>(node);
  SgNode *assignement = NULL;
  FOREACH (it, vref.begin(), vref.end()) {
    SgVarRefExp *vref= *it;
    vector<SgNode*> defs = dfa->getDefFor(
        si::getEnclosingStatement(vref), var);
    if (defs.size() == 0) {
      continue;
    }
    if (assignement == NULL) {
      assignement = defs[0];
    }
    FOREACH (dit, defs.begin(), defs.end()) {
      if (assignement != *dit) return false;
    }
  }
  return true;
}

static void ReplaceVar(SgInitializedName *x,
                       SgExpression *y,
                       SgNode *node) {
  vector<SgVarRefExp*> vref = si::querySubTree<SgVarRefExp>(node);
  FOREACH (vref_it, vref.begin(), vref.end()) {
    SgVarRefExp *v = *vref_it;
    if (v->get_symbol()->get_declaration() == x) {
      //LOG_DEBUG() << "Replacing use of " << x->unparseToString() << "\n";
      //LOG_DEBUG() << "STMT: " << si::getEnclosingStatement(v)->unparseToString() << "\n";
      //NOTE: AST consistency check fails if v is deleted (i.e.,
      //false is given). Why?
      si::replaceExpression(v, si::copyExpression(y), true);
    }
  }
  LOG_DEBUG() << "Removing " << x->get_declaration()->unparseToString() << "\n";
  si::removeStatement(x->get_declaration());
  si::clearUnusedVariableSymbols();
}

int RemoveRedundantVariableCopy(SgNode *top) {
  DefUseAnalysis *dfa = new DefUseAnalysis(si::getProject());
  dfa->run();
  int num_removed_vars = 0;
  vector<SgInitializedName*> vars =
      si::querySubTree<SgInitializedName>(top);
  FOREACH (vit, vars.begin(), vars.end()) {
    SgInitializedName *var = *vit;
    SgScopeStatement *scope = si::getScope(var);
    SgAssignInitializer *init = isSgAssignInitializer(var->get_initializer());
    if (init == NULL) continue;
    SgExpression *rhs = init->get_operand();
    SgVarRefExp *rhs_vref = isSgVarRefExp(rhs);
    if (rhs_vref == NULL) {
      if (isSgUnaryOp(rhs)) {
        rhs_vref = isSgVarRefExp(isSgUnaryOp(rhs)->get_operand());
        if (rhs_vref == NULL) continue;
      } else {
        continue;
      }
    }
    LOG_DEBUG() << "Variable " << var->unparseToString()
                << " is assigned with " << rhs->unparseToString() << "\n";
    SgInitializedName *rhs_var = isSgInitializedName(
        rhs_vref->get_symbol()->get_declaration());
    if (!IsAssignedOnce(var, scope, dfa)) continue;
    if (!IsAssignedOnce(rhs_var, scope, dfa)) continue;
    LOG_DEBUG() << "This variable copy can be safely eliminated.\n";
    ReplaceVar(var, rhs, scope);
#if 0
    LOG_DEBUG() << "AFTER\n:"
                << top->unparseToString() << "\n";
#endif      
    ++num_removed_vars;
  }
  return num_removed_vars;
}

static bool DefinedInSource(SgNode *node) {
  //LOG_DEBUG() << "node: " << node->unparseToString() << "\n";
  Sg_File_Info *finfo = node->get_file_info();
  if (finfo->isTransformation()) return true;
  string fname = finfo->get_filenameString();
  //LOG_DEBUG() << "file: " << fname << "\n";
  SgProject *proj = si::getProject();
  BOOST_FOREACH(SgFile *f, proj->get_fileList()) {
    if (finfo->isSameFile(f)) {
      //LOG_DEBUG() << "Defined in input source\n";
      return true;
    }
  }
  return false;
}

int RemoveUnusedFunction(SgNode *scope) {
  int num_removed_funcs = 0;
  BOOST_FOREACH(SgFunctionDeclaration *fdecl,
                si::querySubTree<SgFunctionDeclaration>(scope)) {
    // Don't remove non-static functions
    if (!si::isStatic(fdecl)) continue;
    if (!DefinedInSource(fdecl)) continue;
    SgSymbol *fs = fdecl->search_for_symbol_from_symbol_table();
    bool used = false;
    BOOST_FOREACH(SgFunctionRefExp *fref,
                  si::querySubTree<SgFunctionRefExp>(scope)) {
      if (fs == fref->get_symbol()) {
        used = true;
        break;
      }
    }
    if (used) continue;

    LOG_DEBUG() << "Removing " << fs->get_name() << "\n";
    
    // si::removeStatement does not automatically relocate
    // preprocessing info even if explicitly requested with the second
    // argument. Move it beforehand.
    si::movePreprocessingInfo(fdecl, si::getNextStatement(fdecl),
                              PreprocessingInfo::undef,
                              PreprocessingInfo::undef, true);
    si::removeStatement(fdecl, true);
    ++num_removed_funcs;
  }
  
  return num_removed_funcs;
}

}  // namespace rose_util
}  // namespace translator
}  // namespace physis
