// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "translator/ast_processing.h"

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
                       SgInitializedName *y,
                       SgNode *node,
                       DefUseAnalysis *dfa) {
  vector<SgVarRefExp*> vref = si::querySubTree<SgVarRefExp>(node);
  FOREACH (vref_it, vref.begin(), vref.end()) {
    SgVarRefExp *v = *vref_it;
    if (dfa->getUseFor(v, x).size() > 0) {
      //LOG_DEBUG() << "Replacing use of " << x->unparseToString() << "\n";
      si::replaceExpression(v, sb::buildVarRefExp(y));
    }
  }
  si::removeStatement(x->get_declaration());
  si::clearUnusedVariableSymbols();
}

int RemoveRedundantVariableCopy(SgNode *top) {
  vector<SgFunctionDefinition*> fdefs =
      si::querySubTree<SgFunctionDefinition>(top);
  DefUseAnalysis *dfa = new DefUseAnalysis(si::getProject());
  dfa->run();
  int num_removed_vars = 0;
  FOREACH (it, fdefs.begin(), fdefs.end()) {
    SgFunctionDefinition *fdef = *it;
    vector<SgInitializedName*> vars =
      si::querySubTree<SgInitializedName>(fdef);
    FOREACH (vit, vars.begin(), vars.end()) {
      SgInitializedName *var = *vit;
      SgAssignInitializer *init = isSgAssignInitializer(var->get_initializer());
      if (init == NULL) continue;
      SgVarRefExp *rhs = isSgVarRefExp(init->get_operand());
      if (rhs == NULL) continue;
      //LOG_DEBUG() << "Variable " << var->unparseToString()
      //<< " is assigned with " << rhs->unparseToString() << "\n";
      SgInitializedName *rhs_var = isSgInitializedName(
          rhs->get_symbol()->get_declaration());
      if (!IsAssignedOnce(var, fdef, dfa)) continue;
      if (!IsAssignedOnce(rhs_var, fdef, dfa)) continue;
      //LOG_DEBUG() << "This variable copy can be safely eliminated.\n";
      ReplaceVar(var, rhs_var, fdef, dfa);
#if 0      
      LOG_DEBUG() << "AFTER\n:"
                  << fdef->unparseToString() << "\n";
#endif      
      ++num_removed_vars;
    }
  }
  return num_removed_vars;
}


}  // namespace rose_util
}  // namespace translator
}  // namespace physis
