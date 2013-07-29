// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/def_analysis.h"

#include <memory>

using std::auto_ptr;

namespace si = SageInterface;

namespace physis {
namespace translator {

#if 0
virtual const FunctionDecl *findCallee(CallExpr *ce) {
  const FunctionDecl *callee = NULL;
  if (UpdateFunction::isCallToUpdate(ce)) {
    // find kernel
    LOG_DEBUG() << "call to update\n";
    DeclRefExpr *kernelRef =
        removeIgnorableCasts(ce->getArg(0));
    assert(kernelRef);
    callee =  dyn_cast<FunctionDecl>(kernelRef->getDecl());
    assert(callee);
  } else {
    callee = dyn_cast<FunctionDecl>(ce->getCalleeDecl());
  }
  return callee;
}

virtual const ParmVarDecl *getParam(CallExpr *ce, unsigned i) {
  const FunctionDecl *callee = findCallee(ce);
  if (UpdateFunction::isCallToUpdate(ce)) {
    // find the dimensionality of this grid
    int d = getDimFromType(ce->getArg(i)->getType());
    LOG_DEBUG() << "call dim: " << d << "\n";
    i--;  // kernel
    i += d;  // indices (i, j, k)
  }
  return callee->getParamDecl(i);
}

virtual void VisitCallExpr(CallExpr *ce) {
  LOG_DEBUG() << "Visiting call\n";
  const FunctionDecl *callee = findCallee(ce);
  if (!callee) {
    // callee is a indirectly referenced function;
    // skipping
    return;
  }
  ENUMERATE(i, ait, ce->arg_begin(), ce->arg_end()) {
    Expr *arg = *ait;
    if (!isRelevant(arg)) continue;
    // propagetes analysis partial results
    LOG_DEBUG() << "propagating info to callee\n";
    ce->getCallee()->dump();

    assert(isa<DeclRefExpr>(arg));
    const VarDecl *argDecl
        = cast<VarDecl>
        (cast<DeclRefExpr>(arg)->getDecl());
    const ParmVarDecl *param = getParam(ce, i);
    merge_defs(param, argDecl);
  }
}
};
}

#endif

// add def if not contained and return true; otherwise false is
// returned
static bool register_def(const SgInitializedName *var,
                         SgExpression *def,
                         DefMap &defMap) {
  DefMap::iterator it = defMap.find(var);
  if (it == defMap.end()) {
    SgExpressionPtrList el;
    el.push_back(def);
    defMap.insert(make_pair(var, el));
    return true;
  } else {
    SgExpressionPtrList &el = it->second;
    if (!isContained(el, def)) {
      el.push_back(def);
      return true;
    } else {
      return false;
    }
  }
}

static bool merge_defs(const SgInitializedName *dst_var,
                       const SgInitializedName *src_var,
                       DefMap &defMap) {
  DefMap::iterator it = defMap.find(src_var);
  if (it == defMap.end()) return false;
  SgExpressionPtrList &srcDefs = it->second;
  // NOTE: Using sets as substrate of definitions would be more
  // efficient
  bool changed = false;
  for (SgExpressionPtrList::iterator it = srcDefs.begin(),
           end = srcDefs.end(); it != end; it++) {
    changed |= register_def(dst_var, *it, defMap);
  }
  return changed;
}


static bool handleVarDecl(SgInitializedName *in,
                          DefMap &defMap) {
  SgInitializer *initializer = in->get_initializer();

  if (!initializer) {
    LOG_DEBUG() << "No definition\n";
    return register_def(in, NULL, defMap);
  }

  // node is assumed to be of type SgAssignInitializer
  SgExpression *rhs = NULL;
  if (isSgAssignInitializer(initializer)) {
    rhs = isSgAssignInitializer(initializer)->get_operand();
  }else if (isSgAggregateInitializer(initializer)) {
    rhs = isSgAggregateInitializer(initializer)->get_initializers();
  } else {
    LOG_ERROR() << "Unknown node: "
                << initializer->class_name() << "\n";
  }

  if (isSgVarRefExp(rhs)) {
    SgInitializedName *rhsName =
        si::convertRefToInitializedName(isSgVarRefExp(rhs));
    return merge_defs(in, rhsName, defMap);
  } else {
    return register_def(in, rhs, defMap);
  }
}


auto_ptr<DefMap>
findDefinitions(SgNode *topLevelNode, const vector<SgType*> &relevantTypes) {
  DefMap *defMap = new DefMap();

  SgNodePtrList vars =
      NodeQuery::querySubTree(topLevelNode, V_SgInitializedName);
  SgNodePtrList asns =
      NodeQuery::querySubTree(topLevelNode, V_SgAssignOp);
  bool changed = true;
  while (changed) {
    changed = false;

    FOREACH(it, vars.begin(), vars.end()) {
      SgInitializedName *in = isSgInitializedName(*it);

      SgType *type = in->get_type();
      if (std::find(relevantTypes.begin(), relevantTypes.end(),
                    type) == relevantTypes.end())
        continue;

      changed |= handleVarDecl(in, *defMap);
    }

    FOREACH(it, asns.begin(), asns.end()) {
      SgAssignOp *aop = isSgAssignOp(*it);
      assert(aop);
      SgType *type = aop->get_type();
      if (std::find(relevantTypes.begin(), relevantTypes.end(),
                    type) == relevantTypes.end())
        continue;

      SgVarRefExp *lhs = isSgVarRefExp(aop->get_lhs_operand());
      assert(lhs);
      SgInitializedName *lhsIn = si::convertRefToInitializedName(lhs);
      assert(lhsIn);
      SgVarRefExp *rhs = isSgVarRefExp(aop->get_rhs_operand());
      assert(rhs);
      SgInitializedName *rhsIn = si::convertRefToInitializedName(rhs);
      assert(rhsIn);

      changed |= merge_defs(lhsIn, rhsIn, *defMap);
    }
  }

  return auto_ptr<DefMap>(defMap);
}
} // namespace translator
} // namespace physis 
