// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/def_analysis.h"

#include <memory>
#include <boost/foreach.hpp>

using std::auto_ptr;

namespace si = SageInterface;

namespace physis {
namespace translator {

// add def if not contained and return true; otherwise false is
// returned
static bool register_def(const SgInitializedName *var,
                         SgExpression *def,
                         DefMap &def_map) {
  DefMap::iterator it = def_map.find(var);
  if (it == def_map.end()) {
    SgExpressionPtrList el;
    el.push_back(def);
    def_map.insert(make_pair(var, el));
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
                       DefMap &def_map) {
  DefMap::iterator it = def_map.find(src_var);
  if (it == def_map.end()) return false;
  SgExpressionPtrList &srcDefs = it->second;
  // NOTE: Using sets as substrate of definitions would be more
  // efficient
  bool changed = false;
  for (SgExpressionPtrList::iterator it = srcDefs.begin(),
           end = srcDefs.end(); it != end; it++) {
    changed |= register_def(dst_var, *it, def_map);
  }
  return changed;
}

static bool HandleAssign(SgInitializedName *in,
                         SgExpression *rhs,
                         DefMap &def_map) {
  if (isSgVarRefExp(rhs)) {
    SgInitializedName *rhsName =
        si::convertRefToInitializedName(isSgVarRefExp(rhs));
    return merge_defs(in, rhsName, def_map);
  } else {
    return register_def(in, rhs, def_map);
  }
}
                         

static bool handleVarDecl(SgInitializedName *in,
                          DefMap &def_map) {
  SgInitializer *initializer = in->get_initializer();

  if (!initializer) {
    LOG_DEBUG() << "No definition\n";
    return register_def(in, NULL, def_map);
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

  return HandleAssign(in, rhs, def_map);
}

static bool IsRelevant(SgType *type, const vector<SgType*> &relevantTypes) {
  bool relevant = false;
  if (isSgArrayType(type)) {
    SgType *base_type = isSgArrayType(type)->get_base_type();
    relevant = std::find(relevantTypes.begin(), relevantTypes.end(),
                         base_type) != relevantTypes.end();
    if (relevant) {
      LOG_DEBUG() << "Relevant base type found:"
                  << base_type->unparseToString() <<"\n";
    }
  }
  if (!relevant) {
    relevant = std::find(relevantTypes.begin(), relevantTypes.end(),
                         type) != relevantTypes.end();
    if (relevant) {
      LOG_DEBUG() << "Relevant type found:"
                  << type->unparseToString() <<"\n";
    }
  }
  return relevant;
}

auto_ptr<DefMap>
findDefinitions(SgNode *topLevelNode, const vector<SgType*> &relevantTypes) {
  DefMap *def_map = new DefMap();

  vector<SgVariableDeclaration*> vars =
      si::querySubTree<SgVariableDeclaration>(topLevelNode);
  vector<SgAssignOp*> asns =
      si::querySubTree<SgAssignOp>(topLevelNode);
  bool changed = true;
  while (changed) {
    changed = false;

    BOOST_FOREACH(SgVariableDeclaration *vde, vars) {
      BOOST_FOREACH (SgInitializedName *in, vde->get_variables()) {
        SgType *type = in->get_type();
        if (IsRelevant(type, relevantTypes)) {
          changed |= handleVarDecl(in, *def_map);
        }
      }
    }

    BOOST_FOREACH(SgAssignOp *aop, asns) {
      assert(aop);
      SgType *type = aop->get_type();
      if (!IsRelevant(type, relevantTypes)) continue;
      SgVarRefExp *lhs = isSgVarRefExp(aop->get_lhs_operand());
      assert(lhs);
      SgInitializedName *lhsIn = si::convertRefToInitializedName(lhs);
      assert(lhsIn);
      SgExpression *rhs = aop->get_rhs_operand();
      changed |= HandleAssign(lhsIn, rhs, *def_map);
    }
  }

  return auto_ptr<DefMap>(def_map);
}
} // namespace translator
} // namespace physis 
