// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/alias_analysis.h"

#include <boost/foreach.hpp>
#include "translator/rose_util.h"


namespace si = SageInterface;

namespace physis {
namespace translator {

Counter AliasGraphNode::c;

bool AliasVarNode::addAlias(AliasVarNode *alias) {
  return aliases.insert(alias).second;
}

bool AliasVarNode::addOrigin(AliasGraphNode *origin) {
  return origins.insert(origin).second;
}

std::ostream &AliasVarNode::print(std::ostream &os) const {
  AliasGraphNode::print(os);
  StringJoin originStr;
  FOREACH(oit, origins.begin(), origins.end()) {
    originStr << (*oit)->getIndex();
    // riginStr << *oit;
  }
  StringJoin aliasStr;
  FOREACH(oit, aliases.begin(), aliases.end()) {
    aliasStr << (*oit)->getIndex();
    // liasStr << *oit;
  }
  os << "Var (" << this << "): " << v->unparseToString()
     << ", origins: " << originStr.get()
     << ", aliases: " << aliasStr.get()
     << ">";
  return os;
}

bool AliasGraph::addAlias(AliasVarNode *alias, AliasGraphNode *origin) {
  bool changed = false;
  changed |= alias->addOrigin(origin);
  AliasVarNode *v = dynamic_cast<AliasVarNode*>(origin);
  if (v) {
    changed |= v->addAlias(alias);
  }
  return changed;
}

AliasVarNode *AliasGraph::findAliasVar(SgInitializedName *var) {
  AliasVarMap::iterator it = vars.find(var);
  if (it == vars.end()) {
    return NULL;
  } else {
    return it->second;
  }
}

AliasVarNode *AliasGraph::createOrFindAliasVar(SgInitializedName *var) {
  AliasVarMap::iterator it = vars.find(var);
  AliasVarNode *av;
  if (it == vars.end()) {
    av = new AliasVarNode(var);
    vars.insert(std::make_pair(var, av));
  } else {
    av = it->second;
  }
  return av;
}

bool AliasGraph::handleVarDecl(SgInitializedName *in) {
  bool changed = false;
  SgInitializer *initializer = in->get_initializer();
  AliasVarNode *av = createOrFindAliasVar(in);

  if (!initializer) {
    LOG_DEBUG() << "No init found: "
                << in->unparseToString()
                << " at line " << in->get_file_info()->get_line()
                << "\n";
    if (rose_util::IsFuncParam(in)) {
      LOG_DEBUG() << "Func param\n";
      changed |= av->addOrigin(AliasFuncParamNode::getInstance());
    } else {
      LOG_DEBUG() << "Null init\n";
      changed |= av->addOrigin(AliasNullInitNode::getInstance());
    }
    return changed;
  }

  // node is assumed to be of type SgAssignInitializer
  SgExpression *rhs = NULL;
  if (isSgAssignInitializer(initializer)) {
    rhs = isSgAssignInitializer(initializer)->get_operand();
  } else {
    LOG_ERROR() << "Unknown node: "
                << initializer->class_name() << "\n";
  }

  if (isSgVarRefExp(rhs)) {
    AliasVarNode *rhsV =
        createOrFindAliasVar(si::convertRefToInitializedName(isSgVarRefExp(rhs)));
    return av->addOrigin(rhsV);
  } else {
    return av->addOrigin(AliasUnknownOriginNode::getInstance());
  }
}

bool AliasGraph::handleVarAssignment(SgExpression *lhs, SgExpression *rhs) {
  assert(isSgVarRefExp(lhs));
  assert(isSgVarRefExp(rhs));
  SgInitializedName *lhsIn = si::convertRefToInitializedName(isSgVarRefExp(lhs));
  assert(lhsIn);
  SgInitializedName *rhsIn = si::convertRefToInitializedName(isSgVarRefExp(rhs));
  assert(rhsIn);

  return addAlias(createOrFindAliasVar(lhsIn),
                  createOrFindAliasVar(rhsIn));
}


void AliasGraph::build(const SgTypePtrList &relevantTypes) {
  vector<SgInitializedName*> vars =
      si::querySubTree<SgInitializedName>(topLevelNode);
  vector<SgAssignOp*> asns =
      si::querySubTree<SgAssignOp>(topLevelNode);
  bool changed = true;
  while (changed) {
    changed = false;

    BOOST_FOREACH(SgInitializedName *in, vars) {
      SgType *type = in->get_type();
      if (std::find(relevantTypes.begin(), relevantTypes.end(),
                    type) == relevantTypes.end())
        continue;
      changed |= handleVarDecl(in);
    }

    BOOST_FOREACH(SgAssignOp *aop, asns) {
      SgType *type = aop->get_type();
      if (std::find(relevantTypes.begin(), relevantTypes.end(),
                    type) == relevantTypes.end())
        continue;
      changed |= handleVarAssignment(aop->get_lhs_operand(),
                                     aop->get_rhs_operand());
    }
  }
}


bool AliasGraph::hasMultipleOriginVar() const {
  FOREACH(vit, vars.begin(), vars.end()) {
    AliasVarNode *v = vit->second;
    if (v->getNumOrigins() > 1) return true;
  }
  return false;
}

bool AliasGraph::hasNullInitVar() const {
  FOREACH(vit, vars.begin(), vars.end()) {
    AliasVarNode *v = vit->second;
    FOREACH(oit, v->origins.begin(), v->origins.end()) {
      AliasGraphNode *node = *oit;
      if (dynamic_cast<AliasNullInitNode*>(node))
        return true;
    }
  }
  return false;
}


std::ostream &AliasGraph::print(std::ostream &os) const {
  os << "Alias Graph\n";
  os << *AliasFuncParamNode::getInstance() << "\n";
  os << *AliasUnknownOriginNode::getInstance() << "\n";
  os << *AliasNullInitNode::getInstance() << "\n";
  FOREACH(nit, vars.begin(), vars.end()) {
    AliasGraphNode *node = nit->second;
    os << *node << "\n";
  }
  return os;
}

AliasGraphNode *AliasVarNode::findRoot() {
  AliasVarNode *node = this;
  AliasGraphNode *root = NULL;
  while (true) {
    assert(node->origins.size() == 1);
    root = *(node->origins.begin());
    if (dynamic_cast<AliasVarNode*>(root)) {
      node = dynamic_cast<AliasVarNode*>(root);
    } else {
      break;
    }
  }
  return root;
}

SgInitializedName *AliasGraph::findOriginalVar(SgInitializedName *var) {
  AliasVarNode *vn = findAliasVar(var);
  assert(vn);
  while (true) {
    assert(vn->origins.size() == 1);
    AliasGraphNode *origin = *(vn->origins.begin());
    if (dynamic_cast<AliasVarNode*>(origin)) {
      vn = dynamic_cast<AliasVarNode*>(origin);
    } else {
      break;
    }
  }
  return vn->v;
}

} // namespace translator
} // namespace physis

