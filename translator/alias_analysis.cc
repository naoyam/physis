// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/alias_analysis.h"

#include <boost/foreach.hpp>
#include "translator/rose_util.h"


namespace si = SageInterface;

namespace physis {
namespace translator {

Counter AliasGraphNode::c;

bool AliasVarNode::AddAlias(AliasVarNode *alias) {
  return aliases.insert(alias).second;
}

bool AliasVarNode::AddOrigin(AliasGraphNode *origin) {
  return origins.insert(origin).second;
}

std::ostream &AliasVarNode::Print(std::ostream &os) const {
  AliasGraphNode::Print(os);
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

bool AliasGraph::AddAlias(AliasVarNode *alias, AliasGraphNode *origin) {
  bool changed = false;
  changed |= alias->AddOrigin(origin);
  AliasVarNode *v = dynamic_cast<AliasVarNode*>(origin);
  if (v) {
    changed |= v->AddAlias(alias);
  }
  return changed;
}

AliasVarNode *AliasGraph::FindAliasVar(SgInitializedName *var) {
  AliasVarMap::iterator it = vars.find(var);
  if (it == vars.end()) {
    return NULL;
  } else {
    return it->second;
  }
}

AliasVarNode *AliasGraph::CreateOrFindAliasVar(SgInitializedName *var) {
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

bool AliasGraph::HandleVarDecl(SgInitializedName *in) {
  bool changed = false;
  SgInitializer *initializer = in->get_initializer();
  AliasVarNode *av = CreateOrFindAliasVar(in);

  if (!initializer) {
    LOG_DEBUG() << "No init found: "
                << in->unparseToString()
                << " at line " << in->get_file_info()->get_line()
                << "\n";
    if (rose_util::IsFuncParam(in)) {
      LOG_DEBUG() << "Func param\n";
      changed |= av->AddOrigin(AliasFuncParamNode::getInstance());
    } else {
      LOG_DEBUG() << "Null init\n";
      changed |= av->AddOrigin(AliasNullInitNode::getInstance());
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
        CreateOrFindAliasVar(si::convertRefToInitializedName(isSgVarRefExp(rhs)));
    return av->AddOrigin(rhsV);
  } else {
    return av->AddOrigin(AliasUnknownOriginNode::getInstance());
  }
}

bool AliasGraph::HandleVarAssignment(SgExpression *lhs, SgExpression *rhs) {
  assert(isSgVarRefExp(lhs));
  assert(isSgVarRefExp(rhs));
  SgInitializedName *lhsIn = si::convertRefToInitializedName(isSgVarRefExp(lhs));
  assert(lhsIn);
  SgInitializedName *rhsIn = si::convertRefToInitializedName(isSgVarRefExp(rhs));
  assert(rhsIn);

  return AddAlias(CreateOrFindAliasVar(lhsIn),
                  CreateOrFindAliasVar(rhsIn));
}


void AliasGraph::Build(const SgTypePtrList &relevantTypes) {
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
      changed |= HandleVarDecl(in);
    }

    BOOST_FOREACH(SgAssignOp *aop, asns) {
      SgType *type = aop->get_type();
      if (std::find(relevantTypes.begin(), relevantTypes.end(),
                    type) == relevantTypes.end())
        continue;
      changed |= HandleVarAssignment(aop->get_lhs_operand(),
                                     aop->get_rhs_operand());
    }
  }
}


bool AliasGraph::HasMultipleOriginVar() const {
  FOREACH(vit, vars.begin(), vars.end()) {
    AliasVarNode *v = vit->second;
    if (v->getNumOrigins() > 1) return true;
  }
  return false;
}

bool AliasGraph::HasNullInitVar() const {
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


std::ostream &AliasGraph::Print(std::ostream &os) const {
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

AliasGraphNode *AliasVarNode::FindRoot() {
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

SgInitializedName *AliasGraph::FindOriginalVar(SgInitializedName *var) {
  AliasVarNode *vn = FindAliasVar(var);
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

