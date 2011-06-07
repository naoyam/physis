// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_ALIAS_ANALYSIS_H_
#define PHYSIS_TRANSLATOR_ALIAS_ANALYSIS_H_

#include <map>
#include <set>

#include "translator/translator_common.h"

using std::map;
using std::set;

namespace physis {
namespace translator {

class AliasGraphNode {
  int index;
  static Counter c;
 public:
  AliasGraphNode(): index(c.next()) {}
  virtual ~AliasGraphNode() {}
  virtual std::ostream &print(std::ostream &os) const {
    return os << "[" << getIndex() << "]";
  }
  int getIndex() const {
    return index;
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasGraphNode &node) {
  return node.print(os);
}


typedef set<AliasGraphNode*> AliasNodeSet;
class AliasVarNode;
typedef set<AliasVarNode*> AliasVarSet;

class AliasVarNode: public AliasGraphNode {
  friend class AliasGraph;
  SgInitializedName *v;
  AliasNodeSet origins;
  AliasVarSet aliases;
  bool addOrigin(AliasGraphNode *origin);
  bool addAlias(AliasVarNode *aliasee);
  explicit AliasVarNode(SgInitializedName *v): v(v) {}
  virtual ~AliasVarNode() {}
 public:
  int getNumOrigins() const {
    return origins.size();
  }
  int getNumAliases() const {
    return aliases.size();
  }
  virtual std::ostream &print(std::ostream &os) const;
  AliasGraphNode *findRoot();
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasVarNode &node) {
  return node.print(os);
}


class AliasFuncParamNode: public AliasGraphNode {
  AliasFuncParamNode() {}
  virtual ~AliasFuncParamNode() {}
 public:
  static AliasFuncParamNode *getInstance() {
    static AliasFuncParamNode *obj = new AliasFuncParamNode();
    return obj;
  }
  virtual std::ostream &print(std::ostream &os) const {
    AliasGraphNode::print(os);
    return os << "<FuncParam>";
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasFuncParamNode &node) {
  return node.print(os);
}


class AliasNullInitNode: public AliasGraphNode {
  AliasNullInitNode() {}
  virtual ~AliasNullInitNode() {}
 public:
  static AliasNullInitNode *getInstance() {
    static AliasNullInitNode *null = new AliasNullInitNode();
    return null;
  }
  virtual std::ostream &print(std::ostream &os) const {
    AliasGraphNode::print(os);
    return os << "<NULL>";
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasNullInitNode &node) {
  return node.print(os);
}


class AliasUnknownOriginNode: public AliasGraphNode {
  AliasUnknownOriginNode() {}
  virtual ~AliasUnknownOriginNode() {}
 public:
  static AliasUnknownOriginNode *getInstance() {
    static AliasUnknownOriginNode *un
        = new AliasUnknownOriginNode();
    return un;
  }
  virtual std::ostream &print(std::ostream &os) const {
    AliasGraphNode::print(os);
    return os << "<UNKNOWN>";
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasUnknownOriginNode &node) {
  return node.print(os);
}


class AliasGraph {
 public:
  typedef map<SgInitializedName*, AliasVarNode*> AliasVarMap;
  SgNode *topLevelNode;
  AliasVarMap vars;

  AliasGraph(SgNode *topLevelNode,
             const SgTypePtrList &relevantTypes)
      : topLevelNode(topLevelNode) {
    assert(topLevelNode);
    build(relevantTypes);
  }
  bool hasMultipleOriginVar() const;
  bool hasNullInitVar() const;
  std::ostream &print(std::ostream &os) const;
  SgInitializedName *findOriginalVar(SgInitializedName *var);
 protected:
  AliasVarNode *createOrFindAliasVar(SgInitializedName *var);
  AliasVarNode *findAliasVar(SgInitializedName *var);
  void build(const SgTypePtrList &relevantTypes);
  bool handleVarDecl(SgInitializedName *in);
  bool handleVarAssignment(SgExpression *lhs,
                           SgExpression *rhs);
  bool addAlias(AliasVarNode *alias, AliasGraphNode *origin);
};

inline std::ostream &operator<<(std::ostream &os, const AliasGraph &ag) {
  return ag.print(os);
}

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_ALIAS_ANALYSIS_H_ */
