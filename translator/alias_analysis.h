// Licensed under the BSD license. See LICENSE.txt for more details.

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
  virtual std::ostream &Print(std::ostream &os) const {
    return os << "[" << getIndex() << "]";
  }
  int getIndex() const {
    return index;
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasGraphNode &node) {
  return node.Print(os);
}


typedef set<AliasGraphNode*> AliasNodeSet;
class AliasVarNode;
typedef set<AliasVarNode*> AliasVarSet;

class AliasVarNode: public AliasGraphNode {
  friend class AliasGraph;
  SgInitializedName *v;
  AliasNodeSet origins;
  AliasVarSet aliases;
  bool AddOrigin(AliasGraphNode *origin);
  bool AddAlias(AliasVarNode *aliasee);
  explicit AliasVarNode(SgInitializedName *v): v(v) {}
  virtual ~AliasVarNode() {}
 public:
  int getNumOrigins() const {
    return origins.size();
  }
  int getNumAliases() const {
    return aliases.size();
  }
  virtual std::ostream &Print(std::ostream &os) const;
  AliasGraphNode *FindRoot();
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasVarNode &node) {
  return node.Print(os);
}


class AliasFuncParamNode: public AliasGraphNode {
  AliasFuncParamNode() {}
  virtual ~AliasFuncParamNode() {}
 public:
  static AliasFuncParamNode *getInstance() {
    static AliasFuncParamNode *obj = new AliasFuncParamNode();
    return obj;
  }
  virtual std::ostream &Print(std::ostream &os) const {
    AliasGraphNode::Print(os);
    return os << "<FuncParam>";
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasFuncParamNode &node) {
  return node.Print(os);
}


class AliasNullInitNode: public AliasGraphNode {
  AliasNullInitNode() {}
  virtual ~AliasNullInitNode() {}
 public:
  static AliasNullInitNode *getInstance() {
    static AliasNullInitNode *null = new AliasNullInitNode();
    return null;
  }
  virtual std::ostream &Print(std::ostream &os) const {
    AliasGraphNode::Print(os);
    return os << "<NULL>";
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasNullInitNode &node) {
  return node.Print(os);
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
  virtual std::ostream &Print(std::ostream &os) const {
    AliasGraphNode::Print(os);
    return os << "<UNKNOWN>";
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const AliasUnknownOriginNode &node) {
  return node.Print(os);
}

// Intra-procedural alias analysis
class AliasGraph {
 public:
  typedef map<SgInitializedName*, AliasVarNode*> AliasVarMap;
  SgNode *topLevelNode;
  AliasVarMap vars;

  AliasGraph(SgNode *topLevelNode,
             const SgTypePtrList &relevantTypes)
      : topLevelNode(topLevelNode) {
    assert(topLevelNode);
    Build(relevantTypes);
  }
  bool HasMultipleOriginVar() const;
  bool HasNullInitVar() const;
  std::ostream &Print(std::ostream &os) const;
  SgInitializedName *FindOriginalVar(SgInitializedName *var);
 protected:
  AliasVarNode *CreateOrFindAliasVar(SgInitializedName *var);
  AliasVarNode *FindAliasVar(SgInitializedName *var);
  void Build(const SgTypePtrList &relevantTypes);
  bool HandleVarDecl(SgInitializedName *in);
  bool HandleVarAssignment(SgExpression *lhs,
                           SgExpression *rhs);
  bool AddAlias(AliasVarNode *alias, AliasGraphNode *origin);
};

inline std::ostream &operator<<(std::ostream &os, const AliasGraph &ag) {
  return ag.Print(os);
}

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_ALIAS_ANALYSIS_H_ */
