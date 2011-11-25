// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_ROSE_TRAVERSAL_H_
#define PHYSIS_TRANSLATOR_ROSE_TRAVERSAL_H_

#include "rose.h"
#include "physis/physis_util.h"
#include "translator/physis_exception.h"

namespace physis {
namespace translator {
namespace rose_util {

class RoseASTTraversal {
 public:
  RoseASTTraversal(): skipChildren(false) {}
  virtual ~RoseASTTraversal() {}
  virtual void traverseBottomUp(SgNode *node);
  virtual void traverseTopDown(SgNode *node);

 private:
  void VisitInternal(SgNode *node);
  
 protected:
  list<SgStatement*> stmtStack;
  list<SgFunctionDeclaration*> funcStack;
  list<SgScopeStatement*> scopeStack;
  virtual void Visit(SgForStatement *node) {}
  virtual void Visit(SgTypedefDeclaration *node) {}
  virtual void Visit(SgFunctionCallExp *node) {}
  virtual void Visit(SgClassDeclaration *node) {}
  virtual void Visit(SgExpression *node) {}
  virtual void Visit(SgFunctionDeclaration *node) {}
  virtual void Visit(SgNode *node) {}
  bool skipChildren;
  virtual void setSkipChildren(bool s = true) {
    skipChildren = s;
  }
  virtual bool doesSkipChildren() {
    return skipChildren;
  }
  SgStatement *getContainingStatement(SgNode *node) const {
    node = node->get_parent();
    while (!isSgStatement(node)) {
      if (!node) throw PhysisException("Statment not found");
      node = node->get_parent();
    }
    return isSgStatement(node);
  }

  SgFunctionDeclaration *getContainingFunction(SgNode *node) const {
    node = node->get_parent();
    while (!isSgFunctionDeclaration(node)) {
      if (!node) throw PhysisException("Statment not found");
      node = node->get_parent();
    }
    return isSgFunctionDeclaration(node);
  }

  SgScopeStatement *getContainingScopeStatement(SgNode *node)
      const {
    node = node->get_parent();
    while (!isSgScopeStatement(node)) {
      if (!node) throw PhysisException("Scope not found");
      node = node->get_parent();
    }
    return isSgScopeStatement(node);
  }
};

} // namespace rose_util
} // namespace translator
} // namespace physis


#endif /* ROSETRAVERSAL_H_ */
