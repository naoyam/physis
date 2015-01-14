// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/rose_traversal.h"

namespace physis {
namespace translator {
namespace rose_util {

void RoseASTTraversal::VisitInternal(SgNode *node) {
  if (isSgForStatement(node)) {
    Visit(isSgForStatement(node));
  } else if (isSgTypedefDeclaration(node)) {
    Visit(isSgTypedefDeclaration(node));
  } else if (isSgClassDeclaration(node)) {
    Visit(isSgClassDeclaration(node));    
  } else if (isSgFunctionCallExp(node)) {
    Visit(isSgFunctionCallExp(node));
  } else if (isSgDotExp(node)) {
    Visit(isSgDotExp(node));
  } else if (isSgPntrArrRefExp(node)) {
    Visit(isSgPntrArrRefExp(node));
  } else if (isSgExpression(node)) {
    Visit(isSgExpression(node));
  } else if (isSgFunctionDeclaration(node)) {
    Visit(isSgFunctionDeclaration(node));
  } else {
    Visit(node);
  }
}

void RoseASTTraversal::traverseTopDown(SgNode *node) {
  LOG_DEBUG() << "Visiting " << node->class_name()
#if 0
              << ": " << node->unparseToString()
#endif
              << "\n";

  if (isSgStatement(node)) {
    stmtStack.push_back(isSgStatement(node));
  }
  if (isSgFunctionDeclaration(node)) {
    funcStack.push_back(isSgFunctionDeclaration(node));
  }
  if (isSgScopeStatement(node)) {
    scopeStack.push_back(isSgScopeStatement(node));
  }

  VisitInternal(node);

  if (doesSkipChildren()) {
    setSkipChildren(false);
    return;
  }

  // New nodes may be added while traversing children, but
  // they should not be visited. So just copies the
  // references to children to a vector first.
  vector<SgNode*> children;
  for (unsigned i = 0; i < node->get_numberOfTraversalSuccessors();
       i++) {
    SgNode *child = node->get_traversalSuccessorByIndex(i);
    if (!child) {
      // Why child can be NULL?
      // LOG_WARNING() << "Ignoring NULL child\n";
      continue;
    }
    children.push_back(child);
  }
  FOREACH(it, children.begin(), children.end()) {
    traverseTopDown(*it);
  }

  if (isSgStatement(node)) {
    stmtStack.pop_back();
  }
  if (isSgFunctionDeclaration(node)) {
    funcStack.pop_back();
  }
  if (isSgScopeStatement(node)) {
    scopeStack.pop_back();
  }
}


void RoseASTTraversal::traverseBottomUp(SgNode *node) {
  // OG_DEBUG() << "traverseBottomUp\n";
  // New nodes may be added while traversing children, but
  // they should not be visited. So just copies the
  // references to children to a vector first.
  vector<SgNode*> children;
  for (unsigned i = 0; i < node->get_numberOfTraversalSuccessors();
       i++) {
    SgNode *child = node->get_traversalSuccessorByIndex(i);
    if (!child) {
      // Why child can be NULL?
      // LOG_WARNING() << "Ignoring NULL child\n";
      continue;
    }
    children.push_back(child);
  }
  FOREACH(it, children.begin(), children.end()) {
    traverseBottomUp(*it);
  }

  // LOG_DEBUG() << "Visiting " << node->class_name() << "\n";

  VisitInternal(node);
}

} // namespace rose_util
} // namespace translator
} // namespace physis
