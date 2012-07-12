// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_ROSE_UTIL_H_
#define PHYSIS_TRANSLATOR_ROSE_UTIL_H_

#include "translator/translator_common.h"
#include "physis/internal_common.h"

namespace physis {
namespace translator {
namespace rose_util {

// This doesn't work.
// template <class T, class SgT, class TypeConverter>
// bool copyConstantFuncArgs(SgFunctionCallExp *call,
//                           vector<T> &constantHolder,
//                           TypeConverter tconv) {
//   SgExprListExp *args = call->get_args();
//   SageInterface::constantFolding(args);

//   SgExpressionPtrList &expl = args->get_expressions();

//   FOREACH(it, expl.begin(), expl.end()) {
//     SgExpression *arg = *it;
//     // gIntVal *v = isSgIntVal(arg);
//     SgT *v = tconv(arg);
//     if (!v) {
//       LOG_DEBUG() << "Non constant func arg: " << arg->unparseToString() << "\n";
//       return false;
//     }
//     constantHolder.push_back(v->get_value());
//   }

//   return true;
// }

template <class T>
bool copyConstantFuncArgs(SgFunctionCallExp *call,
                          vector<T> &constantHolder) {
  SgExprListExp *args = call->get_args();
  SageInterface::constantFolding(args);

  SgExpressionPtrList &expl = args->get_expressions();

  FOREACH(it, expl.begin(), expl.end()) {
    SgExpression *arg = *it;
    // gIntVal *v = isSgIntVal(arg);
    SgValueExp *arg_val = isSgValueExp(arg);
    if (!arg_val || !SageInterface::isStrictIntegerType(arg_val->get_type())) {
      LOG_DEBUG() << "Non constant func arg: " << arg->unparseToString() << "\n";
      return false;
    }
    T v = (T)SageInterface::getIntegerConstantValue(arg_val);
    constantHolder.push_back(v);
  }

  return true;
}

template <class T>
int copyConstantFuncArgs(SgExpressionPtrList::const_iterator it,
                         SgExpressionPtrList::const_iterator end,
                         vector<T> &constantHolder) {
  int num_constants = 0;
  while (it != end) {
    SgExpression *arg = *it;
    SgValueExp *arg_val = isSgValueExp(arg);
    if (!arg_val ||
        !SageInterface::isStrictIntegerType(arg_val->get_type())) {
      LOG_VERBOSE() << "Non constant func arg: "
                    << arg->unparseToString() << "\n";
      break;
    }
    T v = (T)SageInterface::getIntegerConstantValue(arg_val);
    constantHolder.push_back(v);
    ++it;
    ++num_constants;
  }

  return num_constants;
}


template <class T>
const string getName(const T &x) {
  return x->get_name().getString();
}

SgType *getType(SgNode *topLevelNode, const string &typeName);
// For debugging
void printAllTypeNames(SgNode *topLevelNode, std::ostream &os);
// recursively removes all casts
SgExpression *removeCasts(SgExpression *exp);
SgFunctionDeclaration *getFuncDeclFromFuncRef(SgExpression *refExp);
SgClassDefinition *getDefinition(SgClassType *t);
SgFunctionDeclaration *getContainingFunction(SgNode *node);
string getFuncName(SgFunctionRefExp *fref);
string getFuncName(SgFunctionCallExp *call);
SgExpression *copyExpr(const SgExpression *expr);
SgInitializedName *copySgNode(const SgInitializedName *expr);
SgFunctionSymbol *getFunctionSymbol(SgFunctionDeclaration *f);
SgFunctionRefExp *getFunctionRefExp(SgFunctionDeclaration *decl);
SgVarRefExp *buildFieldRefExp(SgClassDeclaration *decl, string name);
bool isFuncParam(SgInitializedName *in);
SgInitializedName *getInitializedName(SgVarRefExp *var);
string generateUniqueName(SgScopeStatement *scope = NULL,
                          const string &prefix = "__v");
void SetFunctionStatic(SgFunctionDeclaration *fdecl);
SgExpression *buildNULL(SgScopeStatement *global_scope);
SgVariableDeclaration *buildVarDecl(const string &name,
                                    SgType *type,
                                    SgExpression *val,
                                    SgScopeStatement *scope);
void AppendExprStatement(SgScopeStatement *scope,
                         SgExpression *exp);

SgVariableDeclaration *DeclarePSVectorInt(const std::string &name,
                                          const IntVector &vec,
                                          SgScopeStatement *block);


SgValueExp *BuildIntLikeVal(int v);
SgValueExp *BuildIntLikeVal(long v);
SgValueExp *BuildIntLikeVal(long long v);

void RedirectFunctionCalls(SgNode *node,
                           const std::string &current_func,
                           SgFunctionDeclaration *new_func);

void RedirectFunctionCall(SgFunctionCallExp *call,
                          SgExpression *new_target);


SgFunctionDeclaration *CloneFunction(SgFunctionDeclaration *decl,
                                     const std::string &new_name,
                                     SgScopeStatement *scope=NULL);

bool IsIntLikeType(const SgType *t);

template <class T> 
inline bool IsIntLikeType(const T *t) {
  return IsIntLikeType(t->get_type());
}

//! Check an AST node is conditional
/*!
  @param node An AST node.
  @return The conditional node if the given node is conditional. Null
  otherwise. 
 */
SgNode *IsConditional(const SgNode *node);

//! Find the nearest common parent.
SgNode *FindCommonParent(SgNode *n1, SgNode *n2);

template <class T>
T *GetASTAttribute(const SgNode *node) {
  return static_cast<T*>(node->getAttribute(T::name));
}

template <class T>
void AddASTAttribute(SgNode *node, T *attr) {
  PSAssert(attr);
  if (GetASTAttribute<T>(node)) {
    LOG_ERROR() << "Duplicated attribute: " << T::name << "\n";
    PSAbort(1);
  }
  node->addNewAttribute(T::name, attr);
}

template <class T>
void CopyASTAttribute(SgNode *dst_node,
                      const SgNode *src_node,
                      bool deep=true) {
  T *attr = GetASTAttribute<T>(src_node);
  PSAssert(attr);
  if (deep) {
    attr = static_cast<T*>(attr->copy());
  }
  AddASTAttribute<T>(dst_node, attr);
}

template <class T>
class QueryASTNodeVisitor: public AstSimpleProcessing {
 public:
  QueryASTNodeVisitor() {}
  virtual void visit(SgNode *node) {
    //LOG_DEBUG() << "query node: " << node->class_name() << "\n";
    //if (isSgCudaKernelExecConfig(node)) { return; }
    //LOG_DEBUG() << "query node: " << node->unparseToString() << "\n";
    if (GetASTAttribute<T>(node)) {
      nodes_.push_back(node);
    }
  }
  std::vector<SgNode *> nodes_;  
};

//! Query tree nodes with a given attribute type.
/*!
  \param top A traversal root node.
  \return A vector of nodes with an attribute of the given template
  type. The order of nodes is the same as the pre-order traversal of
  the tree.
 */
template <class T>
std::vector<SgNode *>
QuerySubTreeAttribute(SgNode *top) {
  QueryASTNodeVisitor<T> q;
  q.traverse(top, preorder);
  return q.nodes_;
}

//! Build a field reference.
/*!
  \param struct_var A reference to a struct or a struct value.
  \param field A reference to a field.
  \return A reference to the field in the struct.
*/
SgExpression *BuildFieldRef(
    SgExpression *struct_var, SgExpression *field);

//! Build a min expression.
SgExpression *BuildMin(SgExpression *x, SgExpression *y);

//! Build a maxb expression.
SgExpression *BuildMax(SgExpression *x, SgExpression *y);

void PrependExpression(SgExprListExp *exp_list,
                       SgExpression *exp);

void ReplaceFuncBody(SgFunctionDeclaration *func,
                     SgBasicBlock *new_body);

SgGlobal *GetGlobalScope();

SgExpression *GetVariableDefinitionRHS(SgVariableDeclaration *vdecl);

}  // namespace rose_util
}  // namespace translator
}  // namespace physis

#endif /* PHYSIS_TRANSLATOR_ROSE_UTIL_H__ */
