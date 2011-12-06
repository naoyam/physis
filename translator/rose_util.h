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
SgExpression *buildNULL();
SgVariableDeclaration *buildVarDecl(const string &name,
                                    SgType *type,
                                    SgExpression *val,
                                    SgScopeStatement *scope);
void AppendExprStatement(SgScopeStatement *scope,
                         SgExpression *exp);

SgVariableDeclaration *DeclarePSVectorInt(const std::string &name,
                                          const physis::util::IntVector &vec,
                                          SgScopeStatement *block);


SgValueExp *BuildIntLikeVal(int v);
SgValueExp *BuildIntLikeVal(long v);
SgValueExp *BuildIntLikeVal(long long v);

void RedirectFunctionCalls(SgNode *node,
                           const std::string &current_func,
                           SgFunctionRefExp *new_func);


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
  @return The conditional node or statement govering the given node.
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
  node->addNewAttribute(T::name, attr);
}

}  // namespace rose_util
}  // namespace translator
}  // namespace physis

#endif /* PHYSIS_TRANSLATOR_ROSE_UTIL_H__ */
