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

SgType *FindType(SgNode *topLevelNode, const string &typeName);
// For debugging
void printAllTypeNames(SgNode *topLevelNode, std::ostream &os);
// recursively removes all casts
SgExpression *removeCasts(SgExpression *exp);
SgFunctionDeclaration *getFuncDeclFromFuncRef(SgExpression *refExp);
template <class T>
T *GetDefiningDecl(T *decl);
template <> inline
SgFunctionDeclaration *GetDefiningDecl<SgFunctionDeclaration>(
    SgFunctionDeclaration *decl) {
  return isSgFunctionDeclaration(decl->get_definingDeclaration());
}
SgClassDefinition *getDefinition(SgClassType *t);
SgFunctionDeclaration *getContainingFunction(SgNode *node);
string getFuncName(SgFunctionRefExp *fref);
string getFuncName(SgFunctionCallExp *call);
void CopyExpressionPtrList(const SgExpressionPtrList &src,
                           SgExpressionPtrList &dst);
SgFunctionSymbol *getFunctionSymbol(SgFunctionDeclaration *f);
SgFunctionRefExp *getFunctionRefExp(SgFunctionDeclaration *decl);
SgVarRefExp *buildFieldRefExp(SgClassDeclaration *decl, string name);
bool IsFuncParam(SgInitializedName *in);
string generateUniqueName(SgScopeStatement *scope = NULL,
                          const string &prefix = "__ps_");
void SetFunctionStatic(SgFunctionDeclaration *fdecl);
SgExpression *BuildNULL();
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
void RemoveASTAttribute(SgNode *node) {
  if (!GetASTAttribute<T>(node)) {
    LOG_ERROR() << "No such attribute: " << T::name << "\n";
    PSAbort(1);
  }
  node->removeAttribute(T::name);
  return;
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
SgType *GetType(SgVariableDeclaration *decl);
SgName GetName(SgVariableDeclaration *decl);
SgName GetName(const SgVarRefExp *decl);
SgName GetName(const SgFunctionDeclaration *func);

SgExpression *ParseString(const string &s, SgScopeStatement *scope);

template <class T>
bool GetIntLikeVal(SgExpression *v, T &x) {
  if (isSgIntVal(v)) {
    x = (T)(isSgIntVal(v)->get_value());
  } else if (isSgUnsignedIntVal(v)) {
    x = (T)(isSgUnsignedIntVal(v)->get_value());
  } else if (isSgLongIntVal(v)) {
    x = (T)(isSgLongIntVal(v)->get_value());
  } else if (isSgUnsignedLongVal(v)) {
    x = (T)(isSgUnsignedLongVal(v)->get_value());
  } else if (isSgLongLongIntVal(v)) {
    x = (T)(isSgLongLongIntVal(v)->get_value());
  } else if (isSgUnsignedLongLongIntVal(v)) {
    x = (T)(isSgUnsignedLongLongIntVal(v)->get_value());
  } else {
    LOG_DEBUG() << "Not an int like value: "
                << v->unparseToString() << "\n";
    return false;
  }
  return true;
}

void ReplaceWithCopy(SgExpressionVector &ev);

bool IsInSameFile(SgLocatedNode *n1, SgLocatedNode *n2);

SgVarRefExp *GetUniqueVarRefExp(SgExpression *exp);

SgDeclarationStatement *GetDecl(SgVarRefExp *vref);

string GetInputFileSuffix(SgProject *proj);
bool IsFortran(SgProject *proj);

SgDeclarationStatement *FindMember(const SgClassType *ct,
                                   const string &member);
SgDeclarationStatement *FindMember(const SgClassDeclaration *cdecl,
                                   const string &member);
SgDeclarationStatement *FindMember(const SgClassDefinition *cdef,
                                   const string &member);

bool IsCLikeLanguage();
bool IsFortranLikeLanguage();

void SetAccessModifierUnspecified(SgDeclarationStatement *d);
void SetAccessModifier(SgDeclarationStatement *d,
                       SgAccessModifier::access_modifier_enum mod);

//! Builds a function declaration for C and Fortran
/*!

  Transparently hnandles the difference of C and Fortran function
  declrations. The parameters are mostly the same as
  SageBuilder::buildDefiningFunctionDeclaration. 
 */
SgFunctionDeclaration *BuildFunctionDeclaration(
    const string &name, SgType *ret_type, SgFunctionParameterList *parlist,
    SgScopeStatement *scope=NULL);

//! Builds an either C for loop or Fortran do loop
SgScopeStatement *BuildForLoop(SgInitializedName *ivar,
                               SgExpression *begin,
                               SgExpression *end,
                               SgExpression *incr,
                               SgBasicBlock *body);

//! Builds an either C or Fortran variable declration
SgVariableDeclaration *BuildVariableDeclaration(const string &name,
                                                SgType *type,
                                                SgInitializer *initializer=NULL,
                                                SgScopeStatement *scope=NULL);

SgEnumVal* BuildEnumVal(unsigned int value, SgEnumDeclaration* decl);

void GetArrayDim(SgArrayType *at, vector<size_t> &dims);


}  // namespace rose_util
}  // namespace translator
}  // namespace physis

#endif /* PHYSIS_TRANSLATOR_ROSE_UTIL_H__ */
