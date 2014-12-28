// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/rose_util.h"

#include <cctype>
#include <boost/foreach.hpp>

#include "translator/ast_traversal.h"
#include "translator/rose_fortran.h"

namespace sb = SageBuilder;
namespace si = SageInterface;
namespace rf = physis::translator::rose_fortran;

namespace physis {
namespace translator {
namespace rose_util {

SgType *FindType(SgNode *topLevelNode, const string &typeName) {
  SgName typeNameNode(typeName);
  Rose_STL_Container<SgNode*> types =
      NodeQuery::querySubTree(topLevelNode, V_SgNamedType);
  FOREACH(it, types.begin(), types.end()) {
    SgNamedType *type = isSgNamedType(*it);
    assert(type);
    if (type->get_name().getString() == typeName) {
      return type;
    }
  }
  return NULL;
}

void printAllTypeNames(SgNode *topLevelNode,
                       std::ostream &os) {
  Rose_STL_Container<SgNode*> types =
      NodeQuery::querySubTree(topLevelNode,
                              // odeQuery::VariableTypes);
                              V_SgType);
  FOREACH(it, types.begin(), types.end()) {
    SgType *type = isSgType(*it);
    assert(type);
    if (isSgNamedType(type)) {
      os << isSgNamedType(type)->get_name().getString() << "\n";
    }
  }
}

SgExpression *removeCasts(SgExpression *exp) {
  if (isSgCastExp(exp)) {
    return removeCasts(isSgCastExp(exp)->get_operand());
  } else {
    return exp;
  }
}

SgFunctionDeclaration *getFuncDeclFromFuncRef(SgExpression *refExp) {
  refExp = removeCasts(refExp);
  if (isSgFunctionRefExp(refExp)) {
    SgFunctionRefExp *rexp = isSgFunctionRefExp(refExp);
    return rexp->get_symbol()->get_declaration();
  } else {
    assert(0 && "Unsupported case");
  }
}

SgClassDefinition *getDefinition(SgClassType *t) {
  SgClassDeclaration *d =
      isSgClassDeclaration(t->get_declaration()->get_definingDeclaration());
  return isSgClassDefinition(d->get_definition());
}

SgFunctionDeclaration *getContainingFunction(SgNode *node) {
  while (node) {
    SgFunctionDeclaration *f = isSgFunctionDeclaration(node);
    if (f) return f;
    node = node->get_parent();
  }
  return NULL;
}

string getFuncName(SgFunctionRefExp *fref) {
  SgFunctionSymbol *fs = fref->get_symbol();
  return fs->get_name().getString();
}

string getFuncName(SgFunctionCallExp *call) {
  SgFunctionRefExp *fref = isSgFunctionRefExp(call->get_function());
  if (!fref) {
    LOG_ERROR() << "Non-direct function calls are not supported: "
                << call->unparseToString()
                << "\n";
    PSAbort(1);
  }
  return getFuncName(fref);
}

void CopyExpressionPtrList(const SgExpressionPtrList &src,
                           SgExpressionPtrList &dst) {
  dst.clear();
  FOREACH (it, src.begin(), src.end()) {
    dst.push_back(si::copyExpression(*it));
  }
}

SgFunctionSymbol *getFunctionSymbol(SgFunctionDeclaration *f) {
  return isSgFunctionSymbol(f->search_for_symbol_from_symbol_table());
}

SgFunctionRefExp *getFunctionRefExp(SgFunctionDeclaration *decl) {
  SgFunctionSymbol *fs = getFunctionSymbol(decl);
  return SageBuilder::buildFunctionRefExp(fs);
}

SgVarRefExp *buildFieldRefExp(SgClassDeclaration *decl, string name) {
  decl = isSgClassDeclaration(decl->get_definingDeclaration());
  SgVarRefExp *f =
      SageBuilder::buildVarRefExp(name, decl->get_definition());
  return f;
}

bool IsFuncParam(SgInitializedName *in) {
  const SgInitializedNamePtrList &params =
      si::getEnclosingFunctionDeclaration(in)->get_args();
  return isContained(params, in);
}

static int unique_name_var_index = 0;
string generateUniqueName(SgScopeStatement *scope, const string &prefix) {
  if (scope == NULL) {
    //scope = SageBuilder::topScopeStack();
    scope = si::getFirstGlobalScope(si::getProject());
  }
  ROSE_ASSERT(scope);
  string name;  
#if 0  
  SgSymbolTable *symbol_table = scope->get_symbol_table();
  int post_fix = symbol_table->size();
  do {
    name = prefix + toString(post_fix);
    post_fix++;
  } while (symbol_table->exists(name));
#else
  do {
    int index = unique_name_var_index++;
    name = prefix + toString(index);
  } while (si::lookupVariableSymbolInParentScopes(name, scope));
#endif
  return name;
}

void SetFunctionStatic(SgFunctionDeclaration *fdecl) {
  fdecl->get_declarationModifier().get_storageModifier().setStatic();
}

SgExpression *buildNULL(SgScopeStatement *global_scope) {
#if 0  
  static SgVariableDeclaration *dummy_null = NULL;
  if (!dummy_null) {
    dummy_null =
        sb::buildVariableDeclaration(
            "NULL",
            sb::buildPointerType(sb::buildVoidType()),
            NULL, global_scope);
  }
  return sb::buildVarRefExp(dummy_null);
#else
  return sb::buildOpaqueVarRefExp("NULL");
#endif
}

SgVariableDeclaration *buildVarDecl(const string &name,
                                    SgType *type,
                                    SgExpression *val,
                                    SgScopeStatement *scope) {
  SgAssignInitializer *init =
      sb::buildAssignInitializer(val, type);
  SgVariableDeclaration *sdecl
      = sb::buildVariableDeclaration(name, type, init, scope);
  si::appendStatement(sdecl, scope);
  return sdecl;
}

void AppendExprStatement(SgScopeStatement *scope,
                         SgExpression *exp) {
  si::appendStatement(sb::buildExprStatement(exp), scope);
  return;
}

SgValueExp *BuildIntLikeVal(int v) {
  return sb::buildIntVal(v);
}

SgValueExp *BuildIntLikeVal(long v) {
  return sb::buildLongIntVal(v);
}

SgValueExp *BuildIntLikeVal(long long v) {
  return sb::buildLongLongIntVal(v);
}

SgVariableDeclaration *DeclarePSVectorInt(const std::string &name,
                                          const IntVector &vec,
                                          SgScopeStatement *block) {
  
  SgType *vec_type = sb::buildArrayType(sb::buildIntType(),
                                        sb::buildIntVal(PS_MAX_DIM));
  SgExprListExp *init_expr = sb::buildExprListExp();
  FOREACH (it, vec.begin(), vec.end()) {
    IntVector::value_type v = *it;
    si::appendExpression(init_expr, BuildIntLikeVal(v));
  }
  
  SgAggregateInitializer *init
      = sb::buildAggregateInitializer(init_expr, vec_type);
  SgVariableDeclaration *decl
      = sb::buildVariableDeclaration(name, vec_type, init, block);
  si::appendStatement(decl, block);
  
  return decl;
}

void RedirectFunctionCalls(SgNode *node,
                           const std::string &current_func,
                           SgFunctionDeclaration *new_func) {
  Rose_STL_Container<SgNode*> calls =
      NodeQuery::querySubTree(node, V_SgFunctionCallExp);
  SgFunctionSymbol *curfs =
      si::lookupFunctionSymbolInParentScopes(current_func);
  FOREACH (it, calls.begin(), calls.end()) {
    SgFunctionCallExp *fc = isSgFunctionCallExp(*it);
    PSAssert(fc);
    if (fc->getAssociatedFunctionSymbol() != curfs)
      continue;
    LOG_DEBUG() << "Redirecting call to " << current_func
                << " to " << new_func << "\n";
    SgFunctionCallExp *new_call =
        sb::buildFunctionCallExp(
            sb::buildFunctionRefExp(new_func),
            isSgExprListExp(si::copyExpression(fc->get_args())));
    si::replaceExpression(fc, new_call);
  }
  
}

SgFunctionDeclaration * CloneFunction(SgFunctionDeclaration *decl,
                                      const std::string &new_name,
                                      SgScopeStatement *scope) {
  SgFunctionDeclaration *new_decl
      = sb::buildDefiningFunctionDeclaration(
          new_name,
          decl->get_type()->get_return_type(),
          static_cast<SgFunctionParameterList*>(
              si::copyStatement(decl->get_parameterList())),
          scope);
  SgFunctionDefinition *new_def =
      static_cast<SgFunctionDefinition*>(
          si::copyStatement(decl->get_definition()));
  new_decl->set_definition(new_def);
  new_def->set_declaration(new_decl);
  new_decl->get_functionModifier() = decl->get_functionModifier();
  return new_decl;
}

bool IsIntLikeType(const SgType *t) {
  if (isSgModifierType(t)) {
    return IsIntLikeType(isSgModifierType(t)->get_base_type());
  }
  return t->isIntegerType();
}

SgNode *IsConditional(const SgNode *node) {
  while (true) {
    SgNode *parent = node->get_parent();
    // Stop if no parent found
    if (!parent) break;
    // True if the parent is a node for conditional execution
    if (isSgIfStmt(parent) ||
        isSgConditionalExp(parent)) {
      return parent;
    }
    // Stop when crossing function boundary
    if (isSgFunctionDeclaration(parent)) {
      break;
    }
    node = parent;
  }
  return NULL;
}


SgNode *FindCommonParent(SgNode *n1, SgNode *n2) {
  while (n1) {
    if (n1 == n2 || si::isAncestor(n1, n2)) return n1;
    n1 = n1->get_parent();
  }
  return NULL;
}

SgExpression *BuildFieldRef(
    SgExpression *struct_var, SgExpression *field) {
  if (!IsFortranLikeLanguage() &&
      si::isPointerType(struct_var->get_type())) {
    return sb::buildArrowExp(struct_var, field);
  } else {
    return sb::buildDotExp(struct_var, field);
  }
}

SgExpression *BuildMin(SgExpression *x,
                       SgExpression *y) {
  return sb::buildConditionalExp(
      sb::buildLessThanOp(x, y),
      si::copyExpression(x), si::copyExpression(y));
}

SgExpression *BuildMax(SgExpression *x,
                       SgExpression *y) {
  return sb::buildConditionalExp(
      sb::buildGreaterThanOp(x, y),
      si::copyExpression(x), si::copyExpression(y));
}

void RedirectFunctionCall(SgFunctionCallExp *call,
                          SgExpression *new_target) {
  SgFunctionCallExp *new_call =
      sb::buildFunctionCallExp(
          new_target,
          isSgExprListExp(si::copyExpression(call->get_args())));
  si::replaceExpression(call, new_call);
}

void PrependExpression(SgExprListExp *exp_list,
                       SgExpression *exp) {
  // Based on SageInterface::appendExpression
  PSAssert(exp_list);
  PSAssert(exp);
  exp_list->prepend_expression(exp);
  exp->set_parent(exp_list);
}

void ReplaceFuncBody(SgFunctionDeclaration *func,
                     SgBasicBlock *new_body) {
  func = isSgFunctionDeclaration(func->get_definingDeclaration());
  SgBasicBlock *cur_body = func->get_definition()->get_body();
  si::replaceStatement(cur_body, new_body);
}

SgGlobal *GetGlobalScope() {
  SgGlobal *g = si::getFirstGlobalScope(si::getProject());
  PSAssert(g);
  return g;
}

SgExpression *GetVariableDefinitionRHS(SgVariableDeclaration *vdecl) {
  SgAssignInitializer *asinit =
      isSgAssignInitializer(
          vdecl->get_definition()->get_vardefn()->get_initializer());
  PSAssert(asinit);
  SgExpression *rhs =  asinit->get_operand();
  PSAssert(rhs);
  return rhs;
}

SgType *GetType(SgVariableDeclaration *decl) {
  return decl->get_variables()[0]->get_type();
}
SgName GetName(SgVariableDeclaration *decl) {
  return decl->get_variables()[0]->get_name();
}

SgName GetName(const SgVarRefExp *x) {
  return x->get_symbol()->get_name();
}

SgName GetName(const SgFunctionDeclaration *func) {
  return func->get_name();
}

// TODO: Complete string parsing
// Currently only limited string value is supported.
// - just an integer
// - just a variable reference
SgExpression *ParseString(const string &s, SgScopeStatement *scope) {
#if 0  
  bool is_numeric = true;
  FOREACH (it, s.begin(), s.end()) {
    if (!isdigit(*it)) {
      is_numeric = false;
      break;
    }
  }
  if (is_numeric) {
    int x = toInteger(s);
    return sb::buildIntVal(x);
  }
  // NOTE: assume variable reference
  return sb::buildVarRefExp(s);
#else
  SgExpression* result = NULL;
  assert (scope != NULL);
  // set input and context for the parser
  AstFromString::c_char = s.c_str();
  assert (AstFromString::c_char== s.c_str());
  AstFromString::c_sgnode = scope;
  if (AstFromString::afs_match_expression()) {
    result = isSgExpression(AstFromString::c_parsed_node); // grab the result
    assert (result != NULL);
    LOG_DEBUG() << "Parsed expression: " << result->unparseToString() << "\n";
  } else {
    LOG_ERROR() <<"Error. buildStatementFromString() cannot parse input string:"<<s
                <<"\n\t under the given scope:"<<scope->class_name() << "\n";
    PSAssert(0);
  }
  return result;
#endif
}

void ReplaceWithCopy(SgExpressionVector &ev) {
  FOREACH (it, ev.begin(), ev.end()) {
    SgExpression *e = si::copyExpression(*it);
    *it = e;
  }
}

bool IsInSameFile(SgLocatedNode *n1, SgLocatedNode *n2) {
  const string &n1_name = n1->get_file_info()->get_filenameString();
  const string &n2_name = n2->get_file_info()->get_filenameString();
  LOG_DEBUG() << "N1: " << n1_name << ", N2: " << n2_name << "\n";
  return n1->get_file_info()->isSameFile(n2->get_file_info());
}

SgVarRefExp *GetUniqueVarRefExp(SgExpression *exp) {
  vector<SgVarRefExp*> v = si::querySubTree<SgVarRefExp>(exp);
  if (v.size() == 1) {
    return v.front();
  } else {
    return NULL;
  }
}

SgDeclarationStatement *GetDecl(SgVarRefExp *vref) {
  PSAssert(vref);
  SgInitializedName *in = vref->get_symbol()->get_declaration();
  PSAssert(in);
  return in->get_declaration();
}

string GetInputFileSuffix(SgProject *proj) {
  string name = proj->get_sourceFileNameList()[0];
  string suffix = name.substr(name.rfind(".")+1);
  return suffix;
}

bool IsFortran(SgProject *proj) {
  string suffix = GetInputFileSuffix(proj);
  return suffix == "f90" || suffix == "F90";
}

SgDeclarationStatement *FindMember(const SgClassType *ct,
                                   const string &member) {
  SgClassDeclaration *decl =
      isSgClassDeclaration(ct->get_declaration());
  return FindMember(decl, member);
}

SgDeclarationStatement *FindMember(const SgClassDeclaration *cdecl,
                                   const string &member) {
  SgClassDefinition *def =
      isSgClassDeclaration(cdecl->get_definingDeclaration())->
      get_definition();
  return FindMember(def, member);
}

SgDeclarationStatement *FindMember(const SgClassDefinition *cdef,
                                   const string &member) {
  const SgDeclarationStatementPtrList &members =
      cdef->get_members();
  BOOST_FOREACH (SgDeclarationStatement *decl, members) {
    string name = "";
    if (isSgVariableDeclaration(decl)) {
      name = rose_util::GetName(isSgVariableDeclaration(decl));
    } else if (isSgFunctionDeclaration(decl)) {
      name = rose_util::GetName(isSgFunctionDeclaration(decl));
    }
    if (name != "" && name == member) {
      return decl;
    }
  }
  return NULL;
}

bool IsCLikeLanguage() {
  return si::is_C_language() || si::is_Cxx_language();
}

bool IsFortranLikeLanguage() {
  return si::is_Fortran_language();
}

void SetAccessModifierUnspecified(SgDeclarationStatement *d) {
  if (IsFortranLikeLanguage()) {
    SetAccessModifier(d, SgAccessModifier::e_undefined);
  }
}

void SetAccessModifier(SgDeclarationStatement *d,
                       SgAccessModifier::access_modifier_enum mod) {
  SgAccessModifier &am = d->get_declarationModifier().get_accessModifier();
  am.set_modifier(mod);
}

SgFunctionDeclaration *BuildFunctionDeclaration(
    const string &name, SgType *ret_type, SgFunctionParameterList *parlist,
    SgScopeStatement *scope) {
  SgFunctionDeclaration *func = NULL;
  if (IsCLikeLanguage()) {
    func = sb::buildDefiningFunctionDeclaration(
        name, ret_type, parlist, scope);
  } else {
    // If the ret_type is void, builds a subroutine.
    SgProcedureHeaderStatement::subprogram_kind_enum kind =
        isSgTypeVoid(ret_type) ?
        SgProcedureHeaderStatement::e_subroutine_subprogram_kind :
        SgProcedureHeaderStatement::e_function_subprogram_kind;
    func = sb::buildProcedureHeaderStatement(
        name.c_str(), ret_type, parlist, kind, scope);
  }
  PSAssert(func);
  return func;
}

SgScopeStatement *BuildForLoop(SgInitializedName *ivar,
                               SgExpression *begin,
                               SgExpression *end,
                               SgExpression *incr,
                               SgBasicBlock *body) {
  SgAssignOp *init_expr = sb::buildAssignOp(
      sb::buildVarRefExp(ivar), begin);
  SgScopeStatement *loop = NULL;
  bool is_static_incr = false;
  int static_incr = 0;
  if (isSgIntVal(incr)) {
    is_static_incr = true;
    static_incr = isSgIntVal(incr)->get_value();
  }
  if (IsCLikeLanguage()) {
    SgExpression *cmp = NULL;
    SgExpression *for_incr = NULL;    
    if (is_static_incr) {
      if (static_incr > 0) {
        cmp = sb::buildLessOrEqualOp(
            sb::buildVarRefExp(ivar), end);
        for_incr = sb::buildPlusAssignOp(
            sb::buildVarRefExp(ivar), incr);
      } else {
        cmp = sb::buildGreaterOrEqualOp(
            sb::buildVarRefExp(ivar), end);
        for_incr = sb::buildMinusAssignOp(
            sb::buildVarRefExp(ivar),
            sb::buildIntVal(-static_incr));
      }
    } else {
      cmp = sb::buildNotEqualOp(sb::buildVarRefExp(ivar), end);
      for_incr = sb::buildPlusAssignOp(
          sb::buildVarRefExp(ivar), incr);
    }
    loop = sb::buildForStatement(
        sb::buildExprStatement(init_expr),
        sb::buildExprStatement(cmp), for_incr, body);
  } else if (IsFortranLikeLanguage()) {
    loop = rf::BuildFortranDo(init_expr, end, incr, body);
  }
  PSAssert(loop);
  return loop;
}


SgEnumVal* BuildEnumVal(unsigned int value, SgEnumDeclaration* decl) {
  ROSE_ASSERT(decl);
  SgInitializedNamePtrList &members = decl->get_enumerators();
  ROSE_ASSERT(value < members.size());
  SgInitializedName *name = members[value];
  SgEnumVal* enum_val= sb::buildEnumVal_nfi(value, decl, name->get_name());
  ROSE_ASSERT(enum_val);
  si::setOneSourcePositionForTransformation(enum_val);
  return enum_val;
}

SgVariableDeclaration *BuildVariableDeclaration(const string &name,
                                                SgType *type,
                                                SgInitializer *initializer,
                                                SgScopeStatement *scope) {
  SgVariableDeclaration *d =
      sb::buildVariableDeclaration(name, type, initializer, scope);
  if (IsFortranLikeLanguage()) SetAccessModifierUnspecified(d);
  return d;
}

}  // namespace rose_util
}  // namespace translator
}  // namespace physis
