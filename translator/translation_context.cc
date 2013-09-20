// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/translation_context.h"

#include <memory>
#include <ostream>
#include <vector>
#include <boost/foreach.hpp>

#include "translator/rose_util.h"
#include "translator/def_analysis.h"
#include "translator/stencil_analysis.h"
#include "translator/physis_names.h"

using std::vector;
using std::auto_ptr;
using std::ostream;

namespace sb = SageBuilder;
namespace si = SageInterface;
namespace ru = physis::translator::rose_util;

namespace physis {
namespace translator {

void TranslationContext::AnalyzeGridTypes() {
  Rose_STL_Container<SgNode*> structDefs =
      NodeQuery::querySubTree(project_, NodeQuery::StructDefinitions);
  FOREACH(it, structDefs.begin(), structDefs.end()) {
    SgClassDefinition *def = isSgClassDefinition(*it);
    if (def == NULL) {
      // LOG_DEBUG() << (*it)->class_name()
      // < " is not class def\n";
      continue;
    }
    
    if (def->get_file_info()->isCompilerGenerated()) {
      // skip
      continue;
    }
    SgClassDeclaration *decl = def->get_declaration();
    SgClassType *type = decl->get_type();
    const string typeName = type->get_name().getString();

    if (!GridType::isGridType((typeName))) continue;
    
    LOG_DEBUG() << "type name: " << typeName << "\n";

    // Find the user-visile type 
    SgNamedType *utype = NULL;
    if (rose_util::IsFortran(project_)) {
      // In Fortran, the same type is used in the user code
      utype = type;
    } else {
      // In C, typedef'ed type is used
      SgPointerType *gridPtr = type->get_ptr_to();
      utype =
          isSgTypedefType(*(gridPtr->get_typedefs()->get_typedefs().begin()));
    }
    
    const string utypeName = utype->get_name().getString();

    LOG_DEBUG() << "Grid type found: " << utypeName
                << " (" << typeName
                << ")\n";

    GridType *gt = new GridType(type, utype);
    LOG_DEBUG() << "Created grid type: " << gt->toString() << "\n";
    registerGridType(utype, gt);
    rose_util::AddASTAttribute<GridType>(utype, gt);
    rose_util::AddASTAttribute<GridType>(decl, gt);
  }
}

/*!
  Ensure every grid SgInitializedName has GridVarAttribute, which
  should replace the (SgInitializedName, GridSet) map in
  TranslationContext.

  \param in Grid variable
  \return true if new attr is attached
*/
static bool EnsureGridVarAttribute(SgInitializedName *in) {
  if (rose_util::GetASTAttribute<GridVarAttribute>(in)) {
    return false;
  } else {
    GridType *gt = 
        rose_util::GetASTAttribute<GridType>(in);
    PSAssert(gt);
    GridVarAttribute *gva = new GridVarAttribute(gt);
    rose_util::AddASTAttribute<GridVarAttribute>(in, gva);
    LOG_DEBUG() << "Adding GridVarAttribute to "
                << in->unparseToString() << "\n";
    return true;
  }
}

static bool HandleGridNew(SgInitializedName *in,
                          SgFunctionCallExp *callExp,
                          TranslationContext &tx) {
  Grid *g = tx.getOrCreateGrid(callExp);
  return tx.associateVarWithGrid(in, g);
}

static bool HandleGridNewFortran(SgFunctionCallExp *callExp,
                                 TranslationContext &tx) {
  Grid *g = tx.getOrCreateGrid(callExp);
  SgInitializedName *gv =
      si::convertRefToInitializedName(
          callExp->get_args()->get_expressions()[0]);
  LOG_DEBUG() << "Fortran GridNew to " << gv->unparseToString() << "\n";  
  return tx.associateVarWithGrid(gv, g);
}


static bool HandleGridAlias(SgInitializedName *dst,
                            SgInitializedName *src,
                            TranslationContext &tx) {
  /*
    LOG_DEBUG() << "original: " << src->unparseToString()
    << ", alias: " << dst->unparseToString()
    << "\n";
  */

  const GridSet *gs = tx.findGrid(src);
  if (gs == NULL) {
    // src is not yet analyzerd; skip this assignment this
    // time
    LOG_DEBUG() << "Src is not yet analyzed\n";
    return false;
  }

  // LOG_DEBUG() << "src grid found\n";

  bool changed = false;
  FOREACH(it, gs->begin(), gs->end()) {
    /*
      if (*it)
      LOG_DEBUG() << "src: " << (*it)->toString() << "\n";
    */
    changed |= tx.associateVarWithGrid(dst, *it);
  }

  return changed;
}

static bool HandleGridAssignment(SgInitializedName *lhs_in,
                                 SgExpression *rhs_exp,
                                 TranslationContext &tx) {
  // rhs is either a grid variable or a call to grid new
  if (isSgVarRefExp(rhs_exp)) {
    SgVarRefExp *rhs_vref = isSgVarRefExp(rhs_exp);
    return HandleGridAlias(
        lhs_in, si::convertRefToInitializedName(rhs_vref), tx);
  } else if (isSgFunctionCallExp(rhs_exp)) {
    return HandleGridNew(lhs_in, isSgFunctionCallExp(rhs_exp), tx);
  } else {
    // error
    LOG_ERROR() << "Invalid rhs: " << rhs_exp->unparseToString() << "\n";
    PSAbort(1);
    return false;
  }
}

static bool IsInGlobalScope(SgInitializedName *in) {
  return si::getGlobalScope(in) == si::getScope(in);
}

static bool HandleGridDeclaration(SgInitializedName *in,
                                  TranslationContext &tx) {

  EnsureGridVarAttribute(in);
  
  if (IsInGlobalScope(in)) {
    LOG_DEBUG() << "Global variable: " << in->unparseToString() << "\n";
    return tx.associateVarWithGrid(in, NULL);
  }
  
  bool changed = false;

  SgDeclarationStatement *decl = in->get_declaration();
  
  if (isSgFunctionParameterList(decl)) {
    // Case: Function parameter    
    SgFunctionDeclaration *func = si::getEnclosingFunctionDeclaration(in);
    if (si::isStatic(func)) {
      // in is a parameter of a static function.
      // This is handled at call sites.
    } else {
      changed |= tx.associateVarWithGrid(in, NULL);
    }
  } else if (isSgVariableDeclaration(decl)) {
    // Case: Local variable declaration
    SgAssignInitializer *asn = isSgAssignInitializer(
        in->get_initializer());
    if (asn) {
      SgExpression *rhs = asn->get_operand();
      changed |= HandleGridAssignment(in, rhs, tx);
    } else {
      // when not assigned
      changed |= tx.associateVarWithGrid(in, NULL);
    }
  } else {
    LOG_ERROR() << "Unsupported InitializedName: "
                << in->unparseToString()
                << " appearing in "
                << si::getEnclosingStatement(in)->unparseToString()
                << "\n";
    PSAbort(1);
  }

  return changed;
}

static bool PropagateGridVarMapAcrossCall(SgExpressionPtrList &args,
                                          SgInitializedNamePtrList &params,
                                          TranslationContext &tx) {
  bool changed = false;
  int arg_num = args.size();
  PSAssert(args.size() == params.size());
  for (int i = 0; i < arg_num; ++i) {
    SgExpression *arg = args[i];
    SgInitializedName *param = params[i];
    if (!GridType::isGridType(arg->get_type())) continue;    
    SgVarRefExp *gv = isSgVarRefExp(arg);
    // grids as argument must be a variable
    assert(gv);
    SgInitializedName *src = si::convertRefToInitializedName(gv);
    LOG_DEBUG() << "arg: " << arg->unparseToString()
                << " (" << src << ")"
                << ", param: " << param->unparseToString()
                << " (" << param << ")" << "\n";
    bool b = HandleGridAlias(param, src, tx);
    LOG_DEBUG() << "Changed? " << b << "\n";
    changed |= b;
  }
  return changed;
}

static bool PropagateGridVarMapAcrossStencilCall(SgFunctionCallExp *c,
                                                 StencilMap *m,
                                                 TranslationContext &tx) {
  LOG_DEBUG() << "StencilMap: " << m->getKernel()->get_name().str() << "\n";
  int param_skip = 2;
  // Skip parameter PSStencil in Fortran
  if (ru::IsFortranLikeLanguage()) ++param_skip;
  SgExpressionPtrList args(
      c->get_args()->get_expressions().begin() + param_skip,
      c->get_args()->get_expressions().end());
  SgInitializedNamePtrList params(
      m->getKernel()->get_args().begin() + m->getNumDim(),
      m->getKernel()->get_args().end());
  return PropagateGridVarMapAcrossCall(args, params, tx);
}

static bool PropagateGridVarMapAcrossOrdinaryCall(SgFunctionCallExp *c,
                                                  TranslationContext &tx) {

  SgExpressionPtrList &args = c->get_args()->get_expressions();
  bool grid_is_used = false;
  FOREACH (it, args.begin(), args.end()) {
    if (GridType::isGridType((*it)->get_type())) {
      grid_is_used = true;
      break;
    }
  }
  // No grid argument passed
  if (!grid_is_used) return false;

  SgFunctionDeclaration *callee_decl =
      si::getDeclarationOfNamedFunction(c->get_function());
  if (!callee_decl) {
    LOG_ERROR() << "Callee not found: " << c->unparseToString() << "\n";
    PSAbort(1);
  }

  // These are not taken care here.
  if (StencilMap::IsMap(c) || Reduce::IsReduce(c) ||
      callee_decl->get_name() == PSF_GRID_NEW_NAME ||
      tx.IsCopyin(c) || tx.IsCopyout(c)) return false;
  
  LOG_DEBUG() << "Grid var propagation: " <<
      callee_decl->unparseToString() << "\n";  
  
  callee_decl = isSgFunctionDeclaration(
      callee_decl->get_definingDeclaration());
  PSAssert(callee_decl);
  
  SgInitializedNamePtrList &params = callee_decl->get_args();
  return PropagateGridVarMapAcrossCall(args, params, tx);
}

void TranslationContext::AnalyzeGridVars() {
  LOG_DEBUG() << "Analyzing Grid variables\n";
  vector<SgInitializedName*> vars =
      si::querySubTree<SgInitializedName>(project_);
  vector<SgAssignOp*> asns =
      si::querySubTree<SgAssignOp>(project_);
  vector<SgFunctionCallExp*> calls =
      si::querySubTree<SgFunctionCallExp>(project_);
  bool changed = true;
  while (changed) {
    changed = false;
    BOOST_FOREACH(SgInitializedName *in, vars) {
      SgType *type = in->get_type();
      GridType *gt = findGridType(type);
      if (gt == NULL) continue;
      LOG_DEBUG() << "Grid object of type "
                  << gt->toString() << " found: "
                  << in->unparseToString() << "\n";
      if (!rose_util::GetASTAttribute<GridType>(in)) {
          rose_util::AddASTAttribute(in, gt);
      }
      changed |= HandleGridDeclaration(in, *this);
    }
    BOOST_FOREACH(SgAssignOp *aop, asns) {
      // Skip non-grid assignments
      SgType *type = aop->get_type();
      if (!findGridType(type)) continue;
      LOG_DEBUG() << "Grid assignment: "
                  << aop->unparseToString() << "\n";
      SgVarRefExp *lhs = isSgVarRefExp(aop->get_lhs_operand());
      assert(lhs);
      changed |= HandleGridAssignment(
          si::convertRefToInitializedName(lhs),
          aop->get_rhs_operand(), *this);
    }
    // Handle stencil map
    FOREACH(it, mapBegin(), mapEnd()) {
      SgFunctionCallExp *c = isSgFunctionCallExp(it->first);
      if (!c) continue;
      StencilMap *m = it->second;
      changed |= PropagateGridVarMapAcrossStencilCall(c, m, *this);
    }
    
    // Note: grids are read-only in reduction kernels, so no need to
    // analyze them

    // Handle normal function call
    BOOST_FOREACH (SgFunctionCallExp *c, calls) {
      SgFunctionDeclaration *callee_decl =
          si::getDeclarationOfNamedFunction(c->get_function());
      if (callee_decl && callee_decl->get_name() == PSF_GRID_NEW_NAME) {
        changed |= HandleGridNewFortran(c, *this);
      } else {
        changed |= PropagateGridVarMapAcrossOrdinaryCall(c, *this);
      }
    }
  }
  LOG_DEBUG() << "Grid variable analysis done\n";
}

static void AssociateDomainAttr(SgExpression *exp, Domain *dom,
                                TranslationContext *tx) {
  LOG_DEBUG() << "Domain attr: "
              << dom->toString() << "\n";
  Domain *old = rose_util::GetASTAttribute<Domain>(exp);
  if (old) {
    old->Merge(*dom);
    delete dom;
  } else {
    //tx->associateExpWithDomain(exp, dom);
    rose_util::AddASTAttribute<Domain>(exp, dom);
  }
}

void TranslationContext::AnalyzeDomainExpr() {
  assert(PS_MAX_DIM == 3);
  SgType* domTypes[] = {dom1d_type_, dom2d_type_, dom3d_type_};
  vector<SgType*> v(domTypes, domTypes+4);
  auto_ptr<DefMap> defMap = findDefinitions(project_, v);
  
  LOG_DEBUG() << "Domain DefMap: " << toString(*defMap) << "\n";

  vector<SgExpression*> exps = si::querySubTree<SgExpression>(project_);
  BOOST_FOREACH(SgExpression *exp, exps) {
    SgType *dom_type = exp->get_type();
    if (!Domain::isDomainType(dom_type)) continue;

    if (isSgVarRefExp(exp)) {
      SgInitializedName *in =
          si::convertRefToInitializedName(isSgVarRefExp(exp));
      SgExpressionPtrList &defs = (*defMap)[in];
      FOREACH(defIt, defs.begin(), defs.end()) {
        SgExpression *def = *defIt;
        LOG_DEBUG() << "Dom Def: "
                    << (def ? def->unparseToString() : string("<NULL>"))
                    << "\n";
        Domain *dom = NULL;
        if (isSgFunctionCallExp(def)) {
          dom = Domain::GetDomain(isSgFunctionCallExp(def));
        } else if (dom == NULL) {
          dom = Domain::GetDomain(getNumDimOfDomain(dom_type));
        }
        AssociateDomainAttr(exp, dom, this);
      }
    } else if (isSgFunctionCallExp(exp)) {
      SgFunctionCallExp *call = isSgFunctionCallExp(exp);
      Domain *dom = Domain::GetDomain(call);
      //associateExpWithDomain(call, dom);
      rose_util::AddASTAttribute<Domain>(call, dom);      
    } else {
      // OG_DEBUG() << "Ignoring domain expression of type: "
      // < exp->class_name() << "\n";
    }
  }
}

void TranslationContext::AnalyzeMap() {
  LOG_DEBUG() << "Analyzing Map\n";
  vector<SgFunctionCallExp*> calls =
      si::querySubTree<SgFunctionCallExp>(project_);
  BOOST_FOREACH(SgFunctionCallExp *call, calls) {
    if (!StencilMap::IsMap(call)) continue;
    LOG_DEBUG() << "Call to stencil_map found: "
                << call->unparseToString() << "\n";
    StencilMap *mc = new StencilMap(call, this);
    registerMap(call, mc);
  }
  LOG_DEBUG() << "Map analysis done\n";
}

void TranslationContext::locateDomainTypes() {
  dom1d_type_ = rose_util::FindType(project_, PS_DOMAIN1D_TYPE_NAME);
  dom2d_type_ = rose_util::FindType(project_, PS_DOMAIN2D_TYPE_NAME);
  dom3d_type_ = rose_util::FindType(project_, PS_DOMAIN3D_TYPE_NAME);
  //f_dom_type_ = rose_util::FindType(project_, PSF_DOMAIN_TYPE_NAME);
  assert(dom1d_type_);
  assert(dom2d_type_);
  assert(dom3d_type_);
}

// TODO: Now all types are predefined.
void TranslationContext::Build() {
  // eneratePDF(*project_);
  // enerateDOT(*project_);

  /*
    Not sure call graph analysis is mature. errors are produced,
    though the code works.
    CallGraphBuilder cgb(project_);
    cgb.buildCallGraph();
    call_graph_ = cgb.getGraph();

    // Optional graph output {
    AstDOTGeneration dotgen;
    SgFilePtrList file_list = project_->get_fileList();
    string firstFileName =
    StringUtility::stripPathFromFileName(file_list[0]->getFileName());
    dotgen.writeIncidenceGraphToDOTFile(callGraph,
    firstFileName+"_callgraph.dot");
    }
  */

  // printAllTypeNames(project_, std::cout);

  // preparation
  locateDomainTypes();

  AnalyzeGridTypes();
  AnalyzeDomainExpr();
  AnalyzeMap();
  AnalyzeReduce();

  AnalyzeGridVars();
  LOG_DEBUG() << "grid variable analysis done\n";
  print(std::cout);

  AnalyzeRun();
  AnalyzeKernelFunctions();
#ifdef UNUSED_CODE
  markReadWriteGrids();
#endif
  LOG_INFO() << "Analyzing stencil range\n";

  AnalyzeGet(project_, *this);
  AnalyzeEmit(project_, *this);
  
  FOREACH (it, stencil_map_.begin(), stencil_map_.end()) {
    AnalyzeStencilRange(*(it->second), *this);
  }

  LOG_INFO() << "Translation context built\n";
  print(std::cout);
}

GridType *TranslationContext::findGridTypeByNew(const string &fname) {
  // This is not strictly required. Just filter out most functions.
  if (!endswith(fname, "New")) return NULL;
  FOREACH(it, gridTypeBegin(), gridTypeEnd()) {
    GridType *gt = it->second;
    if (gt->getNewName() == fname) return gt;
  }
  return NULL;
}

bool TranslationContext::isNewFunc(const string &fname) {
  return findGridTypeByNew(fname) != NULL;
}

static SgVarRefExp *ExtractGridVarRef(SgFunctionCallExp *call,
                                      TranslationContext *tx) {
  SgVarRefExp *gexp =
      isSgVarRefExp(
          rose_util::removeCasts(
              call->get_args()->get_expressions()[0]));
  PSAssert(gexp);
  PSAssert(tx->findGridType(gexp));
  return gexp;
}

bool TranslationContext::isNewCall(SgFunctionCallExp *ce) {
  SgFunctionRefExp *fref =
      isSgFunctionRefExp(ce->get_function());
  if (!fref) return false;
  const string name = rose_util::getFuncName(fref);
  return isNewFunc(name);
}

SgVarRefExp *TranslationContext::IsFree(SgFunctionCallExp *ce) {
  SgFunctionRefExp *fref =
      isSgFunctionRefExp(ce->get_function());
  if (!fref) return NULL;
  const string name = rose_util::getFuncName(fref);
  if (name != "PSGridFree") return NULL;
  return ExtractGridVarRef(ce, this);
}

SgVarRefExp *TranslationContext::IsCopyin(SgFunctionCallExp *ce) {
  SgFunctionRefExp *fref =
      isSgFunctionRefExp(ce->get_function());
  if (!fref) return NULL;
  const string name = rose_util::getFuncName(fref);
  if (name != "PSGridCopyin") return NULL;
  return ExtractGridVarRef(ce, this);
}

SgVarRefExp *TranslationContext::IsCopyout(
    SgFunctionCallExp *ce) {
  SgFunctionRefExp *fref =
      isSgFunctionRefExp(ce->get_function());
  if (!fref) return NULL;
  const string name = rose_util::getFuncName(fref);
  if (name != "PSGridCopyout") return NULL;
  return ExtractGridVarRef(ce, this);
}

static bool IsBuiltInFunction(const string &func_name) {
  // TODO: more complete support for builtin math functions
  string buildin_functions[] = {
    "exp", "expf", "log", "powf", "pow", "sqrtf", "sqrt", "cos", "cosf",
    "sin", "sinf", "acos", "acosf", "fabs", "fabsf"};
  int num_functions = sizeof(buildin_functions) / sizeof(string);
  LOG_DEBUG() << "num_functions: " << num_functions << "\n";
  return std::find(buildin_functions, buildin_functions + num_functions,
                   func_name) != buildin_functions + num_functions;
}

static void EnsureKernelBodyAttribute(SgFunctionDeclaration *kernel_decl) {
  SgScopeStatement *body =
      kernel_decl->get_definition()->get_body();
  if (!rose_util::GetASTAttribute<KernelBody>(body)) {
    rose_util::AddASTAttribute<KernelBody>(
        body, new KernelBody());
  }
}

static void AttachStencilIndexVarAttribute(SgFunctionDeclaration *kernel_decl) {
  SgInitializedNamePtrList &params = kernel_decl->get_parameterList()->get_args();
  int dim = 1;
  BOOST_FOREACH (SgInitializedName *param, params) {
    SgType *ty = param->get_type();
    if (!(si::isConstType(ty) && isSgTypeInt(ty))) {
      // No more index param
      break;
    }
    StencilIndexVarAttribute *attr = new StencilIndexVarAttribute(dim);
    ++dim;
    vector<SgVarRefExp*> vars = si::querySubTree<SgVarRefExp>(kernel_decl);
    BOOST_FOREACH (SgVarRefExp *vr, vars) {
      if (si::convertRefToInitializedName(vr) != param) continue;
      rose_util::AddASTAttribute<StencilIndexVarAttribute>(vr, attr);
    }
  }
}

void TranslationContext::AnalyzeKernelFunctions(void) {
  LOG_DEBUG() << "Analyzing kernel functions\n";
  /*
   * NOTE: Using call graphs would be much simpler. analyzeCallToMap
   * can be simply extended so that after finding calls to map, it
   * can simply traverses reachable functions from the kernel
   * detected by using the call graph. However, the robustness of
   * graph generation is not clear: running CallGraphBuild on
   * sketch3d_map1.c generates several errors, though it still seems
   * to work.
   */
  SgNodePtrList funcCalls =
      NodeQuery::querySubTree(project_, V_SgFunctionCallExp);
  bool done = false;
  while (!done) {
    done = true;
    FOREACH(it, funcCalls.begin(), funcCalls.end()) {
      SgFunctionCallExp *call = isSgFunctionCallExp(*it);
      assert(call);
      StencilMap *m = findMap(call);
      if (m) {
        if (registerEntryKernel(m->getKernel())) {
          done = false;
          //AnalyzeEmit(m->getKernel());
          //AnalyzeGet(m->getKernel());
          AttachStencilIndexVarAttribute(m->getKernel());          
        }
        // For some reason, attribute attached to kernel body causes
        //assertion error when the body is inlined and its attribute
        //is queried. --> This is because KernelBodyAttribute lacked
        //the copy inherited method.
        EnsureKernelBodyAttribute(m->getKernel());
        continue;
      }

      // if the call site is a kernel, then mark the callee as an
      // inner kernel
      SgFunctionDeclaration *callerFunc
          = rose_util::getContainingFunction(call);
      assert(callerFunc);
      Kernel *parentKernel = findKernel(callerFunc);
      if (!parentKernel) continue;

      // Intrinsic functions are special functions.
      if (Grid::IsIntrinsicCall(call)) continue;

      SgFunctionRefExp *calleeExp =
          isSgFunctionRefExp(call->get_function());
      if (!calleeExp) {
        LOG_ERROR() << "Not supported function call: "
                    << call->get_function()->unparseToString()
                    << "\n";
        PSAbort(1);
      }

      SgFunctionDeclaration *calleeFunc
          = rose_util::getFuncDeclFromFuncRef(calleeExp);
      assert(calleeFunc);
      if (IsBuiltInFunction(calleeFunc->get_name())) {
        // builtins are just passed to the runtime
        continue;
      }

      if (registerInnerKernel(calleeFunc, call, parentKernel)) {
        done = false;
      }
    }
  }
  LOG_DEBUG() << "Kernel function analysis done\n";  
}

void TranslationContext::AnalyzeRun() {
  LOG_DEBUG() << "Analyzing Stencil run\n";
  if (ru::IsFortranLikeLanguage()) {
    LOG_DEBUG() << "No analysis for run needed in Fortran\n";
    return;
  }
  vector<SgFunctionCallExp*> calls =
      si::querySubTree<SgFunctionCallExp>(project_);
  BOOST_FOREACH(SgFunctionCallExp *call, calls) {
    if (!Run::isRun(call)) continue;
    LOG_DEBUG() << "Call to stencil_run found: "
                << call->unparseToString() << "\n";
    Run *x = new Run(call, this);
    registerRun(call, x);
  }
  LOG_DEBUG() << "Stencil run analysis done\n";
}

SgFunctionCallExpPtrList
TranslationContext::getGridCalls(SgScopeStatement *scope,
                                 string methodName) {
  SgNodePtrList calls = NodeQuery::querySubTree(scope, V_SgFunctionCallExp);
  SgFunctionCallExpPtrList targets;
  FOREACH(it, calls.begin(), calls.end()) {
    SgFunctionCallExp *call = isSgFunctionCallExp(*it);
    if (!GridType::isGridTypeSpecificCall(call)) continue;
    SgInitializedName* gv = GridType::getGridVarUsedInFuncCall(call);
    assert(gv);
    if (methodName != GridType::GetGridFuncName(call)) continue;
    // Now find a get call
    targets.push_back(call);
  }
  return targets;
}

SgFunctionCallExpPtrList
TranslationContext::getGridEmitCalls(SgScopeStatement *scope) {
  return getGridCalls(scope, "emit");
}

SgFunctionCallExpPtrList
TranslationContext::getGridGetCalls(SgScopeStatement *scope) {
  return getGridCalls(scope, "get");
}

SgFunctionCallExpPtrList
TranslationContext::getGridGetPeriodicCalls(SgScopeStatement *scope) {
  return getGridCalls(scope, "get_periodic");
}

void TranslationContext::print(ostream &os) const {
  os << "TranslationContext:\n";
  os << "Var-Grid associations:\n";
  FOREACH(gvit, grid_var_map_.begin(), grid_var_map_.end()) {
    const SgInitializedName *v = gvit->first;
    const GridSet &gs = gvit->second;
    StringJoin sj(", ");
    os << v->get_name().getString()
       << "@" << v->get_file_info()->get_line()
       << " -> ";
    FOREACH(git, gs.begin(), gs.end()) {
      Grid *g = *git;
      if (g) {
        sj << (*git)->toString();
      } else {
        sj << "NULL";
      }
    }
    os << sj << "\n";
  }
  // Map calls
  os << "Stencils:\n";
  FOREACH(it, stencil_map_.begin(), stencil_map_.end()) {
    const SgExpression *e = it->first;
    const StencilMap *s = it->second;
    os << e->unparseToString() << " -> "
       << s->toString() << "\n";
  }
}
#ifdef UNUSED_CODE
void TranslationContext::markReadWriteGrids() {
  FOREACH(it, grid_new_map_.begin(), grid_new_map_.end()) {
    Grid *g = it->second;
    FOREACH(kit, entry_kernels_.begin(), entry_kernels_.end()) {
      Kernel *k = kit->second;
      if (!(k->IsGridUnmodified(g) || k->IsGridUnread(g))) {
        g->setReadWrite(true);
        LOG_DEBUG() << g->toString()
                    << " is read and modified in a single update.\n";
        break;
      }
    }
  }
}
#endif

bool TranslationContext::IsInit(SgFunctionCallExp *call) const {
  if (!isSgFunctionRefExp(call->get_function())) return false;
  const string &fname = rose_util::getFuncName(call);
  return fname == PS_INIT_NAME;
}

const GridSet *TranslationContext::findGrid(SgInitializedName *var) const {
  GridVarMap::const_iterator it = grid_var_map_.find(var);
  if (it == grid_var_map_.end()) {
    return NULL;
  } else {
    return &(it->second);
  }
}


Grid *TranslationContext::getOrCreateGrid(SgFunctionCallExp *newCall) {
  Grid *g;
  if (isContained(grid_new_map_, newCall)) {
    g = grid_new_map_[newCall];
  } else {
    GridType *gt = NULL;
    if (si::is_C_language() || si::is_Cxx_language()) {
      gt = rose_util::GetASTAttribute<GridType>(
          newCall->get_type());
    } else if (si::is_Fortran_language()) {
      gt = rose_util::GetASTAttribute<GridType>(
          newCall->get_args()->get_expressions()[0]->get_type());
    }
    assert(gt);
    g = new Grid(gt, newCall);
    grid_new_map_.insert(std::make_pair(newCall, g));
  }
  return g;
}

Grid *TranslationContext::findGrid(SgFunctionCallExp *newCall) {
  if (isContained(grid_new_map_, newCall)) {
    return grid_new_map_[newCall];
  } else {
    return NULL;
  }
}

bool TranslationContext::registerInnerKernel(SgFunctionDeclaration *fd,
                                             SgFunctionCallExp *call,
                                             Kernel *parentKernel) {
  if (isInnerKernel(fd)) return false;
  LOG_DEBUG() << "Inner kernel: "
              << fd->get_name().getString() << "\n";
  Kernel *k = new Kernel(fd, this, parentKernel);
  parentKernel->appendChild(call, k);
  inner_kernels_.insert(std::make_pair(fd, k));
  return true;
}

bool TranslationContext::registerEntryKernel(SgFunctionDeclaration *fd) {
  if (isEntryKernel(fd)) return false;
  LOG_DEBUG() << "Entry kernel: "
              << fd->get_name().getString() << "\n";
  entry_kernels_.insert(std::make_pair(fd, new Kernel(fd, this)));
  return true;
}

bool TranslationContext::isEntryKernel(SgFunctionDeclaration *fd) {
  return entry_kernels_.find(fd) != entry_kernels_.end();
}

bool TranslationContext::isInnerKernel(SgFunctionDeclaration *fd) {
  return inner_kernels_.find(fd) != inner_kernels_.end();
}

bool TranslationContext::isKernel(SgFunctionDeclaration *fd) {
  return isEntryKernel(fd) || isInnerKernel(fd);
}

Kernel *TranslationContext::findKernel(SgFunctionDeclaration *fd) {
  KernelMap::iterator it = entry_kernels_.find(fd);
  if (it != entry_kernels_.end()) {
    return it->second;
  }
  KernelMap::iterator it2 = inner_kernels_.find(fd);
  if (it2 != inner_kernels_.end()) {
    return it2->second;
  }
  return NULL;
}

StencilMap *TranslationContext::findMap(SgExpression *e) {
  MapCallMap::iterator it = stencil_map_.find(e);
  if (it == stencil_map_.end()) {
    return NULL;
  } else {
    return it->second;
  }
}

void TranslationContext::AnalyzeReduce() {
  LOG_INFO() << "Analyzing reductions\n";
  Rose_STL_Container<SgNode*> calls =
      NodeQuery::querySubTree(project_, V_SgFunctionCallExp);
  FOREACH(it, calls.begin(), calls.end()) {
    SgFunctionCallExp *call = isSgFunctionCallExp(*it);
    assert(call);
    if (!Reduce::IsReduce(call)) continue;
    LOG_DEBUG() << "Call to reduce found: "
                << call->unparseToString() << "\n";
    Reduce *rd = new Reduce(call);
    rose_util::AddASTAttribute(call, rd);
  }
  LOG_INFO() << "Reduction analysis done.\n";  
}

} // namespace translator
} // namespace physis
