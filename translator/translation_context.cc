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

#include "translator/rose_util.h"
#include "translator/def_analysis.h"
#include "translator/stencil_analysis.h"

using std::vector;
using std::auto_ptr;
using std::ostream;

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

void TranslationContext::analyzeGridTypes() {
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
    
    //LOG_DEBUG() << "type name: " << typeName << "\n";

    if (!GridType::isGridType((typeName))) continue;

    SgPointerType *gridPtr = type->get_ptr_to();

    SgTypedefType *utype =
        isSgTypedefType(*(gridPtr->get_typedefs()->get_typedefs().begin()));
    const string utypeName = utype->get_name().getString();

    LOG_DEBUG() << "Grid type found: " << utypeName
                << " (" << typeName
                << ")\n";

    GridType *gt = new GridType(type, utype);
    LOG_DEBUG() << gt->toString() << "\n";
    registerGridType(utype, gt);
  }
}

static bool handleGridNew(SgInitializedName *in,
                          SgFunctionCallExp *callExp,
                          TranslationContext &tx) {
  Grid *g = tx.getOrCreateGrid(callExp);
  return tx.associateVarWithGrid(in, g);
}

static bool handleGridAlias(SgInitializedName *dst,
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

static bool handleGridAssignment(SgInitializedName *lhs_in,
                                 SgExpression *rhs_exp,
                                 TranslationContext &tx) {
  // rhs is either a grid variable or a call to grid new
  if (isSgVarRefExp(rhs_exp)) {
    SgVarRefExp *rhs_vref = isSgVarRefExp(rhs_exp);
    return handleGridAlias(lhs_in, 
                           rose_util::getInitializedName(rhs_vref), tx);
  } else if (isSgFunctionCallExp(rhs_exp)) {
    return handleGridNew(lhs_in, isSgFunctionCallExp(rhs_exp), tx);
  } else {
    // error
    LOG_ERROR() << "Invalid rhs: " << rhs_exp->unparseToString() << "\n";
    PSAbort(1);
    return false;
  }
}

static bool handleGridDeclaration(SgInitializedName *in,
                                  TranslationContext &tx,
                                  DefUseAnalysis &dua) {
  // just passing in as the first argument doesn't return the RHS of
  // the defining statement
  vector<SgNode*> defs = dua.getDefFor(in->get_declaration(), in);

  bool changed = false;
  FOREACH(it, defs.begin(), defs.end()) {
    SgNode *node = *it;
    LOG_DEBUG() << "declaration: "
                << in->unparseToString()
                << " <- " << node->unparseToString() << "\n";

    // When no rhs, in is returned
    if (node == in) {
      LOG_DEBUG() << "No definition\n";
      if (rose_util::isFuncParam(in)) {
        // in is a function parameter
        // this is handled at call sites
        continue;
      } else {
        changed |= tx.associateVarWithGrid(in, NULL);
        continue;
      }
    }

    // node is assumed to be of type SgAssignInitializer
    PSAssert(isSgAssignInitializer(node));
    SgExpression *rhs
        = isSgAssignInitializer(node)->get_operand();

#if 1    
    changed |= handleGridAssignment(in, rhs, tx);
#else
    // RHS can be either a call to grid_new or assignment of
    // another grid object handle
    if (isSgFunctionCallExp(rhs)) {
      changed |= handleGridNew(in, isSgFunctionCallExp(rhs), tx);
    } else if (isSgVarRefExp(rhs)) {
      SgInitializedName *src =
          rose_util::getInitializedName(isSgVarRefExp(rhs));
      changed |= handleGridAlias(in, src, tx);
    } else {
      // error
      LOG_ERROR() << "Unsupported: " << node->class_name()
                  << "\n";
      abort();
    }
#endif    
  }

  return changed;
}

static bool propagateGridVarMapAcrossCall(SgExpressionPtrList &args,
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
    SgInitializedName *src = rose_util::getInitializedName(gv);
    LOG_DEBUG() << "arg: " << arg->unparseToString()
                << " (" << src << ")"
                << ", param: " << param->unparseToString()
                << "(" << param << ")" << "\n";
    bool b = handleGridAlias(param, src, tx);
    LOG_DEBUG() << "Changed? " << b << "\n";
    changed |= b;
  }
  return changed;
}

static bool propagateGridVarMapAcrossStencilCall(SgFunctionCallExp *c,
                                                 StencilMap *m,
                                                 TranslationContext &tx) {
  LOG_DEBUG() << "StencilMap: " << m->getKernel()->get_name().str() << "\n";
  SgExpressionPtrList args(
      c->get_args()->get_expressions().begin() + 2,
      c->get_args()->get_expressions().end());
  SgInitializedNamePtrList params(
      m->getKernel()->get_args().begin() + m->getNumDim(),
      m->getKernel()->get_args().end());
  return propagateGridVarMapAcrossCall(args, params, tx);
}

static bool propagateGridVarMapAcrossOrdinaryCall(SgFunctionCallExp *c,
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

  // These are not taken care here.
  if (StencilMap::isMap(c) ||
      Reduce::IsReduce(c)) return false;
  
  LOG_DEBUG() << "Grid var propagation: " <<
      c->get_function()->unparseToString() << "\n";  
  
  SgFunctionDeclaration *callee_decl =
      si::getDeclarationOfNamedFunction(c->get_function());

  if (!callee_decl) {
    LOG_ERROR() << "Callee not found: " << c->unparseToString() << "\n";
    PSAbort(1);
  }
  
  callee_decl = isSgFunctionDeclaration(
      callee_decl->get_definingDeclaration());
  PSAssert(callee_decl);
  
  SgInitializedNamePtrList &params = callee_decl->get_args();
  return propagateGridVarMapAcrossCall(args, params, tx);
}


void TranslationContext::analyzeGridVars(DefUseAnalysis &dua) {
  SgNodePtrList vars =
      NodeQuery::querySubTree(project_, V_SgInitializedName);
  SgNodePtrList asns =
      NodeQuery::querySubTree(project_, V_SgAssignOp);
  SgNodePtrList calls =
      NodeQuery::querySubTree(project_, V_SgFunctionCallExp);
  bool changed = true;
  while (changed) {
    changed = false;
    FOREACH(it, vars.begin(), vars.end()) {
      SgInitializedName *in = isSgInitializedName(*it);
      SgType *type = in->get_type();
      GridType *gt = findGridType(type);
      if (gt == NULL) continue;
      LOG_DEBUG() << "Grid object of type "
                  << gt->toString() << " found: "
                  << in->unparseToString() << "\n";
      changed |= handleGridDeclaration(in, *this, dua);
    }
    FOREACH(it, asns.begin(), asns.end()) {
      SgAssignOp *aop = isSgAssignOp(*it);
      assert(aop);
      // Skip non-grid assignments
      SgType *type = aop->get_type();
      if (!findGridType(type)) continue;
      LOG_DEBUG() << "Grid assignment: "
                  << aop->unparseToString() << "\n";
      SgVarRefExp *lhs = isSgVarRefExp(aop->get_lhs_operand());
      assert(lhs);
      changed |= handleGridAssignment(rose_util::getInitializedName(lhs),
                                      aop->get_rhs_operand(),
                                      *this);
    }
    // Handle stencil map
    FOREACH(it, mapBegin(), mapEnd()) {
      SgFunctionCallExp *c = isSgFunctionCallExp(it->first);
      if (!c) continue;
      StencilMap *m = it->second;
      changed |= propagateGridVarMapAcrossStencilCall(c, m, *this);
    }
    
    // Note: grids are read-only in reduction kernels, so no need to
    // analyze them
    
    // Handle normal function call
    FOREACH (it, calls.begin(), calls.end()) {
      SgFunctionCallExp *c = isSgFunctionCallExp(*it);
      if (!c) continue;
      changed |= propagateGridVarMapAcrossOrdinaryCall(c, *this);
    }
  }
}

void TranslationContext::analyzeDomainExpr(DefUseAnalysis &dua) {
  assert(PS_MAX_DIM == 3);
  SgType* domTypes[] = {dom1d_type_, dom2d_type_, dom3d_type_};
  vector<SgType*> v(domTypes, domTypes+3);
  auto_ptr<DefMap> defMap = findDefinitions(project_, v);
  
  LOG_DEBUG() << "Domain DefMap: " << toString(*defMap);

  Rose_STL_Container<SgNode*> exps =
      NodeQuery::querySubTree(project_, V_SgExpression);
  FOREACH(it, exps.begin(), exps.end()) {
    SgExpression *exp = isSgExpression(*it);
    assert(exp);
    if (!Domain::isDomainType(exp->get_type())) continue;

    if (isSgVarRefExp(exp)) {
      SgInitializedName *in =
          rose_util::getInitializedName(isSgVarRefExp(exp));
      SgExpressionPtrList &defs = (*defMap)[in];
      FOREACH(defIt, defs.begin(), defs.end()) {
        SgExpression *def = *defIt;
        LOG_DEBUG() << "Dom Def: "
                    << def->unparseToString() << "\n";
        assert(isSgFunctionCallExp(def));
        Domain *dom =
            Domain::GetDomain(isSgFunctionCallExp(def));
        associateExpWithDomain(exp, dom);
      }

    } else if (isSgFunctionCallExp(exp)) {
      SgFunctionCallExp *call = isSgFunctionCallExp(exp);
      Domain *dom = Domain::GetDomain(call);
      associateExpWithDomain(call, dom);
    } else {
      // OG_DEBUG() << "Ignoring domain expression of type: "
      // < exp->class_name() << "\n";
    }
  }
}

void TranslationContext::analyzeMap() {
  Rose_STL_Container<SgNode*> calls =
      NodeQuery::querySubTree(project_, V_SgFunctionCallExp);
  FOREACH(it, calls.begin(), calls.end()) {
    SgFunctionCallExp *call = isSgFunctionCallExp(*it);
    assert(call);
    if (!StencilMap::isMap(call)) continue;
    LOG_DEBUG() << "Call to stencil_map found: "
                << call->unparseToString() << "\n";
    StencilMap *mc = new StencilMap(call, this);
    registerMap(call, mc);
  }
}

void TranslationContext::locateDomainTypes() {
  dom1d_type_ = rose_util::getType(project_, PSDOMAIN1D_TYPE_NAME);
  assert(dom1d_type_);
  dom2d_type_ = rose_util::getType(project_, PSDOMAIN2D_TYPE_NAME);
  assert(dom2d_type_);
  dom3d_type_ = rose_util::getType(project_, PSDOMAIN3D_TYPE_NAME);
  assert(dom3d_type_);
}


// TODO: Now all types are predefined.
void TranslationContext::build() {
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

  DefUseAnalysis *dua = new DefUseAnalysis(project_);
  dua->run(false);
  //dua->dfaToDOT();

  analyzeGridTypes();
  analyzeDomainExpr(*dua);
  analyzeMap();
  AnalyzeReduce();

  analyzeGridVars(*dua);
  LOG_DEBUG() << "grid variable analysis done\n";
  print(std::cout);

  analyzeRun(*dua);
  analyzeKernelFunctions();

  markReadWriteGrids();
  
  LOG_INFO() << "Analyzing stencil range\n";
  
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

bool TranslationContext::isNewCall(SgFunctionCallExp *ce) {
  SgFunctionRefExp *fref = isSgFunctionRefExp(ce->get_function());
  if (!fref) return false;
  SgFunctionSymbol *fs = fref->get_symbol();
  PSAssert(fs);
  const string name = fs->get_name().getString();
  // OG_DEBUG() << "new call? name: " << name << "\n";
  return isNewFunc(name);
}

static bool IsBuiltInFunction(const string &func_name) {
  // TODO: more complete support for builtin math functions
  string buildin_functions[] = {
    "exp", "expf", "log", "pow", "sqrt", "cos", "cosf",
    "sin", "sinf", "acos", "acosf"};
  int num_functions = sizeof(buildin_functions) / sizeof(string);
  LOG_DEBUG() << "num_functions: " << num_functions << "\n";
  return std::find(buildin_functions, buildin_functions + num_functions,
                   func_name) != buildin_functions + num_functions;
}

void TranslationContext::analyzeKernelFunctions(void) {
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
        }
        continue;
      }

      // if the call site is a kernel, then mark the callee as an
      // inner kernel
      SgFunctionDeclaration *callerFunc
          = rose_util::getContainingFunction(call);
      assert(callerFunc);
      Kernel *parentKernel = findKernel(callerFunc);
      if (!parentKernel) continue;

      // Grid functions are special functions.
      if (GridType::isGridCall(call)) continue;

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
}

/*
  bool TranslationContext::isGridTypeSpecificCall(SgFunctionCallExp *ce) {
  return getGridVarUsedInFuncCall(ce) != NULL;
  }

  SgInitializedName*
  TranslationContext::getGridVarUsedInFuncCall(SgFunctionCallExp *call) {
  SgPointerDerefExp *pdref =
  isSgPointerDerefExp(call->get_function());
  if (!pdref) return false;

  SgArrowExp *exp = isSgArrowExp(pdref->get_operand());
  if (!exp) return NULL;

  SgExpression *lhs = exp->get_lhs_operand();
  SgVarRefExp *vre = isSgVarRefExp(lhs);
  if (!vre) return NULL;

  SgInitializedName *in = vre->get_symbol()->get_declaration();
  assert(in);

  if (!GridType::isGridType(in->get_type())) return NULL;

  assert(findGrid(in));
  return in;
  }
*/

string TranslationContext::getGridFuncName(SgFunctionCallExp *call) {
  if (!GridType::isGridTypeSpecificCall(call)) {
    throw PhysisException("Not a grid function call");
  }
  SgPointerDerefExp *pdref =
      isSgPointerDerefExp(call->get_function());
  SgArrowExp *exp = isSgArrowExp(pdref->get_operand());

  SgVarRefExp *rhs = isSgVarRefExp(exp->get_rhs_operand());
  assert(rhs);
  const string name = rhs->get_symbol()->get_name().getString();
  LOG_VERBOSE() << "method name: " << name << "\n";

  return name;
}

void TranslationContext::analyzeRun(DefUseAnalysis &dua) {
  Rose_STL_Container<SgNode*> calls =
      NodeQuery::querySubTree(project_, V_SgFunctionCallExp);
  FOREACH(it, calls.begin(), calls.end()) {
    SgFunctionCallExp *call = isSgFunctionCallExp(*it);
    assert(call);
    if (!Run::isRun(call)) continue;
    LOG_DEBUG() << "Call to stencil_run found: "
                << call->unparseToString() << "\n";
    Run *x = new Run(call, this);
    registerRun(call, x);
  }
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
    if (methodName != getGridFuncName(call)) continue;
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
  os << "Exp-Dom associations:\n";
  FOREACH(domIt, domain_map_.begin(), domain_map_.end()) {
    const SgExpression *exp = domIt->first;
    const DomainSet &ds = domIt->second;
    StringJoin sj(", ");
    os << exp->unparseToString()
       << " -> ";
    FOREACH(dit, ds.begin(), ds.end()) {
      Domain *d = *dit;
      if (d) {
        sj << (*dit)->toString();
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

void TranslationContext::markReadWriteGrids() {
  FOREACH(it, grid_new_map_.begin(), grid_new_map_.end()) {
    Grid *g = it->second;
    FOREACH(kit, entry_kernels_.begin(), entry_kernels_.end()) {
      Kernel *k = kit->second;
      if (k->isModified(g) && k->isRead(g)) {
        g->setReadWrite(true);
        LOG_DEBUG() << g->toString()
                    << " is read and modified in a single update.\n";
        break;
      }
    }
  }
}

bool TranslationContext::IsInit(SgFunctionCallExp *call) const {
  const string &fname = rose_util::getFuncName(call);
  //LOG_DEBUG() << "func name: " << fname << "\n";
  return fname == "PSInit";
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
    GridType *gt = findGridType(newCall->get_type());
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

bool TranslationContext::associateExpWithDomain(SgExpression *exp,
                                                Domain *d) {
  return makeAssociation(exp, d, domain_map_);
}

const DomainSet *TranslationContext::findDomainAll(SgExpression *exp) {
  DomainMap::const_iterator it = domain_map_.find(exp);
  if (it == domain_map_.end()) {
    return NULL;
  } else {
    return &(it->second);
  }
}

Domain *TranslationContext::findDomain(SgExpression *exp) {
  DomainMap::const_iterator it = domain_map_.find(exp);
  if (it == domain_map_.end()) {
    return NULL;
  } else {
    return *(it->second.begin());
  }
}

StencilMap *TranslationContext::findMap(SgExpression *e) {
  MapCallMap::iterator it = stencil_map_.find(e);
  if (it == stencil_map_.end()) {
    return NULL;
  } else {
    return it->second;
  }
}

void TranslationContext::registerStencilIndex(SgFunctionCallExp *call,
                                              const StencilIndexList &sil) {
  stencil_indices_.insert(make_pair(call, sil));
}

const StencilIndexList* TranslationContext::findStencilIndex(
    SgFunctionCallExp *call) {
  StencilIndexMap::iterator it = stencil_indices_.find(call);
  if (it == stencil_indices_.end()) {
    return NULL;
  } else {
    return &(it->second);
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
    call->addNewAttribute(Reduce::name, rd);
  }
  LOG_INFO() << "Reduction analysis done.\n";  
}

Reduce *TranslationContext::GetReduce(SgFunctionCallExp *call) const {
  Reduce *rd = static_cast<Reduce*>(call->getAttribute(Reduce::name));
  return rd;
}


} // namespace translator
} // namespace physis
