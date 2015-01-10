// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/reference_translator.h"

#include <float.h> /* for FLT_MAX */
#include <boost/foreach.hpp>

#include "translator/rose_util.h"
#include "translator/ast_processing.h"
#include "translator/translation_context.h"
#include "translator/reference_runtime_builder.h"
#include "translator/builder_interface.h"
#include "translator/physis_names.h"
#include "translator/rose_fortran.h"

namespace si = SageInterface;
namespace sb = SageBuilder;
namespace ru = physis::translator::rose_util;
namespace rf = physis::translator::rose_fortran;

namespace physis {
namespace translator {

ReferenceTranslator::ReferenceTranslator(const Configuration &config):
    Translator(config),
    flag_constant_grid_size_optimization_(true),
    validate_ast_(true),
    grid_create_name_("__PSGridNew") {
  target_specific_macro_ = "PHYSIS_REF";
}

ReferenceTranslator::~ReferenceTranslator() {
}

void ReferenceTranslator::Translate() {
  defineMacro(target_specific_macro_);
  
  FOREACH(it, tx_->gridTypeBegin(),
          tx_->gridTypeEnd()) {
#if 0    
    LOG_DEBUG() << "grid type";
    SgTypedefType *t = isSgTypedefType(it->first);
    assert(t);
    SgPointerType *userGridPtrType
        = isSgPointerType(t->get_base_type());
    SgClassType *userGridType
        = isSgClassType(userGridPtrType->get_base_type());
    assert(userGridType);
    SgClassDeclaration *userGridTypeDecl =
        isSgClassDeclaration(userGridType->get_declaration()
                             ->get_definingDeclaration());
    assert(userGridTypeDecl);

    const string gname = t->get_name().getString();
    SgTypedefDeclaration *td
        = sb::buildTypedefDeclaration(gname,
                                      grid_ptr_type_, global_scope_);
    //si::replaceStatement(t->get_declaration(), td);

    SgFunctionSymbol *fs
        = si::lookupFunctionSymbolInParentScopes(SgName(gname+"_new"),
                                                 global_scope_);
    si::removeStatement(fs->get_declaration());
#endif
    
  }

  ProcessUserDefinedPointType();

  DefineMapSpecificTypesAndFunctions();
  
  traverseBottomUp(project_);

  // In Fortran, the change after unparsing makes the stencil run
  // function is actually referenced from the map struct type, which
  // is not the case in the current ROSE AST. So running this function
  // over the AST is not allowed in Fortran.
  if (!ru::IsFortranLikeLanguage())
      rose_util::RemoveUnusedFunction(project_);
  
  FixAST();
  ValidateASTConsistency();
}

void ReferenceTranslator::FixAST() {
  if (!rose_util::IsFortranLikeLanguage()) {
    FixGridType();
    si::fixVariableReferences(project_);
  }
}

void ReferenceTranslator::ValidateASTConsistency() {
  if (validate_ast_) {
    // Run internal consistency tests on AST  
    LOG_INFO() << "Validating AST consistency.\n";
    AstTests::runAllTests(project_);
    LOG_INFO() << "Validated successfully.\n";
  } else {
    LOG_WARNING() <<
        "AST consistency is not checked, and may be invalid.\n";
  }
}

void ReferenceTranslator::SetUp(SgProject *project,
                                TranslationContext *context,
                                BuilderInterface *rt_builder) {
  Translator::SetUp(project, context, rt_builder);
}

void ReferenceTranslator::Finish() {
  Translator::Finish();
}

void ReferenceTranslator::Optimize() {
  if (flag_constant_grid_size_optimization_) {
    optimizeConstantSizedGrids();
  }
}

void ReferenceTranslator::optimizeConstantSizedGrids() {
#if NOT_SUPPORTED_YET
  SgNodePtrList func_calls =
      NodeQuery::querySubTree(project_, V_SgFunctionCallExp);
  FOREACH(it, func_calls.begin(), func_calls.end()) {
    SgFunctionCallExp *func_call = isSgFunctionCallExp(*it);
    ROSE_ASSERT(func_call);
    LOG_DEBUG() << "func call: " << func_call->unparseToString() << "\n";
    SgFunctionSymbol *func_symbol = func_call->getAssociatedFunctionSymbol();
    ROSE_ASSERT(func_symbol);

    if (func_symbol != grid_dim_get_func_) continue;
    SgExpression *grid_arg_exp =
        func_call->get_args()->get_expressions()[0];
    ROSE_ASSERT(grid_arg_exp);
    SgInitializedName* grid_var =
        si::convertRefToInitializedName(grid_arg_exp);
    ROSE_ASSERT(grid_var);
    const GridSet *grid_set = tx_->findGrid(grid_var);

    // TODO (tatsuo): grid_set shouldn't be NULL, namely grid_var must always
    // associated with Grids.
    if (grid_set) {
      break;
    }

    const Grid *grid = *(grid_set->begin());

    if (grid->hasStaticSize()) {
      unsigned int size = grid->getStaticSize()[i];
      SgExpression *size_exp = sb::buildIntVal(size);
      si::replaceExpression(func_call, size_exp);
    }
  }
#endif  
}

void ReferenceTranslator::TranslateKernelDeclaration(
    SgFunctionDeclaration *node) {
  SgFunctionModifier &modifier = node->get_functionModifier();
  modifier.setInline();
}

SgExprListExp *ReferenceTranslator::BuildNewArg(GridType *gt, Grid *g,
                                                SgVariableDeclaration *dim_decl,
                                                SgVariableDeclaration *type_info_decl) {
  SgExpression *type_info_exp = sb::buildAddressOfOp(Var(type_info_decl));
  SgExprListExp *new_args
      = sb::buildExprListExp(type_info_exp, Int(gt->rank()), Var(dim_decl));
  //SgExpression *attr = g->BuildAttributeExpr();
  //if (!attr) attr = sb::buildIntVal(0);
  //si::appendExpression(new_args, attr);
  appendNewArgExtra(new_args, g, dim_decl);
  return new_args;
}

void ReferenceTranslator::appendNewArgExtra(SgExprListExp *args,
                                            Grid *g,
                                            SgVariableDeclaration *dim_decl) {
  return;
}

void ReferenceTranslator::TranslateNew(SgFunctionCallExp *node,
                                       GridType *gt) {
  Grid *g = tx_->findGrid(node);
  PSAssert(g);

  SgExprListExp *dims = builder()->BuildSizeExprList(g);
  SgBasicBlock *tmpBlock = sb::buildBasicBlock();
  SgVariableDeclaration *dimDecl
      = sb::buildVariableDeclaration(
          "dims", ivec_type_,
          sb::buildAggregateInitializer(dims, ivec_type_),
          tmpBlock);
  si::appendStatement(dimDecl, tmpBlock);

  // TypeInfo
  SgStatementPtrList build_type_info_stmts;
  SgVariableDeclaration *type_info = builder()->BuildTypeInfo(gt, build_type_info_stmts);
  si::appendStatementList(build_type_info_stmts, tmpBlock);

  SgExprListExp *new_args = BuildNewArg(gt, g, dimDecl, type_info);

  SgFunctionSymbol *grid_new
      = si::lookupFunctionSymbolInParentScopes(grid_create_name_,
                                               global_scope_);
  
  // sb::build a call to grid_new
  SgFunctionCallExp *new_call
      = sb::buildFunctionCallExp(sb::buildFunctionRefExp(grid_new),
                                 new_args);
  //SgVariableDeclaration *curStmt
  //= isSgVariableDeclaration(getContainingStatement(node));  
  SgStatement  *curStmt  = getContainingStatement(node);
  assert(curStmt);
  si::insertStatementAfter(curStmt, tmpBlock);  
  SgExpression *grid_var = NULL;
  if (isSgVariableDeclaration(curStmt)) {
    SgVariableDeclaration *decl = isSgVariableDeclaration(curStmt);
    grid_var = sb::buildVarRefExp(decl);
    decl->reset_initializer(NULL);
  } else if (isSgExprStatement(curStmt)) {
    SgExprStatement *assn = isSgExprStatement(curStmt);    
    SgAssignOp *aop = isSgAssignOp(assn->get_expression());
    PSAssert(aop);
    grid_var = aop->get_lhs_operand();
    si::removeStatement(curStmt);    
  }
  si::appendStatement(
      sb::buildAssignStatement(grid_var, new_call),
      tmpBlock);
  return;
}

void ReferenceTranslator::TranslateGet(SgFunctionCallExp *node,
                                       SgInitializedName *gv,
                                       bool is_kernel,
                                       bool is_periodic) {
  /*
    (type*)(g->p0)[offset]
  */
  GridType *gt = tx_->findGridType(gv->get_type());
  const StencilIndexList *sil =
      rose_util::GetASTAttribute<GridGetAttribute>(node)->GetStencilIndexList();
  SgExpressionPtrList args;
  rose_util::CopyExpressionPtrList(
      node->get_args()->get_expressions(), args);
  SgExpression *p0 = builder()->BuildGridGet(
      sb::buildVarRefExp(gv->get_name(), si::getScope(node)),
      rose_util::GetASTAttribute<GridVarAttribute>(gv),
      gt,
      &args, sil, is_kernel, is_periodic);
  rose_util::GetASTAttribute<GridGetAttribute>(p0)->gv() = gv;  
  si::replaceExpression(node, p0);
}

void ReferenceTranslator::RemoveEmitDummyExp(SgExpression *emit) {
  // For EmitUtype, dummy pointer dereference needs to be removed,
  // i.e., (*(type *)emit_exp) -> emit_exp
  PSAssert(isSgCastExp(emit->get_parent()));
  SgPointerDerefExp *deref_exp = isSgPointerDerefExp(
      emit->get_parent()->get_parent());
  PSAssert(deref_exp);
  si::replaceExpression(deref_exp, si::copyExpression(emit));
}

void ReferenceTranslator::TranslateEmit(SgFunctionCallExp *node,
                                        GridEmitAttribute *attr) {
  bool is_grid_type_specific_call =
      GridType::isGridTypeSpecificCall(node);

  GridType *gt = attr->gt();
  int nd = gt->rank();
  SgInitializedNamePtrList &params = getContainingFunction(node)->get_args();
  SgExpressionPtrList args;
  for (int i = 0; i < nd; ++i) {
    SgInitializedName *p = params[i];
    args.push_back(sb::buildVarRefExp(p, si::getScope(node)));
  }

  SgExpression *emit_val =
      si::copyExpression(node->get_args()->get_expressions().back());

  SgExpression *emit =
      builder()->BuildGridEmit(sb::buildVarRefExp(attr->gv()),
                                 attr, &args, emit_val,
                                 si::getScope(node));

  si::replaceExpression(node, emit);
  
  if (!is_grid_type_specific_call) {
    RemoveEmitDummyExp(emit);
  }

}

static SgType *GetDomType(StencilMap *sm) {
  SgType *t = sm->getDom()->get_type();
  PSAssert(t);
  return t;
}

void ReferenceTranslator::TranslateMap(SgFunctionCallExp *node,
                                       StencilMap *stencil) {
  // gFunctionDeclaration *mapFunc = GenerateMap(stencil);
  // nsertStatement(getContainingFunction(node), mapFunc);

  SgFunctionDeclaration *realMap = stencil->getFunc();
  assert(realMap);
  // This doesn't work because the real map does not have
  // defining declarations
  // SgFunctionRefExp *ref = sb::buildFunctionRefExp(realMap);
  LOG_DEBUG() << "realmap: "
              << realMap->get_name() << "\n";
  SgFunctionRefExp *ref = rose_util::getFunctionRefExp(realMap);
  assert(ref);
  //LOG_DEBUG() << "Map function: " << ref->unparseToString() << "\n";

  SgExpressionPtrList &args = node->get_args()->get_expressions();
  SgExpressionPtrList::iterator args_it = args.begin();
  SgExprListExp *new_args = sb::buildExprListExp();
  if (ru::IsFortranLikeLanguage()) {
    si::appendExpression(new_args, si::copyExpression(*args_it));
    ++args_it;
  }
  // Skip the kernel argument
  ++args_it;
  FOREACH(it, args_it, args.end()) {
    si::appendExpression(new_args, si::copyExpression(*it));
  }
  si::replaceExpression(node,
                        sb::buildFunctionCallExp(ref, new_args));
}

void ReferenceTranslator::InsertStencilSpecificType(StencilMap *s,
                                                    SgClassDeclaration *type_decl) {
  if (ru::IsFortranLikeLanguage()) {
    // Assumes the kernel function is located in a module. Finds the
    // type declration part (before "contains" statement), and insert
    // the new type at the end of the pre-contains area
    SgClassDefinition *mod = isSgClassDefinition(s->getKernel()->get_parent());
    if (!mod) {
      LOG_ERROR() << "Unknown kernel parent node ("
                  << s->getKernel()->get_parent()->class_name() << "): "
                  << s->getKernel()->get_parent()->unparseToString() << "\n";
      PSAbort(1);
    }
    vector<SgContainsStatement*> csv = si::querySubTree<SgContainsStatement>(mod);
    if (csv.size() == 0) {
      LOG_ERROR() << "No \'contains\' statement in the module.\n";
      PSAbort(1);
    }
    SgContainsStatement *cs = csv[0];
    si::insertStatementBefore(cs, type_decl);
  } else {
    si::insertStatementBefore(s->getKernel(), type_decl);
  }
}

void ReferenceTranslator::InsertStencilSpecificFunc(StencilMap *s,
                                                    SgFunctionDeclaration *func) {
  si::insertStatementAfter(s->getKernel(), func);
}

void ReferenceTranslator::DefineMapSpecificTypesAndFunctions() {
  FOREACH(it, tx_->mapBegin(),
          tx_->mapEnd()) {
    StencilMap *s = it->second;
    // multiple stencil_map may call use the same kernel with the
    // same dimensionality
    SgType *t =
        si::lookupNamedTypeInParentScopes(s->GetTypeName(), global_scope_);
    if (t) {
      // necessary types and functions are already generated
      s->stencil_type() = isSgClassType(t);
      SgFunctionSymbol *mapSymbol
          = si::lookupFunctionSymbolInParentScopes(s->GetMapName(),
                                                   global_scope_);
      assert(mapSymbol);
      s->setFunc(mapSymbol->get_declaration());
      SgFunctionSymbol *runSymbol
          = si::lookupFunctionSymbolInParentScopes(s->GetRunName(),
                                                   global_scope_);
      assert(runSymbol);
      s->run() = runSymbol->get_declaration();

      SgFunctionSymbol *runInnerSymbol
          = si::lookupFunctionSymbolInParentScopes(
              s->GetRunName() + "_" + string(PS_STENCIL_MAP_INNER_SUFFIX_NAME),
              global_scope_);
      
      if (runInnerSymbol) {
        s->run_inner() = runInnerSymbol->get_declaration();
      }
      continue;
    }

    SgClassDeclaration *sm_type = builder()->BuildStencilMapType(s);
    InsertStencilSpecificType(s, sm_type);
    s->stencil_type() = sm_type->get_type();

    // define real stencil_map function
    SgFunctionDeclaration *realMap = builder()->BuildMap(s);
    assert(realMap);
    InsertStencilSpecificFunc(s, realMap);
    s->setFunc(realMap);
#if 1
    SgFunctionDeclaration *runKernel = builder()->BuildRunKernelFunc(s);
#else
    SgFunctionParameterList *pl =
        builder()->BuildRunKernelFuncParameterList(s);
    vector<SgVariableDeclaration*> indices;
    SgBasicBlock *body = builder()->BuildRunKernelFuncBody(s, pl, indices);
    SgFunctionDeclaration *runKernel =
        builder()->BuildRunKernelFunc(s, pl, body, indices);
#endif
    assert(runKernel);
    s->run() = runKernel;
    InsertStencilSpecificFunc(s, runKernel);

    SgFunctionDeclaration *runInnerKernel = BuildRunInteriorKernel(s);
    if (runInnerKernel) {
      s->run_inner() = runInnerKernel;
      InsertStencilSpecificFunc(s, runInnerKernel);
    }
    vector<SgFunctionDeclaration*> runBoundaryKernel =
        BuildRunBoundaryKernel(s);
    FOREACH (it, runBoundaryKernel.begin(), runBoundaryKernel.end()) {
      InsertStencilSpecificFunc(s, *it);
    }
  }
}

/** generate dlopen and dlsym code
 * @param[in] run
 * @param[in] ref ... function reference  '__PSStencilRun_0(1, ...);'
 * @param[in] index ... VAR's expression
 * @param[in] scope
 * @return    statement like below
 *  if (l == VAR / N) {
 *    l = VAR / N;
 *    if (handle) dlclose(handle);
 *    <<< AddSyncAfterDlclose(...) >>>
 *    handle = dlopen(__dl_fname[l], RTLD_NOW);
 *    if (!handle) { fprintf(stderr, "%s\n", dlerror()); PSAbort(EXIT_FAILURE); }
 *    dlerror();
 *    *(void **)(&__PSStencilRun_0) = dlsym(handle, "__PSStencilRun_0");
 *    if ((error = dlerror()) != NULL) { fprintf(stderr, "%s\n", error); PSAbort(EXIT_FAILURE); }
 *  }
 */
SgStatement *ReferenceTranslator::GenerateDlopenDlsym(
    Run *run, SgFunctionRefExp *ref,
    SgExpression *index, SgScopeStatement *scope) {
  SgBasicBlock *if_true = sb::buildBasicBlock();
  SgExpression *l_exp = index;
  if (config_.ndynamic() > 1) {
    l_exp = sb::buildIntegerDivideOp(
        l_exp, sb::buildIntVal(config_.ndynamic()));
  }
  /* l = VAR / N; */
  si::appendStatement(
      sb::buildAssignStatement(sb::buildVarRefExp("l", scope), l_exp),
      if_true);
  /* if (handle) dlclose(handle); */
  si::appendStatement(
      sb::buildIfStmt(
          sb::buildVarRefExp("handle", scope),
          sb::buildExprStatement(
              sb::buildFunctionCallExp(
                  sb::buildFunctionRefExp("dlclose"),
                  sb::buildExprListExp(
                      sb::buildVarRefExp("handle", scope)))),
          NULL),
      if_true);
  builder()->AddSyncAfterDlclose(if_true);
  /* handle = dlopen(__dl_fname[l], RTLD_NOW); */
  si::appendStatement(
      sb::buildAssignStatement(
          sb::buildVarRefExp("handle", scope),
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("dlopen"),
              sb::buildExprListExp(
                  sb::buildPntrArrRefExp(
                      sb::buildVarRefExp(
                          sb::buildVariableDeclaration(
                              "__dl_fname",
                              sb::buildArrayType(
                                  sb::buildPointerType(
                                      sb::buildCharType())))),
                      sb::buildVarRefExp("l", scope)),
                  sb::buildOpaqueVarRefExp("RTLD_NOW")
                  //sb::buildIntVal(RTLD_NOW)
                  ))),
      if_true);
  /* if (!handle) { fprintf(stderr, "%s\n", dlerror()); PSAbort(EXIT_FAILURE); } */
  si::appendStatement(
      sb::buildIfStmt(
          sb::buildNotOp(sb::buildVarRefExp("handle", scope)),
          sb::buildBasicBlock(
              sb::buildExprStatement(
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("fprintf"),
                      sb::buildExprListExp(
                          sb::buildOpaqueVarRefExp("stderr"),
                          sb::buildStringVal("%s\\n"),
                          sb::buildFunctionCallExp(
                              sb::buildFunctionRefExp("dlerror"), NULL)))),
              sb::buildExprStatement(
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("PSAbort"),
                      sb::buildExprListExp(
                          sb::buildIntVal(EXIT_FAILURE))))),
          NULL),
      if_true);
  /* dlerror(); */
  si::appendStatement(
      sb::buildExprStatement(
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("dlerror"), NULL)),
      if_true);
  /* all stencils */
  SgFunctionSymbol *func_sym = ref->get_symbol();
  PSAssert(func_sym);
  SgName fname = func_sym->get_name();
  /* *(void **)(&__PSStencilRun_0) = dlsym(handle, "__PSStencilRun_0"); */
  si::appendStatement(
      sb::buildAssignStatement(
          sb::buildPointerDerefExp(
              sb::buildCastExp(
                  sb::buildAddressOfOp(
                      sb::buildOpaqueVarRefExp(
                          fname.getString() + " ")), /* XXX: fake */
                  sb::buildPointerType(
                      sb::buildPointerType(sb::buildVoidType())))),
          sb::buildFunctionCallExp(
              sb::buildFunctionRefExp("dlsym"),
              sb::buildExprListExp(
                  sb::buildVarRefExp("handle", scope),
                  sb::buildStringVal(fname)))),
      if_true);
  /* if ((error = dlerror()) != NULL) { fprintf(stderr, "%s\n", error); PSAbort(EXIT_FAILURE); } */
  si::appendStatement(
      sb::buildIfStmt(
          sb::buildAssignOp(
              sb::buildVarRefExp("error", scope),
              sb::buildFunctionCallExp(
                  sb::buildFunctionRefExp("dlerror"), NULL)),
          sb::buildBasicBlock(
              sb::buildExprStatement(
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("fprintf"),
                      sb::buildExprListExp(
                          sb::buildOpaqueVarRefExp("stderr"),
                          sb::buildStringVal("%s\\n"),
                          sb::buildVarRefExp("error", scope)))),
              sb::buildExprStatement(
                  sb::buildFunctionCallExp(
                      sb::buildFunctionRefExp("PSAbort"),
                      sb::buildExprListExp(
                          sb::buildIntVal(EXIT_FAILURE))))),
          NULL),
      if_true);

  /* if (l != VAR / N) */
  SgStatement *stmt =
      sb::buildIfStmt(
          sb::buildNotEqualOp(sb::buildVarRefExp("l", scope), l_exp),
          if_true, NULL);
  return stmt;
}
/** generate trial code
 * @param[in] run
 * @param[in] ref ... function reference  '__PSStencilRun_0(1, ...);'
 * @return    function declaration like below
 *  static void __PSStencilRun_N_trial(int iter, ...) {
 *    static int count = 0;
 *    static int mindex = 0;
 *    static float mtime = FLT_MAX;
 *    static void *r = __PSRandomInit(N); // for random
 *    void *handle = NULL;
 *    char *error;
 *    int l = -1;
 *    while (count < T && iter > 0) {
 *      float time;
 *      int index = __PSRandom(r, count); // for random
 *      int a = index % N;
 *      <<< GenerateDlopenDlsym(...) >>>
 *      time = __PSStencilRun_0(1, ...);
 *      if (mtime > time) { mtime = time; mindex = index; }
 *      ++count;
 *      --iter;
 *      if (count >= T) __PSRandomFini(r); // for random
 *    }
 *    if (iter > 0) {
 *      int a = mindex % N;
 *      <<< GenerateDlopenDlsym(...) >>>
 *      __PSStencilRun_0(iter, ...);
 *    }
 *    if (handle) dlclose(handle);
 *    <<< AddSyncAfterDlclose(...) >>>
 *  }
 */
SgFunctionDeclaration *ReferenceTranslator::GenerateTrial(
    Run *run, SgFunctionRefExp *ref) {
  // setup the parameter list
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  si::appendArg(parlist, sb::buildInitializedName("iter",
                                                  sb::buildIntType()));

  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *stencil = it->second;
    //SgType *stencilType = sb::buildPointerType(stencil->getType());
    SgType *stencilType = stencil->stencil_type();
    si::appendArg(parlist,
                  sb::buildInitializedName("s" + toString(i),
                                           stencilType));
  }

  // Declare and define the function
  SgFunctionDeclaration *trialFunc =
      sb::buildDefiningFunctionDeclaration(run->GetName() + "_trial",
                                           sb::buildVoidType(),
                                           parlist);
  rose_util::SetFunctionStatic(trialFunc);

  // Function body
  SgFunctionDefinition *fdef = trialFunc->get_definition();
  SgDeclarationStatement *vd;
  SgExprListExp *args;
  SgBasicBlock *funcBlock = sb::buildBasicBlock();
  /* static int count = 0; */
  si::setStatic(
      vd = sb::buildVariableDeclaration(
          "count", sb::buildIntType(),
          sb::buildAssignInitializer(sb::buildIntVal(0)), funcBlock));
  si::appendStatement(vd, funcBlock);
  /* static int mindex = 0; */
  si::setStatic(
      vd = sb::buildVariableDeclaration(
          "mindex", sb::buildIntType(),
          sb::buildAssignInitializer(sb::buildIntVal(0)), funcBlock));
  si::appendStatement(vd, funcBlock);
  /* static float mtime = FLT_MAX; */
  si::setStatic(
      vd = sb::buildVariableDeclaration(
          "mtime", sb::buildFloatType(),
          //sb::buildAssignInitializer(sb::buildOpaqueVarRefExp("FLT_MAX")),
          sb::buildAssignInitializer(sb::buildFloatVal(FLT_MAX)),
          funcBlock));
  si::appendStatement(vd, funcBlock);
  int trial_times = config_.npattern() * config_.ndynamic();
  bool random_flag = false;
  double val;
  if (config_.Lookup("AT_TRIAL_TIMES", val) &&
      (int)val > 0 && trial_times > (int)val) {
    /* static void *r = __PSRandomInit(N); */
    si::setStatic(
        vd = sb::buildVariableDeclaration(
            "r", sb::buildPointerType(sb::buildVoidType()),
            sb::buildAssignInitializer(
                sb::buildFunctionCallExp(
                    sb::buildFunctionRefExp("__PSRandomInit"),
                    sb::buildExprListExp(sb::buildIntVal(trial_times)))),
            funcBlock));
    si::appendStatement(vd, funcBlock);
    random_flag = true;
    trial_times = (int)val;
  }

  if (config_.npattern() > 1) {
    /* void *handle = NULL; */
    si::appendStatement(
        sb::buildVariableDeclaration(
            "handle", sb::buildPointerType(sb::buildVoidType()),
            //sb::buildAssignInitializer(sb::buildOpaqueVarRefExp("NULL")),
            sb::buildAssignInitializer(sb::buildIntVal(NULL)),
            funcBlock),
        funcBlock);
    /* char *error; */
    si::appendStatement(
        sb::buildVariableDeclaration(
            "error", sb::buildPointerType(sb::buildCharType()), NULL,
            funcBlock),
        funcBlock);
    /* int l = -1; */
    si::appendStatement(
        sb::buildVariableDeclaration(
            "l", sb::buildIntType(),
            sb::buildAssignInitializer(sb::buildIntVal(-1)), funcBlock),
        funcBlock);
  }

  SgBasicBlock *while_body = sb::buildBasicBlock();
  /* float time; */
  si::appendStatement(
      sb::buildVariableDeclaration(
          "time", sb::buildFloatType(), NULL, while_body),
      while_body);
  if (random_flag) {
    /* int index = __PSRandom(r, count); */
    si::appendStatement(
        sb::buildVariableDeclaration(
            "index", sb::buildIntType(),
            sb::buildAssignInitializer(
                sb::buildFunctionCallExp(
                    sb::buildFunctionRefExp("__PSRandom"),
                    sb::buildExprListExp(
                        sb::buildVarRefExp("r", funcBlock),
                        sb::buildVarRefExp("count", funcBlock)))),
            while_body),
        while_body);
  } else {
    /* int index = count; */
    si::appendStatement(
        sb::buildVariableDeclaration(
            "index", sb::buildIntType(),
            sb::buildAssignInitializer(
                sb::buildVarRefExp("count", funcBlock)),
            while_body),
        while_body);
  }
  /* int a = index % N; */
  if (config_.ndynamic() > 1) {
    SgExpression *e = sb::buildVarRefExp("index", while_body);
    if (config_.npattern() > 1) {
      e = sb::buildModOp(e, sb::buildIntVal(config_.ndynamic()));
    }
    si::appendStatement(
        sb::buildVariableDeclaration(
            "a", sb::buildIntType(), sb::buildAssignInitializer(e)),
        while_body);
  }
  /* dlopen(), dlsym() */
  if (config_.npattern() > 1) {
    si::appendStatement(
        GenerateDlopenDlsym(
            run, ref, sb::buildVarRefExp("index", while_body), funcBlock),
        while_body);
  }
  /* time = __PSStencilRun_0(1, ...); */
  args = sb::buildExprListExp(sb::buildIntVal(1));
  if (config_.ndynamic() > 1) {
    builder()->AddDynamicArgument(args, sb::buildVarRefExp("a", while_body));
  }
  for (unsigned int i = 0; i < run->stencils().size(); ++i) {
    si::appendExpression(args, sb::buildVarRefExp("s" + toString(i)));
  }
  si::appendStatement(
      sb::buildAssignStatement(
          sb::buildVarRefExp("time", while_body),
          sb::buildFunctionCallExp(ref, args)),
      while_body);
#if 1 /* debug */
  if (config_.ndynamic() > 1) {
    /* fprintf(stderr, "; __PSStencilRun_0_trial: [%05d:%d] time = %f\n", , index/N, index%N, time); */
    si::appendStatement(
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("fprintf"),
                sb::buildExprListExp(
                    sb::buildOpaqueVarRefExp("stderr"),
                    sb::buildStringVal("; " + run->GetName() + "_trial: [%05d:%d] time = %f\\n"),
                    sb::buildIntegerDivideOp(
                        sb::buildVarRefExp("index", funcBlock),
                        sb::buildIntVal(config_.ndynamic())),
                    sb::buildModOp(
                        sb::buildVarRefExp("index", funcBlock),
                        sb::buildIntVal(config_.ndynamic())),
                    sb::buildVarRefExp("time", while_body)))),
        while_body);
  } else {
    /* fprintf(stderr, "; __PSStencilRun_0_trial: [%05d] time = %f\n", index, time); */
    si::appendStatement(
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("fprintf"),
                sb::buildExprListExp(
                    sb::buildOpaqueVarRefExp("stderr"),
                    sb::buildStringVal("; " + run->GetName() + "_trial: [%05d] time = %f\\n"),
                    sb::buildVarRefExp("index", funcBlock),
                    sb::buildVarRefExp("time", while_body)))),
        while_body);
  }
#endif
  /* if (mtime > time) { mtime = time; mindex = index; } */
  si::appendStatement(
      sb::buildIfStmt(
          sb::buildGreaterThanOp(
              sb::buildVarRefExp("mtime", funcBlock),
              sb::buildVarRefExp("time", while_body)),
          sb::buildBasicBlock(
              sb::buildAssignStatement(
                  sb::buildVarRefExp("mtime", funcBlock),
                  sb::buildVarRefExp("time", while_body)),
              sb::buildAssignStatement(
                  sb::buildVarRefExp("mindex", funcBlock),
                  sb::buildVarRefExp("index", while_body))),
          NULL),
      while_body);
  /* ++count; */
  si::appendStatement(
      sb::buildExprStatement(
          sb::buildPlusPlusOp(
              sb::buildVarRefExp("count", funcBlock), SgUnaryOp::prefix)),
      while_body);
  /* --iter; */
  si::appendStatement(
      sb::buildExprStatement(
          sb::buildMinusMinusOp(
              sb::buildVarRefExp("iter", funcBlock), SgUnaryOp::prefix)),
      while_body);
  if (random_flag) {
    /* if (count >= T) __PSRandomFini(r); */
    si::appendStatement(
        sb::buildIfStmt(
            sb::buildGreaterOrEqualOp(
                sb::buildVarRefExp("count", funcBlock),
                sb::buildIntVal(trial_times)),
            sb::buildExprStatement(
                sb::buildFunctionCallExp(
                    sb::buildFunctionRefExp("__PSRandomFini"),
                    sb::buildExprListExp(sb::buildVarRefExp("r", funcBlock)))),
            NULL),
        while_body);
  }
  /* while (count < T && iter > 0) */
  si::appendStatement(
      sb::buildWhileStmt(
          sb::buildAndOp(
              sb::buildLessThanOp(
                  sb::buildVarRefExp("count", funcBlock),
                  sb::buildIntVal(trial_times)),
              sb::buildGreaterThanOp(
                  sb::buildVarRefExp("iter", funcBlock),
                  sb::buildIntVal(0))),
          while_body),
      funcBlock);

  SgBasicBlock *if_true = sb::buildBasicBlock();
#if 1 /* debug */
  if (config_.ndynamic() > 1) {
    /* fprintf(stderr, "; __PSStencilRun_0_trial: [%05d:%d] mtime = %f\n", mindex/N, mindex%N, mtime); */
    si::appendStatement(
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("fprintf"),
                sb::buildExprListExp(
                    sb::buildOpaqueVarRefExp("stderr"),
                    sb::buildStringVal("; " + run->GetName() + "_trial: [%05d:%d] mtime = %f\\n"),
                    sb::buildIntegerDivideOp(
                        sb::buildVarRefExp("mindex", funcBlock),
                        sb::buildIntVal(config_.ndynamic())),
                    sb::buildModOp(
                        sb::buildVarRefExp("mindex", funcBlock),
                        sb::buildIntVal(config_.ndynamic())),
                    sb::buildVarRefExp("mtime", funcBlock)))),
        if_true);
  } else {
    /* fprintf(stderr, "; __PSStencilRun_0_trial: [%05d] mtime = %f\n", mindex, mtime); */
    si::appendStatement(
        sb::buildExprStatement(
            sb::buildFunctionCallExp(
                sb::buildFunctionRefExp("fprintf"),
                sb::buildExprListExp(
                    sb::buildOpaqueVarRefExp("stderr"),
                    sb::buildStringVal("; " + run->GetName() + "_trial: [%05d] mtime = %f\\n"),
                    sb::buildVarRefExp("mindex", funcBlock),
                    sb::buildVarRefExp("mtime", funcBlock)))),
        if_true);
  }
#endif
  /* int a = mindex % N; */
  if (config_.ndynamic() > 1) {
    SgExpression *e = sb::buildVarRefExp("mindex", funcBlock);
    if (config_.npattern() > 1) {
      e = sb::buildModOp(e, sb::buildIntVal(config_.ndynamic()));
    }
    si::appendStatement(
        sb::buildVariableDeclaration(
            "a", sb::buildIntType(), sb::buildAssignInitializer(e)),
        if_true);
  }
  /* dlopen(), dlsym() */
  if (config_.npattern() > 1) {
    si::appendStatement(
        GenerateDlopenDlsym(
            run, ref, sb::buildVarRefExp("mindex", funcBlock), funcBlock),
        if_true);
  }
  /* __PSStencilRun_0(iter, ...); */
  args = sb::buildExprListExp(sb::buildVarRefExp("iter"));
  if (config_.ndynamic() > 1) {
    builder()->AddDynamicArgument(args, sb::buildVarRefExp("a", if_true));
  }
  for (unsigned int i = 0; i < run->stencils().size(); ++i) {
    si::appendExpression(args, sb::buildVarRefExp("s" + toString(i)));
  }
  si::appendStatement(
      sb::buildExprStatement(sb::buildFunctionCallExp(ref, args)),
      if_true);
  /* if (iter > 0) */
  si::appendStatement(
      sb::buildIfStmt(
          sb::buildGreaterThanOp(
              sb::buildVarRefExp("iter", funcBlock), sb::buildIntVal(0)),
          if_true, NULL),
      funcBlock);

  if (config_.npattern() > 1) {
    /* if (handle) dlclose(handle); */
    si::appendStatement(
        sb::buildIfStmt(
            sb::buildVarRefExp("handle", funcBlock),
            sb::buildExprStatement(
                sb::buildFunctionCallExp(
                    sb::buildFunctionRefExp("dlclose"),
                    sb::buildExprListExp(
                        sb::buildVarRefExp("handle", funcBlock)))),
            NULL),
        funcBlock);
    builder()->AddSyncAfterDlclose(funcBlock);
  }

  si::replaceStatement(fdef->get_body(), funcBlock);
  si::attachComment(trialFunc, "Generated by " + string(__FUNCTION__));
  return trialFunc;
}

void ReferenceTranslator::TranslateRun(SgFunctionCallExp *node,
                                       Run *run) {
  // No translation necessary for Fortran binding
  if (ru::IsFortranLikeLanguage()) {
    return;
  }
  
  SgFunctionDeclaration *runFunc = builder()->BuildRunFunc(run);
  si::insertStatementBefore(getContainingFunction(node), runFunc);
  SgFunctionRefExp *ref = rose_util::getFunctionRefExp(runFunc);

  /* auto tuning */
  if (config_.auto_tuning()) {
    rose_util::AddASTAttribute(runFunc, new RunKernelCallerAttribute());
    SgFunctionDeclaration *trialFunc = GenerateTrial(run, ref);
    si::insertStatementBefore(getContainingFunction(node), trialFunc);
    ref = rose_util::getFunctionRefExp(trialFunc);
  }

  SgExprListExp *args = sb::buildExprListExp();
  SgExpressionPtrList &original_args =
      node->get_args()->get_expressions();
  int num_remaining_args = original_args.size();
  // if iteration count is not specified, add 1 to the arg list
  if (!run->HasCount()) {
    si::appendExpression(args, sb::buildIntVal(1));
  } else {
    si::appendExpression(args, si::copyExpression(original_args.back()));
    --num_remaining_args;
  }
  for (int i = 0; i < num_remaining_args; ++i) {
    si::appendExpression(args, si::copyExpression(original_args.at(i)));
  }
  si::replaceExpression(node, sb::buildFunctionCallExp(ref, args));
}

// ReferenceRuntimeBuilder* ReferenceTranslator::GetRuntimeBuilder() const {
//   LOG_DEBUG() << "Using reference runtime builder\n";
//   return new ReferenceRuntimeBuilder();
// }


void ReferenceTranslator::TranslateSet(SgFunctionCallExp *node,
                                       SgInitializedName *gv) {
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->rank();
  SgScopeStatement *scope = getContainingScopeStatement(node);    
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  SgExpressionPtrList &
      args = node->get_args()->get_expressions();
  SgExpressionPtrList indices;
  FOREACH (it, args.begin(), args.begin() + nd) {
    indices.push_back(*it);
  }
  
  SgBasicBlock *set_exp = builder()->BuildGridSet(g, nd, indices,
                                                    args[nd]);
  SgStatement *parent_stmt = isSgStatement(node->get_parent());
  PSAssert(parent_stmt);
  
  si::replaceStatement(parent_stmt, set_exp);
}

void ReferenceTranslator::TranslateReduceGrid(Reduce *rd) {
  // If the element type is a primitive type, use its corresponding
  // reduce function in the reference runtime. Otherwise, create a
  // type-specific reducer function, and then call __PSReduceGrid RT
  // function with the reducer function.
  SgVarRefExp *gv = rd->GetGrid();
  PSAssert(gv);
  //SgInitializedName *gin = gv->get_symbol()->get_declaration();
  GridType *gt = tx_->findGridType(gv);
  SgType *elm_type = gt->point_type();
  
  SgFunctionSymbol *reduce_grid_func;

  if (isSgTypeFloat(elm_type)) {
    PSAssert(reduce_grid_func = 
             si::lookupFunctionSymbolInParentScopes(
                 "__PSReduceGridFloat",
                 global_scope_));
  } else if (isSgTypeDouble(elm_type)) {
    PSAssert(reduce_grid_func = 
             si::lookupFunctionSymbolInParentScopes(
                 "__PSReduceGridDouble",
                 global_scope_));
  } else if (isSgTypeInt(elm_type)) {
    PSAssert(reduce_grid_func = 
             si::lookupFunctionSymbolInParentScopes(
                 "__PSReduceGridInt",
                 global_scope_));
  } else if (isSgTypeLong(elm_type)) {
    PSAssert(reduce_grid_func = 
             si::lookupFunctionSymbolInParentScopes(
                 "__PSReduceGridLong",
                 global_scope_));
  } else {
    LOG_ERROR() << "Unsupported element type.";
    PSAbort(1);
  }

  SgFunctionCallExp *original_rdcall = rd->reduce_call();
  SgFunctionCallExp *new_call =
      sb::buildFunctionCallExp(
          sb::buildFunctionRefExp(reduce_grid_func),
          isSgExprListExp(si::copyExpression(
              original_rdcall->get_args())));
  si::replaceExpression(original_rdcall, new_call);
  return;
}

SgFunctionDeclaration *ReferenceTranslator::BuildReduceGrid(Reduce *rd) {
  return NULL;
}

void ReferenceTranslator::TranslateReduceKernel(Reduce *rd) {
  LOG_ERROR() << "Not implemented yet.";
  PSAbort(1);
}

void ReferenceTranslator::FixGridType() {
  if (ru::IsFortranLikeLanguage()) {
    return;
  }
  SgNodePtrList vdecls =
      NodeQuery::querySubTree(project_, V_SgVariableDeclaration);
  FOREACH (it, vdecls.begin(), vdecls.end()) {
    SgVariableDeclaration *vdecl = isSgVariableDeclaration(*it);
    SgInitializedNamePtrList &vars = vdecl->get_variables();
    FOREACH (vars_it, vars.begin(), vars.end()) {
      SgInitializedName *var = *vars_it;
      if (GridType::isGridType(var->get_type())) {
        var->set_type(sb::buildPointerType(grid_type_));
      }
    }
  }
  SgNodePtrList params =
      NodeQuery::querySubTree(project_, V_SgFunctionParameterList);
  FOREACH (it, params.begin(), params.end()) {
    SgFunctionParameterList *pl = isSgFunctionParameterList(*it);
    SgInitializedNamePtrList &vars = pl->get_args();
    FOREACH (vars_it, vars.begin(), vars.end()) {
      SgInitializedName *var = *vars_it;
      if (GridType::isGridType(var->get_type())) {
        var->set_type(sb::buildPointerType(grid_type_));
      }
    }
  }
  
}

} // namespace translator
} // namespace physis

