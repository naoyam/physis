// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#include "translator/reference_translator.h"

#include "translator/rose_util.h"
#include "translator/translation_context.h"
#include "translator/reference_runtime_builder.h"
#include "translator/runtime_builder.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

ReferenceTranslator::ReferenceTranslator(const Configuration &config):
    Translator(config),
    flag_constant_grid_size_optimization_(true),
    grid_create_name_("__PSGridNew"),
    ref_rt_builder_(NULL) {
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

  defineMapSpecificTypesAndFunctions();
  
  traverseBottomUp(project_);
}

void ReferenceTranslator::SetUp(SgProject *project,
                                TranslationContext *context) {
  Translator::SetUp(project, context);
  ref_rt_builder_ = new ReferenceRuntimeBuilder(global_scope_);  
}

void ReferenceTranslator::Finish() {
  delete ref_rt_builder_;
  ref_rt_builder_ = NULL;
  Translator::Finish();
}

void ReferenceTranslator::Optimize() {
  if (flag_constant_grid_size_optimization_) {
    optimizeConstantSizedGrids();
  }
}

void ReferenceTranslator::optimizeConstantSizedGrids() {
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
#if NOT_SUPPORTED_YET        
    const Grid *grid = *(grid_set->begin());

    if (grid->hasStaticSize()) {
      unsigned int size = grid->getStaticSize()[i];
      SgExpression *size_exp = sb::buildIntVal(size);
      si::replaceExpression(func_call, size_exp);
    }
#endif    
  }
}

void ReferenceTranslator::translateKernelDeclaration(
    SgFunctionDeclaration *node) {
  SgFunctionModifier &modifier = node->get_functionModifier();
  modifier.setInline();
}

SgExprListExp *ReferenceTranslator::generateNewArg(
    GridType *gt, Grid *g, SgVariableDeclaration *dim_decl) {
#if defined(AUTO_DOUBLE_BUFFERING)
  SgExprListExp *new_args
      = sb::buildExprListExp(sb::buildSizeOfOp(gt->getElmType()),
                             sb::buildIntVal(gt->getNumDim()),
                             sb::buildVarRefExp(dim_decl),
                             sb::buildBoolValExp(g->isReadWrite()));
#else
  SgExprListExp *new_args
      = sb::buildExprListExp(sb::buildSizeOfOp(gt->getElmType()),
                             sb::buildIntVal(gt->getNumDim()),
                             sb::buildVarRefExp(dim_decl),
                             sb::buildBoolValExp(false));
#endif  
  appendNewArgExtra(new_args, g);
  return new_args;
}

void ReferenceTranslator::appendNewArgExtra(SgExprListExp *args,
                                            Grid *g) {
  return;
}

void ReferenceTranslator::translateNew(SgFunctionCallExp *node,
                                       GridType *gt) {
  Grid *g = tx_->findGrid(node);
  PSAssert(g);

  SgExprListExp *dims = g->BuildSizeExprList();
  SgBasicBlock *tmpBlock = sb::buildBasicBlock();
  SgVariableDeclaration *dimDecl
      = sb::buildVariableDeclaration(
          "dims", ivec_type_,
          sb::buildAggregateInitializer(dims, ivec_type_),
          tmpBlock);
  tmpBlock->append_statement(dimDecl);


  SgExprListExp *new_args = generateNewArg(gt, g, dimDecl);

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
  tmpBlock->append_statement(sb::buildAssignStatement(grid_var,
                                                      new_call));
  return;
}

SgExpression *ReferenceTranslator::buildOffset(SgInitializedName *gv,
                                               SgScopeStatement *scope,
                                               int numDim,
                                               SgExpressionPtrList &args) {
  /*
    if (3d)
    offset = grid_dimx(g) * grid_dimy(g) * k + grid_dimx(g) * j + i
    if (2d)
    offset = grid_dimx(g) * j + i
    if (1d)
    offset = i
  */

  SgExpression *offset = NULL;
  SgTreeCopy ch;
  for (int i = 0; i < numDim; i++) {
    SgExpression *dim_offset = isSgExpression(args[i]->copy(ch));
    for (int j = 0; j < i; j++) {
      SgExprListExp *arg =
          sb::buildExprListExp(sb::buildVarRefExp(gv->get_name(), scope),
                               sb::buildIntVal(j));
      dim_offset =
          sb::buildMultiplyOp(dim_offset,
                              sb::buildFunctionCallExp(grid_dim_get_func_, arg));
    }
    if (offset) {
      offset = sb::buildAddOp(offset, dim_offset);
    } else {
      offset = dim_offset;
    }
  }
  return offset;
}


void ReferenceTranslator::translateGet(SgFunctionCallExp *node,
                                       SgInitializedName *gv,
                                       bool isKernel) {
  /*
    (type*)(g->p0)[offset]
  */
  SgTreeCopy tc;
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = getContainingScopeStatement(node);
  SgExpression *offset = buildOffset(gv, scope, nd,
                                     node->get_args()->get_expressions());
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  SgExpression *p0 =
      sb::buildArrowExp(g, sb::buildVarRefExp("p0",
                                              grid_decl_->get_definition()));
  p0 = sb::buildCastExp(p0, sb::buildPointerType(gt->getElmType()));
  p0 = sb::buildPntrArrRefExp(p0, offset);

  si::replaceExpression(node, p0);
}

void ReferenceTranslator::translateEmit(SgFunctionCallExp *node,
                                        SgInitializedName *gv) {
  /*
    g->p1[offset] = value;
  */
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgInitializedNamePtrList &params = getContainingFunction(node)->get_args();
  SgExpressionPtrList args;
  FOREACH(it, params.begin(), params.end()) {
    SgInitializedName *p = *it;
    args.push_back(sb::buildVarRefExp(p, getContainingScopeStatement(node)));
  }

  SgExpression *offset = buildOffset(gv, getContainingScopeStatement(node),
                                     nd, args);
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(),
                                      getContainingScopeStatement(node));
#if defined(AUTO_DOUBLE_BUFFERING)
  string dst_buf_name = "p1";
#else
  string dst_buf_name = "p0";
#endif
  SgExpression *p1 =
      sb::buildArrowExp(g, sb::buildVarRefExp(dst_buf_name,
                                              grid_decl_->get_definition()));
  p1 = sb::buildCastExp(p1, sb::buildPointerType(gt->getElmType()));
  SgExpression *lhs = sb::buildPntrArrRefExp(p1, offset);
  LOG_DEBUG() << "emit lhs: " << lhs->unparseToString() << "\n";

  SgTreeCopy tc;
  SgExpression *rhs =
      isSgExpression(node->get_args()->get_expressions()[0]->copy(tc));
  LOG_DEBUG() << "emit rhs: " << rhs->unparseToString() << "\n";

  SgExpression *emit = sb::buildAssignOp(lhs, rhs);
  LOG_DEBUG() << "emit: " << emit->unparseToString() << "\n";

  si::replaceExpression(node, emit, false);
}

SgFunctionDeclaration *ReferenceTranslator::GenerateMap(StencilMap *stencil) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  parlist->append_arg(sb::buildInitializedName("dom", dom_type_));
  SgInitializedNamePtrList &args = stencil->getKernel()->get_args();
  SgInitializedNamePtrList::iterator arg_begin = args.begin();
  arg_begin += tx_->findDomain(stencil->getDom())->num_dims();
  FOREACH(it, arg_begin, args.end()) {
    parlist->append_arg(rose_util::copySgNode(*it));
  }

  SgFunctionDeclaration *mapFunc =
      sb::buildDefiningFunctionDeclaration(stencil->getMapName(),
                                           stencil->stencil_type(),
                                           parlist, global_scope_);
  rose_util::SetFunctionStatic(mapFunc);
  SgFunctionDefinition *mapDef = mapFunc->get_definition();
  SgBasicBlock *bb = sb::buildBasicBlock();
  si::attachComment(bb, "Generated by " + string(__FUNCTION__));
  SgExprListExp *stencil_fields = sb::buildExprListExp();
  SgInitializedNamePtrList &mapArgs = parlist->get_args();
  FOREACH(it, mapArgs.begin(), mapArgs.end()) {
    SgExpression *exp = sb::buildVarRefExp(*it, mapDef);
    stencil_fields->get_expressions().push_back(exp);
    if (GridType::isGridType(exp->get_type())) {
      stencil_fields->append_expression(
          ref_rt_builder_->BuildGridGetID(exp));
    }
  }

  SgVariableDeclaration *svar =
      sb::buildVariableDeclaration(
          "stencil", stencil->stencil_type(),
          sb::buildAggregateInitializer(stencil_fields),
          bb);
  bb->append_statement(svar);
  bb->append_statement(sb::buildReturnStmt(sb::buildVarRefExp(svar)));
  mapDef->set_body(bb);

  return mapFunc;
}

void ReferenceTranslator::translateMap(SgFunctionCallExp *node,
                                       StencilMap *stencil) {
  // gFunctionDeclaration *mapFunc = GenerateMap(stencil);
  // nsertStatement(getContainingFunction(node), mapFunc);

  SgFunctionDeclaration *realMap = stencil->getFunc();
  assert(realMap);
  // This doesn't work because the real map does not have
  // on-defining declarations
  // gFunctionRefExp *ref = sb::buildFunctionRefExp(realMap);
  LOG_DEBUG() << "realmap: "
              << realMap->unparseToString() << "\n";
  SgFunctionRefExp *ref = rose_util::getFunctionRefExp(realMap);
  assert(ref);
  node->set_function(ref);

  SgExpressionPtrList &args = node->get_args()->get_expressions();
  SgExpressionPtrList::iterator b = args.begin();
  si::deepDelete(*b);
  ++b;
  SgExprListExp *new_args = sb::buildExprListExp();
  FOREACH(it, b, args.end()) {
    new_args->append_expression(*it);
  }
  node->set_args(new_args);
}

void ReferenceTranslator::defineMapSpecificTypesAndFunctions() {
  FOREACH(it, tx_->mapBegin(),
          tx_->mapEnd()) {
    StencilMap *s = it->second;
    // multiple stencil_map may call use the same kernel with the
    // same dimensionality
    SgType *t =
        si::lookupNamedTypeInParentScopes(s->getTypeName(), global_scope_);
    if (t) {
      // necessary types and functions are already generated
      s->stencil_type() = isSgClassType(t);
      SgFunctionSymbol *mapSymbol
          = si::lookupFunctionSymbolInParentScopes(s->getMapName(),
                                                   global_scope_);
      assert(mapSymbol);
      s->setFunc(mapSymbol->get_declaration());
      SgFunctionSymbol *runSymbol
          = si::lookupFunctionSymbolInParentScopes(s->getRunName(),
                                                   global_scope_);
      assert(runSymbol);
      s->run() = runSymbol->get_declaration();
      // REFACTORING: ugly usage of direct string
      SgFunctionSymbol *runInnerSymbol
          = si::lookupFunctionSymbolInParentScopes(s->getRunName()
                                                   + "_inner",
                                                   global_scope_);
      if (runInnerSymbol) {
        s->run_inner() = runInnerSymbol->get_declaration();
      }
      continue;
    }

    // build stencil struct
    SgClassDeclaration *decl =
        sb::buildStructDeclaration(s->getTypeName(), global_scope_);
    SgClassDefinition *def = sb::buildClassDefinition(decl);
    def->append_member(
        sb::buildVariableDeclaration(GetStencilDomName(),
                                     dom_type_, NULL, def));
    SgInitializedNamePtrList &args = s->getKernel()->get_args();
    SgInitializedNamePtrList::iterator arg_begin = args.begin();
    // skip the index args
    arg_begin += tx_->findDomain(s->getDom())->num_dims();

    FOREACH(it, arg_begin, args.end()) {
      SgInitializedName *a = *it;
      SgType *type = a->get_type();
      def->append_member(sb::buildVariableDeclaration(a->get_name(),
                                                      type, NULL, def));
      if (GridType::isGridType(type)) {
        def->append_member(sb::buildVariableDeclaration
                           ("__" + a->get_name() + "_index",
                            sb::buildIntType(), NULL, def));
      }
    }

    si::insertStatementAfter(s->getKernel(), decl);
    s->stencil_type() = decl->get_type();

    // define real stencil_map function
    SgFunctionDeclaration *realMap = GenerateMap(s);
    assert(realMap);
    si::insertStatementAfter(decl, realMap);
    s->setFunc(realMap);

    SgFunctionDeclaration *runKernel = BuildRunKernel(s);
    assert(runKernel);
    s->run() = runKernel;
    si::insertStatementAfter(realMap, runKernel);
    SgFunctionDeclaration *runInnerKernel = BuildRunInteriorKernel(s);
    if (runInnerKernel) {
      s->run_inner() = runInnerKernel;
      si::insertStatementAfter(runKernel, runInnerKernel);
    }
    vector<SgFunctionDeclaration*> runBoundaryKernel =
        BuildRunBoundaryKernel(s);
    FOREACH (it, runBoundaryKernel.begin(), runBoundaryKernel.end()) {
      //s->run_boundary() = runBoundaryKernel;
      si::insertStatementAfter(
          runInnerKernel? runInnerKernel : runKernel, *it);
    }
  }
}

static string getLoopIndexName(int d) {
  return "i" + toString(d);
}

static string getStencilArgName() {
  return string("s");
}

SgFunctionCallExp *
ReferenceTranslator::BuildKernelCall(StencilMap *s,
                                        SgExpressionPtrList &indexArgs,
                                        SgScopeStatement *containingScope) {
  SgClassDefinition *stencilDef = s->GetStencilTypeDefinition();
  SgVarRefExp *stencil = sb::buildVarRefExp(getStencilArgName(),
                                            containingScope);
  // append the fields of the stencil type to the argument list
  SgDeclarationStatementPtrList &members = stencilDef->get_members();
  SgExprListExp *args = sb::buildExprListExp();
  FOREACH(it, indexArgs.begin(), indexArgs.end()) {
    args->append_expression(*it);
  }
  FOREACH(it, ++(members.begin()), members.end()) {
    SgVariableDeclaration *d = isSgVariableDeclaration(*it);
    assert(d);
    LOG_DEBUG() << "member: " << d->unparseToString() << "\n";
    SgExpression *exp = sb::buildArrowExp(stencil, sb::buildVarRefExp(d));
    SgVariableDefinition *var_def = d->get_definition();
    ROSE_ASSERT(var_def);
    args->append_expression(exp);
    // skip the grid id
    if (GridType::isGridType(exp->get_type())) {
      ++it;
    }
  }

  SgFunctionCallExp *c =
      sb::buildFunctionCallExp(
          rose_util::getFunctionSymbol(s->getKernel()), args);
  return c;
}

void ReferenceTranslator::appendGridSwap(StencilMap *mc,
                                         SgExpression *stencil,
                                         SgScopeStatement *scope) {
  Kernel *kernel = tx_->findKernel(mc->getKernel());
   FOREACH(it, kernel->getArgs().begin(), kernel->getArgs().end()) {
    SgInitializedName *arg = *it;
    if (!GridType::isGridType(arg->get_type())) continue;
    if (!kernel->isGridParamModified(arg)) continue;
    LOG_DEBUG() << "Modified grid found\n";
    // append grid_swap(arg);
    SgExprListExp *arg_list
        = sb::buildExprListExp(BuildStencilFieldRef(stencil,
                                                    arg->get_name()));
    scope->append_statement
        (sb::buildExprStatement
         (sb::buildFunctionCallExp(grid_swap_, arg_list)));
  }
}

SgBasicBlock* ReferenceTranslator::BuildRunKernelBody(
    StencilMap *s, SgInitializedName *stencil_param) {
  LOG_DEBUG() << "Generating run kernel body\n";
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by BuildRunKernelBody");

  // Generate code like this
  // for (int k = dom.local_min[2]; k < dom.local_max[2]; k++) {
  //   for (int j = dom.local_min[1]; j < dom.local_max[1]; j++) {
  //     for (int i = dom.local_min[0]; i < dom.local_max[0]; i++) {
  //       kernel(i, j, k, g);
  //     }
  //   }
  // }

  SgExpression *stencil = sb::buildVarRefExp(stencil_param);
  SgExpression *minField = BuildStencilDomMinRef(stencil);
  SgExpression *maxField = BuildStencilDomMaxRef(stencil);  
  SgBasicBlock *loopBlock = block;

  LOG_DEBUG() << "Generating nested loop\n";
  SgExpressionPtrList indexArgs;

  SgBasicBlock *innerMostBlock = NULL;
  SgVariableDeclaration *indexDecl = NULL;
  SgStatement *loopStatement = NULL;
  for (int i = 0; i < s->getNumDim(); i++) {
    SgBasicBlock *innerBlock = sb::buildBasicBlock();
    if (!innerMostBlock) innerMostBlock = innerBlock;
    if (indexDecl) {
      innerBlock->append_statement(indexDecl);
      innerBlock->append_statement(loopStatement);
    }
    indexDecl =
        sb::buildVariableDeclaration(getLoopIndexName(i),
                                     sb::buildUnsignedIntType(),
                                     NULL, loopBlock);
    indexArgs.push_back(sb::buildVarRefExp(indexDecl));

    SgStatement *init =
        sb::buildAssignStatement(
            sb::buildVarRefExp(indexDecl),
            sb::buildPntrArrRefExp(minField,
                                   sb::buildIntVal(i)));
    SgStatement *test =
        sb::buildExprStatement(
            sb::buildLessThanOp(
                sb::buildVarRefExp(indexDecl),
                sb::buildPntrArrRefExp(maxField,
                                       sb::buildIntVal(i))));
    SgExpression *incr = sb::buildPlusPlusOp(sb::buildVarRefExp(indexDecl));

    loopStatement = sb::buildForStatement(init, test, incr, innerBlock);
  }
  block->append_statement(indexDecl);
  block->append_statement(loopStatement);

  SgFunctionCallExp *kernelCall = BuildKernelCall(s, indexArgs,
                                                     innerMostBlock);
  innerMostBlock->append_statement(sb::buildExprStatement(kernelCall));

  return block;
}


SgFunctionDeclaration *ReferenceTranslator::BuildRunKernel(StencilMap *s) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  SgType *stencil_type = sb::buildPointerType(s->stencil_type());
  SgInitializedName *stencil_param =
  sb::buildInitializedName(getStencilArgName(),
                           stencil_type);
  parlist->append_arg(stencil_param);
  SgFunctionDeclaration *runFunc =
      sb::buildDefiningFunctionDeclaration(s->getRunName(),
                                           sb::buildVoidType(),
                                           parlist, global_scope_);
  rose_util::SetFunctionStatic(runFunc);

  runFunc->get_definition()->set_body(BuildRunKernelBody(s, stencil_param));
  return runFunc;
}

SgBasicBlock *ReferenceTranslator::BuildRunBody(Run *run) {
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by BuildRunBody");
  SgVariableDeclaration *lv
      = sb::buildVariableDeclaration("i", sb::buildIntType(), NULL, block);
  block->append_statement(lv);
  SgBasicBlock *loopBody = sb::buildBasicBlock();
  SgStatement *loopTest =
      sb::buildExprStatement(
          sb::buildLessThanOp(sb::buildVarRefExp(lv),
                              sb::buildVarRefExp("iter", block)));

  SgForStatement *loop =
      sb::buildForStatement(sb::buildAssignStatement(sb::buildVarRefExp(lv),
                                                     sb::buildIntVal(0)),
                            loopTest,
                            sb::buildPlusPlusOp(sb::buildVarRefExp(lv)),
                            loopBody);

  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *s = it->second;
    SgFunctionSymbol *fs = rose_util::getFunctionSymbol(s->run());
    assert(fs);
    string stencilName = "s" + toString(i);
    SgExpression *stencil = sb::buildVarRefExp(stencilName, block);
    SgExprListExp *args =
        sb::buildExprListExp(sb::buildAddressOfOp(stencil));
    SgFunctionCallExp *c = sb::buildFunctionCallExp(fs, args);
    loopBody->append_statement(sb::buildExprStatement(c));
    appendGridSwap(s, stencil, loopBody);
  }

  TraceStencilRun(run, loop, block);
  
  return block;
}

void ReferenceTranslator::TraceStencilRun(Run *run,
                                          SgScopeStatement *loop,
                                          SgScopeStatement *cur_scope) {
  // tracing
  // build a string message with kernel names
  StringJoin sj;
  FOREACH (it, run->stencils().begin(), run->stencils().end()) {
    sj << it->second->getKernel()->get_name().str();
  }
  // Call the pre trace function
  rose_util::AppendExprStatement(
      cur_scope, BuildTraceStencilPre(sb::buildStringVal(sj.str())));
  // Declare a stopwatch
  SgVariableDeclaration *st_decl = BuildStopwatch("st", cur_scope, global_scope_);
  cur_scope->append_statement(st_decl);
  SgExpression *st_ptr = sb::buildAddressOfOp(sb::buildVarRefExp(st_decl));  
  // Start the stopwatch
  rose_util::AppendExprStatement(cur_scope, BuildStopwatchStart(st_ptr));

  // Enter the loop
  cur_scope->append_statement(loop);

  // Stop the stopwatch and call the post trace function
  rose_util::AppendExprStatement(
      cur_scope, BuildTraceStencilPost(BuildStopwatchStop(st_ptr)));
  return;
}

SgFunctionDeclaration *ReferenceTranslator::GenerateRun(Run *run) {
  // setup the parameter list
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  parlist->append_arg(sb::buildInitializedName("iter",
                                               sb::buildIntType()));
  
  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *stencil = it->second;
    //SgType *stencilType = sb::buildPointerType(stencil->getType());
    SgType *stencilType = stencil->stencil_type();
    parlist->append_arg(sb::buildInitializedName("s" + toString(i),
                                                 stencilType));
  }

  // Declare and define the function
  SgFunctionDeclaration *runFunc =
      sb::buildDefiningFunctionDeclaration(run->GetName(),
                                           sb::buildVoidType(),
                                           parlist, global_scope_);
  rose_util::SetFunctionStatic(runFunc);

  // Function body
  SgFunctionDefinition *fdef = runFunc->get_definition();
  fdef->set_body(BuildRunBody(run));
  si::attachComment(runFunc, "Generated by GenerateRun");
  return runFunc;
}

void ReferenceTranslator::translateRun(SgFunctionCallExp *node,
                                       Run *run) {
  SgFunctionDeclaration *runFunc = GenerateRun(run);
  si::insertStatementBefore(getContainingFunction(node), runFunc);
  // redirect the call to the real function
  SgFunctionRefExp *ref = rose_util::getFunctionRefExp(runFunc);
  node->set_function(ref);

  // if iteration count is not specified, add 1 to the arg list
  if (!run->count()) {
    node->get_args()->prepend_expression(sb::buildIntVal(1));
  } else {
    SgExpressionPtrList &el = node->get_args()->get_expressions();
    SgExpression *last = el.back();
    el.pop_back();
    el.insert(el.begin(), last);
  }
}

// ReferenceRuntimeBuilder* ReferenceTranslator::GetRuntimeBuilder() const {
//   LOG_DEBUG() << "Using reference runtime builder\n";
//   return new ReferenceRuntimeBuilder();
// }

string ReferenceTranslator::GetStencilDomName() const {
  return string("dom");
}

SgExpression *ReferenceTranslator::BuildStencilDomRef(
    SgExpression *stencil) const {
  return BuildStencilFieldRef(stencil, GetStencilDomName());
}
/*
SgVarRefExp *ReferenceTranslator::BuildDomainMinRef(
    SgClassDeclaration *d) const {
  return rose_util::buildFieldRefExp(d, "local_min");
}

SgVarRefExp *ReferenceTranslator::BuildDomainMaxRef(
    SgClassDeclaration *d) const {
  return rose_util::buildFieldRefExp(d, "local_max");
}
*/

SgExpression *ReferenceTranslator::BuildDomMaxRef(SgExpression *domain)
    const {
  SgExpression *field = sb::buildVarRefExp("local_max");
  if (isSgPointerType(domain->get_type())) {
    return sb::buildArrowExp(domain, field);
  } else {
    return sb::buildDotExp(domain, field);
  }
}

SgExpression *ReferenceTranslator::BuildDomMinRef(SgExpression *domain)
    const {
  SgExpression *field = sb::buildVarRefExp("local_min");
  if (isSgPointerType(domain->get_type())) {
    return sb::buildArrowExp(domain, field);
  } else {
    return sb::buildDotExp(domain, field);
  }
}

SgExpression *ReferenceTranslator::BuildDomMaxRef(SgExpression *domain,
                                                  int dim)
    const {
  SgExpression *exp = BuildDomMaxRef(domain);
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}

SgExpression *ReferenceTranslator::BuildDomMinRef(SgExpression *domain,
                                                  int dim)
    const {
  SgExpression *exp = BuildDomMinRef(domain);
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}

SgExpression *ReferenceTranslator::BuildStencilDomMaxRef(
    SgExpression *stencil) const {
  SgExpression *exp =
      BuildStencilFieldRef(stencil, GetStencilDomName());
  // s.dom.local_max
  return BuildDomMaxRef(exp);
}

SgExpression *ReferenceTranslator::BuildStencilDomMaxRef(
    SgExpression *stencil, int dim) const {
  SgExpression *exp = BuildStencilDomMaxRef(stencil);
  // s.dom.local_max[dim]
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}

SgExpression *ReferenceTranslator::BuildStencilDomMinRef(
    SgExpression *stencil) const {
  SgExpression *exp =
      BuildStencilFieldRef(stencil, GetStencilDomName());
  // s.dom.local_max
  return BuildDomMinRef(exp);  
}

SgExpression *ReferenceTranslator::BuildStencilDomMinRef(
    SgExpression *stencil, int dim) const {
  SgExpression *exp = BuildStencilDomMinRef(stencil);
  // s.dom.local_max[dim]
  exp = sb::buildPntrArrRefExp(exp, sb::buildIntVal(dim));
  return exp;
}

SgExpression *ReferenceTranslator::BuildStencilFieldRef(
    SgExpression *stencil_ref, SgExpression *field) const {
  if (isSgPointerType(stencil_ref->get_type())) {
    return sb::buildArrowExp(stencil_ref, field);
  } else {
    return sb::buildDotExp(stencil_ref, field);
  }
}

SgExpression *ReferenceTranslator::BuildStencilFieldRef(
    SgExpression *stencil_ref, string name) const {
  SgVarRefExp *field = sb::buildVarRefExp(name);
  return BuildStencilFieldRef(stencil_ref, field);
}

void ReferenceTranslator::translateSet(SgFunctionCallExp *node,
                                       SgInitializedName *gv) {
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = getContainingScopeStatement(node);    
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  SgExpressionPtrList &args = node->get_args()->get_expressions();
  SgExpressionPtrList indices;
  FOREACH (it, args.begin(), args.begin() + nd) {
    indices.push_back(*it);
  }
  
  SgBasicBlock *set_exp = ref_rt_builder_->BuildGridSet(g, nd, indices,
                                                       args[nd]);
  SgStatement *parent_stmt = isSgStatement(node->get_parent());
  PSAssert(parent_stmt);
  
  si::replaceStatement(parent_stmt, set_exp);
}


} // namespace translator
} // namespace physis

