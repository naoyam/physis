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
    validate_ast_(true),
    grid_create_name_("__PSGridNew"),
    rt_builder_(NULL) {
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

  FixAST();
  ValidateASTConsistency();
}

void ReferenceTranslator::FixAST() {
  FixGridType();
  si::fixVariableReferences(project_);
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
                                TranslationContext *context) {
  Translator::SetUp(project, context);
  rt_builder_ = new ReferenceRuntimeBuilder(global_scope_);  
}

void ReferenceTranslator::Finish() {
  if (rt_builder_) delete rt_builder_;
  rt_builder_ = NULL;
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

void ReferenceTranslator::translateKernelDeclaration(
    SgFunctionDeclaration *node) {
  SgFunctionModifier &modifier = node->get_functionModifier();
  modifier.setInline();
}

SgExprListExp *ReferenceTranslator::generateNewArg(
    GridType *gt, Grid *g, SgVariableDeclaration *dim_decl) {
#ifdef UNUSED_CODE
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
  //SgExpression *attr = g->BuildAttributeExpr();
  //if (!attr) attr = sb::buildIntVal(0);
  //si::appendExpression(new_args, attr);
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
  si::appendStatement(dimDecl, tmpBlock);

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
  si::appendStatement(
      sb::buildAssignStatement(grid_var, new_call),
      tmpBlock);
  return;
}

// TODO: Change args type to SgExpressionPtrList
SgExpression *ReferenceTranslator::BuildOffset(SgInitializedName *gv,
                                               int num_dim,
                                               SgExprListExp *args,
                                               bool is_kernel,
                                               bool is_periodic,
                                               const StencilIndexList *sil,
                                               SgScopeStatement *scope) {

  /*
    __PSGridGetOffsetND(g, i)
  */
  SgExpression *offset = NULL;
#if 0  
  for (int i = 1; i <= num_dim; i++) {
    SgExpression *dim_offset = si::copyExpression(
        args->get_expressions()[i-1]);
    goa->AppendIndex(dim_offset);
    if (is_periodic) {
      const StencilIndex &si = sil->at(i-1);
      // si.dim is assumed to be equal to i, i.e., the i'th index
      // variable is always used for the i'th index when accessing
      // grids. This assumption is made to simplify the implementation
      // of MPI versions, and is actually possible to be relaxed in
      // shared-memory verions such as reference and cuda. Here we
      // assume si.dim can actually be different from i.
      if (si.dim != i || si.offset != 0) {
        dim_offset = sb::buildModOp(
            sb::buildAddOp(
                dim_offset,
                rt_builder_->BuildGridDim(
                    sb::buildVarRefExp(gv->get_name()), i)),
            rt_builder_->BuildGridDim(
                sb::buildVarRefExp(gv->get_name()), i));
      }
    }
    for (int j = 1; j < i; j++) {
      dim_offset = sb::buildMultiplyOp(
          dim_offset,
          rt_builder_->BuildGridDim(
              sb::buildVarRefExp(gv->get_name()), j));
    }
    if (offset) {
      offset = sb::buildAddOp(offset, dim_offset);
    } else {
      offset = dim_offset;
    }
  }
#else
  // Use the getoffset function
  SgExpressionPtrList offset_args;
  for (int i = 1; i <= num_dim; i++) {
    SgExpression *dim_offset = si::copyExpression(
        args->get_expressions()[i-1]);
    offset_args.push_back(dim_offset);    
  }
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);  
  offset = rt_builder_->BuildGridOffset(g, num_dim,
                                        &offset_args,
                                        is_kernel, is_periodic, sil);
#endif
  return offset;
}

void ReferenceTranslator::translateGet(SgFunctionCallExp *node,
                                       SgInitializedName *gv,
                                       bool is_kernel,
                                       bool is_periodic) {
  /*
    (type*)(g->p0)[offset]
  */
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = si::getEnclosingFunctionDefinition(node);
  
  SgExpression *offset = BuildOffset(
      gv, nd, node->get_args(),
      is_kernel, is_periodic,
      rose_util::GetASTAttribute<GridGetAttribute>(node)->GetStencilIndexList(),
      scope);
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  SgExpression *p0 = sb::buildArrowExp(
      g, sb::buildVarRefExp("p0", grid_decl_->get_definition()));
  p0 = sb::buildCastExp(p0, sb::buildPointerType(gt->getElmType()));
  p0 = sb::buildPntrArrRefExp(p0, offset);
  rose_util::CopyASTAttribute<GridGetAttribute>(p0, node);
  rose_util::GetASTAttribute<GridGetAttribute>(p0)->offset() = offset;
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
  //SgExpressionPtrList args;
  SgExprListExp *args = sb::buildExprListExp();
  for (int i = 0; i < nd; ++i) {
    SgInitializedName *p = params[i];
    si::appendExpression(
        args,
        sb::buildVarRefExp(p, getContainingScopeStatement(node)));
  }

  StencilIndexList sil;
  StencilIndexListInitSelf(sil, nd);
  SgExpression *offset = BuildOffset(gv, nd, args, true, false, &sil,
                                     getContainingScopeStatement(node));
  SgVarRefExp *g =
      sb::buildVarRefExp(gv->get_name(),
                         getContainingScopeStatement(node));

#if defined(AUTO_DOUBLE_BUFFERING)
  string dst_buf_name = "p1";
#else
  string dst_buf_name = "p0";
#endif
  SgExpression *p1 =
      sb::buildArrowExp(g,
                        sb::buildVarRefExp(dst_buf_name,
                                           grid_decl_->get_definition()));
  p1 = sb::buildCastExp(p1, sb::buildPointerType(gt->getElmType()));
  SgExpression *lhs = sb::buildPntrArrRefExp(p1, offset);
  LOG_DEBUG() << "emit lhs: " << lhs->unparseToString() << "\n";

  SgExpression *rhs =
      si::copyExpression(node->get_args()->get_expressions()[0]);
  LOG_DEBUG() << "emit rhs: " << rhs->unparseToString() << "\n";

  SgExpression *emit = sb::buildAssignOp(lhs, rhs);
  LOG_DEBUG() << "emit: " << emit->unparseToString() << "\n";

  si::replaceExpression(node, emit);
}

SgFunctionDeclaration *ReferenceTranslator::GenerateMap(StencilMap *stencil) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  si::appendArg(parlist, sb::buildInitializedName("dom", dom_type_));
  SgInitializedNamePtrList &args = stencil->getKernel()->get_args();
  SgInitializedNamePtrList::iterator arg_begin = args.begin();
  arg_begin += tx_->findDomain(stencil->getDom())->num_dims();
  FOREACH(it, arg_begin, args.end()) {
    si::appendArg(parlist, rose_util::copySgNode(*it));
  }

  SgFunctionDeclaration *mapFunc =
      sb::buildDefiningFunctionDeclaration(stencil->getMapName(),
                                           stencil->stencil_type(),
                                           parlist, global_scope_);
  rose_util::SetFunctionStatic(mapFunc);
  SgFunctionDefinition *mapDef = mapFunc->get_definition();
  SgBasicBlock *bb = mapDef->get_body();
  si::attachComment(bb, "Generated by " + string(__FUNCTION__));
  SgExprListExp *stencil_fields = sb::buildExprListExp();
  SgInitializedNamePtrList &mapArgs = parlist->get_args();
  FOREACH(it, mapArgs.begin(), mapArgs.end()) {
    SgExpression *exp = sb::buildVarRefExp(*it, mapDef);
    si::appendExpression(stencil_fields, exp);
    if (GridType::isGridType(exp->get_type())) {
      si::appendExpression(stencil_fields,
                           rt_builder_->BuildGridGetID(exp));
    }
  }

  SgVariableDeclaration *svar =
      sb::buildVariableDeclaration(
          "stencil", stencil->stencil_type(),
          sb::buildAggregateInitializer(stencil_fields),
          bb);
  si::appendStatement(svar, bb);
  si::appendStatement(sb::buildReturnStmt(sb::buildVarRefExp(svar)),
                      bb);
  rose_util::ReplaceFuncBody(mapFunc, bb);

  return mapFunc;
}

void ReferenceTranslator::translateMap(SgFunctionCallExp *node,
                                       StencilMap *stencil) {
  // gFunctionDeclaration *mapFunc = GenerateMap(stencil);
  // nsertStatement(getContainingFunction(node), mapFunc);

  SgFunctionDeclaration *realMap = stencil->getFunc();
  assert(realMap);
  // This doesn't work because the real map does not have
  // defining declarations
  // SgFunctionRefExp *ref = sb::buildFunctionRefExp(realMap);
  LOG_DEBUG() << "realmap: "
              << realMap->unparseToString() << "\n";
  SgFunctionRefExp *ref = rose_util::getFunctionRefExp(realMap);
  assert(ref);
  LOG_DEBUG() << "Map function: " << ref->unparseToString() << "\n";

  SgExpressionPtrList &args = node->get_args()->get_expressions();
  SgExprListExp *new_args = sb::buildExprListExp();
  FOREACH(it, args.begin() + 1, args.end()) {
    si::appendExpression(new_args, si::copyExpression(*it));
  }
  si::replaceExpression(node,
                        sb::buildFunctionCallExp(ref, new_args));
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
    si::appendStatement(
        sb::buildVariableDeclaration(GetStencilDomName(),
                                     dom_type_, NULL, def),
        def);

    SgInitializedNamePtrList &args = s->getKernel()->get_args();
    SgInitializedNamePtrList::iterator arg_begin = args.begin();
    // skip the index args
    arg_begin += tx_->findDomain(s->getDom())->num_dims();

    FOREACH(it, arg_begin, args.end()) {
      SgInitializedName *a = *it;
      SgType *type = a->get_type();
      si::appendStatement(
          sb::buildVariableDeclaration(a->get_name(),
                                       type, NULL, def),
          def);
      if (GridType::isGridType(type)) {
        si::appendStatement(
            sb::buildVariableDeclaration(
                "__" + a->get_name() + "_index",
                sb::buildIntType(), NULL, def),
            def);
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
                                     SgInitializedName *stencil_param) {
  SgClassDefinition *stencilDef = s->GetStencilTypeDefinition();
  // append the fields of the stencil type to the argument list
  SgDeclarationStatementPtrList &members = stencilDef->get_members();
  SgExprListExp *args = sb::buildExprListExp();
  FOREACH (it, indexArgs.begin(), indexArgs.end()) {
    si::appendExpression(args, *it);
  }
  FOREACH (it, ++(members.begin()), members.end()) {
    SgVariableDeclaration *d = isSgVariableDeclaration(*it);
    assert(d);
    LOG_DEBUG() << "member: " << d->unparseToString() << "\n";
    SgVarRefExp *stencil = sb::buildVarRefExp(stencil_param);
    SgExpression *exp = sb::buildArrowExp(stencil, sb::buildVarRefExp(d));
    SgVariableDefinition *var_def = d->get_definition();
    ROSE_ASSERT(var_def);
    si::appendExpression(args, exp);
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
                                         const string &stencil_var_name,
                                         bool is_stencil_ptr,
                                         SgScopeStatement *scope) {
  Kernel *kernel = tx_->findKernel(mc->getKernel());
   FOREACH(it, kernel->getArgs().begin(), kernel->getArgs().end()) {
    SgInitializedName *arg = *it;
    if (!GridType::isGridType(arg->get_type())) continue;
    if (!kernel->isGridParamModified(arg)) continue;
    LOG_DEBUG() << "Modified grid found\n";
    // append grid_swap(arg);
    // Use unbound reference to arg since the scope is not conntected
    // to the outer scope.
    SgExpression *sv = sb::buildVarRefExp(stencil_var_name);
    SgExpression *gv = sb::buildVarRefExp(arg->get_name());
    SgExpression *e = (is_stencil_ptr) ?
        (SgExpression*)(sb::buildArrowExp(sv, gv)) :
        (SgExpression*)(sb::buildDotExp(sv, gv));
    SgExprListExp *arg_list = sb::buildExprListExp(e);
    si::appendStatement(
        sb::buildExprStatement(
            sb::buildFunctionCallExp(grid_swap_, arg_list)),
        scope);
  }
}

SgBasicBlock* ReferenceTranslator::BuildRunKernelBody(
    StencilMap *s, SgInitializedName *stencil_param) {
  LOG_DEBUG() << "Generating run kernel body\n";
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));

  // Generate code like this
  // for (int k = dom.local_min[2]; k < dom.local_max[2]; k++) {
  //   for (int j = dom.local_min[1]; j < dom.local_max[1]; j++) {
  //     for (int i = dom.local_min[0]; i < dom.local_max[0]; i++) {
  //       kernel(i, j, k, g);
  //     }
  //   }
  // }

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
      si::appendStatement(indexDecl, innerBlock);
      si::appendStatement(loopStatement, innerBlock);
    }
    indexDecl =
        sb::buildVariableDeclaration(getLoopIndexName(i),
                                     sb::buildUnsignedIntType(),
                                     NULL, loopBlock);
    rose_util::AddASTAttribute<RunKernelIndexVarAttribute>(
        indexDecl,  new RunKernelIndexVarAttribute(i+1));
    indexArgs.push_back(sb::buildVarRefExp(indexDecl));
    SgExpression *loop_begin =
        sb::buildPntrArrRefExp(
            BuildStencilDomMinRef(sb::buildVarRefExp(stencil_param)),
            sb::buildIntVal(i));
    SgStatement *init =
        sb::buildAssignStatement(
            sb::buildVarRefExp(indexDecl), loop_begin);
    SgExpression *loop_end =
        sb::buildPntrArrRefExp(
            BuildStencilDomMaxRef(sb::buildVarRefExp(stencil_param)),
            sb::buildIntVal(i));
    SgStatement *test =
        sb::buildExprStatement(
            sb::buildLessThanOp(
                sb::buildVarRefExp(indexDecl), loop_end));
    SgExpression *incr = sb::buildPlusPlusOp(
        sb::buildVarRefExp(indexDecl));
    loopStatement = sb::buildForStatement(init, test, incr, innerBlock);
    rose_util::AddASTAttribute(
        loopStatement,
        new RunKernelLoopAttribute(
            i+1,indexDecl->get_variables()[0], loop_begin,
            loop_end));
  }
  si::appendStatement(indexDecl, block);
  si::appendStatement(loopStatement, block);

  SgFunctionCallExp *kernelCall =
      BuildKernelCall(s, indexArgs, stencil_param);
  si::appendStatement(sb::buildExprStatement(kernelCall),
                      innerMostBlock);
  return block;
}


SgFunctionDeclaration *ReferenceTranslator::BuildRunKernel(StencilMap *s) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  SgType *stencil_type =
      sb::buildConstType(sb::buildPointerType(
          sb::buildConstType(s->stencil_type())));
  //SgType *stencil_type = sb::buildPointerType(s->stencil_type());
  SgInitializedName *stencil_param =
      sb::buildInitializedName(getStencilArgName(),
                               stencil_type);
  si::appendArg(parlist, stencil_param);
  SgFunctionDeclaration *runFunc =
      sb::buildDefiningFunctionDeclaration(s->getRunName(),
                                           sb::buildVoidType(),
                                           parlist, global_scope_);
  rose_util::SetFunctionStatic(runFunc);
  si::attachComment(runFunc, "Generated by " + string(__FUNCTION__));

  si::appendStatement(
      BuildRunKernelBody(s, stencil_param),
      runFunc->get_definition());
      
  rose_util::AddASTAttribute(runFunc,
                             new RunKernelAttribute(s, stencil_param));

  si::replaceStatement(runFunc->get_definition()->get_body(),
                       BuildRunKernelBody(s, stencil_param));
  return runFunc;
}

SgBasicBlock *ReferenceTranslator::BuildRunBody(Run *run) {
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));
  SgVariableDeclaration *lv
      = sb::buildVariableDeclaration("i", sb::buildIntType(), NULL, block);
  si::appendStatement(lv, block);
  SgBasicBlock *loopBody = sb::buildBasicBlock();
  SgStatement *loopTest =
      sb::buildExprStatement(
          sb::buildLessThanOp(sb::buildVarRefExp(lv),
                              sb::buildVarRefExp("iter", block)));

  SgForStatement *loop =
      sb::buildForStatement(
          sb::buildAssignStatement(sb::buildVarRefExp(lv),
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
    si::appendStatement(sb::buildExprStatement(c), loopBody);
    appendGridSwap(s, stencilName, false, loopBody);
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
  si::appendStatement(st_decl, cur_scope);
  SgExpression *st_ptr = sb::buildAddressOfOp(sb::buildVarRefExp(st_decl));  
  // Start the stopwatch
  rose_util::AppendExprStatement(cur_scope, BuildStopwatchStart(st_ptr));

  // Enter the loop
  si::appendStatement(loop, cur_scope);

  // Stop the stopwatch and call the post trace function
  si::appendStatement(
      sb::buildVariableDeclaration(
          "f", sb::buildFloatType(),
          sb::buildAssignInitializer(
              BuildStopwatchStop(st_ptr), sb::buildFloatType())),
      cur_scope);
  rose_util::AppendExprStatement(
      cur_scope, BuildTraceStencilPost(sb::buildVarRefExp("f")));
  si::appendStatement(
      sb::buildReturnStmt(sb::buildVarRefExp("f")), cur_scope); /* return f; */
  return;
}

SgFunctionDeclaration *ReferenceTranslator::GenerateRun(Run *run) {
  // setup the parameter list
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  si::appendArg(parlist, sb::buildInitializedName("iter",
                                                  sb::buildIntType()));
  /* auto tuning & has dynamic arguments */
  if (config_.auto_tuning() && config_.ndynamic() > 1) {
    AddDynamicParameter(parlist);
  }
  
  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    StencilMap *stencil = it->second;
    //SgType *stencilType = sb::buildPointerType(stencil->getType());
    SgType *stencilType = stencil->stencil_type();
    si::appendArg(parlist,
                  sb::buildInitializedName("s" + toString(i),
                                           stencilType));
  }

  // Declare and define the function
  SgFunctionDeclaration *runFunc =
      sb::buildDefiningFunctionDeclaration(run->GetName(),
                                           sb::buildFloatType(),
                                           parlist, global_scope_);
  rose_util::SetFunctionStatic(runFunc);

  // Function body
  SgFunctionDefinition *fdef = runFunc->get_definition();
  si::replaceStatement(fdef->get_body(), BuildRunBody(run));
  si::attachComment(runFunc, "Generated by " + string(__FUNCTION__));  
  si::attachComment(runFunc, "Generated by GenerateRun");
  return runFunc;
}

#include <float.h> /* for FLT_MAX */
#include <dlfcn.h> /* for RTLD_NOW */
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
  AddSyncAfterDlclose(if_true);
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
                  //sb::buildOpaqueVarRefExp("RTLD_NOW")
                  sb::buildIntVal(RTLD_NOW)
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
    AddDynamicArgument(args, sb::buildVarRefExp("a", while_body));
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
    AddDynamicArgument(args, sb::buildVarRefExp("a", if_true));
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
    AddSyncAfterDlclose(funcBlock);
  }

  si::replaceStatement(fdef->get_body(), funcBlock);
  si::attachComment(trialFunc, "Generated by " + string(__FUNCTION__));
  return trialFunc;
}
/** add dynamic parameter
 * @param[in/out] parlist ... parameter list
 */
void ReferenceTranslator::AddDynamicParameter(
    SgFunctionParameterList *parlist) {
  /* do nothing for ReferenceTranslator */
}
/** add dynamic argument
 * @param[in/out] args ... arguments
 * @param[in] a_exp ... index expression
 */
void ReferenceTranslator::AddDynamicArgument(
    SgExprListExp *args, SgExpression *a_exp) {
  /* do nothing for ReferenceTranslator */
}
/** add some code after dlclose()
 * @param[in] scope
 */
void ReferenceTranslator::AddSyncAfterDlclose(
    SgScopeStatement *scope) {
  /* do nothing for ReferenceTranslator */
}

void ReferenceTranslator::translateRun(SgFunctionCallExp *node,
                                       Run *run) {
  SgFunctionDeclaration *runFunc = GenerateRun(run);
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

string ReferenceTranslator::GetStencilDomName() const {
  return string("dom");
}

// TODO: These build functions should be moved to the RuntimeBuilder
// class. 
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
  SgClassDeclaration *dom_decl =
      isSgClassDeclaration(
          isSgClassType(dom_type_->get_base_type())->get_declaration()->
          get_definingDeclaration());
  SgExpression *field = sb::buildVarRefExp("local_max",
                                           dom_decl->get_definition());
  //SgExpression *field = sb::buildVarRefExp("local_max");
  if (si::isPointerType(domain->get_type())) {
    return sb::buildArrowExp(domain, field);
  } else {
    return sb::buildDotExp(domain, field);
  }
}

SgExpression *ReferenceTranslator::BuildDomMinRef(SgExpression *domain)
    const {
  //SgExpression *field = sb::buildVarRefExp("local_min");
  SgClassDeclaration *dom_decl =
      isSgClassDeclaration(
          isSgClassType(dom_type_->get_base_type())->get_declaration()->
          get_definingDeclaration());
  LOG_DEBUG() << "domain: " << domain->unparseToString() << "\n";
  SgExpression *field = sb::buildVarRefExp("local_min",
                                           dom_decl->get_definition());
  SgType *ty = domain->get_type();
  PSAssert(ty && !isSgTypeUnknown(ty));
  if (si::isPointerType(ty)) {
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
  SgType *ty = stencil_ref->get_type();
  PSAssert(ty && !isSgTypeUnknown(ty));
  if (si::isPointerType(ty)) {
    return sb::buildArrowExp(stencil_ref, field);
  } else {
    return sb::buildDotExp(stencil_ref, field);
  }
}

SgExpression *ReferenceTranslator::BuildStencilFieldRef(
    SgExpression *stencil_ref, string name) const {
  SgType *ty = stencil_ref->get_type();
  PSAssert(ty && !isSgTypeUnknown(ty));
  SgType *stencil_type = NULL;
  if (si::isPointerType(ty)) {
    stencil_type = si::getElementType(stencil_ref->get_type());
  } else {
    stencil_type = stencil_ref->get_type();
  }
  if (isSgModifierType(stencil_type)) {
    stencil_type = isSgModifierType(stencil_type)->get_base_type();
  }
  SgClassType *stencil_class_type = isSgClassType(stencil_type);
  // If the type is resolved to the actual class type, locate the
  // actual definition of field. Otherwise, temporary create an
  // unbound reference to the name.
  SgVarRefExp *field = NULL;
  if (stencil_class_type) {
    SgClassDefinition *stencil_def =
        isSgClassDeclaration(
            stencil_class_type->get_declaration()->get_definingDeclaration())->
        get_definition();
    field = sb::buildVarRefExp(name, stencil_def);
  } else {
    // Temporary create an unbound reference; this does not pass the
    // AST consistency tests unless fixed.
    field = sb::buildVarRefExp(name);    
  }
  return BuildStencilFieldRef(stencil_ref, field);
}

void ReferenceTranslator::translateSet(SgFunctionCallExp *node,
                                       SgInitializedName *gv) {
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = getContainingScopeStatement(node);    
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  SgExpressionPtrList &
      args = node->get_args()->get_expressions();
  SgExpressionPtrList indices;
  FOREACH (it, args.begin(), args.begin() + nd) {
    indices.push_back(*it);
  }
  
  SgBasicBlock *set_exp = rt_builder_->BuildGridSet(g, nd, indices,
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
  SgType *elm_type = gt->getElmType();
  
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

