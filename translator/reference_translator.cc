// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "translator/reference_translator.h"

#include <float.h> /* for FLT_MAX */
#include <boost/foreach.hpp>

#include "translator/rose_util.h"
#include "translator/ast_processing.h"
#include "translator/translation_context.h"
#include "translator/reference_runtime_builder.h"
#include "translator/runtime_builder.h"
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
                                RuntimeBuilder *rt_builder) {
  Translator::SetUp(project, context, rt_builder);
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

void ReferenceTranslator::TranslateKernelDeclaration(
    SgFunctionDeclaration *node) {
  SgFunctionModifier &modifier = node->get_functionModifier();
  modifier.setInline();
}

SgExprListExp *ReferenceTranslator::generateNewArg(
    GridType *gt, Grid *g, SgVariableDeclaration *dim_decl) {
  SgExprListExp *new_args
      = sb::buildExprListExp(sb::buildSizeOfOp(gt->point_type()),
                             sb::buildIntVal(gt->rank()),
                             sb::buildVarRefExp(dim_decl));
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

  SgExprListExp *dims = rt_builder_->BuildSizeExprList(g);
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
  SgExpression *p0 = rt_builder_->BuildGridGet(
      sb::buildVarRefExp(gv->get_name(), si::getScope(node)),
      rose_util::GetASTAttribute<GridVarAttribute>(gv),
      gt,
      &args, sil, is_kernel, is_periodic);
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
      rt_builder_->BuildGridEmit(sb::buildVarRefExp(attr->gv()),
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
      // REFACTORING: ugly usage of direct string
      SgFunctionSymbol *runInnerSymbol
          = si::lookupFunctionSymbolInParentScopes(s->GetRunName()
                                                   + "_inner",
                                                   global_scope_);
      if (runInnerSymbol) {
        s->run_inner() = runInnerSymbol->get_declaration();
      }
      continue;
    }

    SgClassDeclaration *sm_type = rt_builder_->BuildStencilMapType(s);
    InsertStencilSpecificType(s, sm_type);
    s->stencil_type() = sm_type->get_type();

    // define real stencil_map function
    SgFunctionDeclaration *realMap = rt_builder_->BuildMap(s);
    assert(realMap);
    InsertStencilSpecificFunc(s, realMap);
    s->setFunc(realMap);

    SgFunctionDeclaration *runKernel = BuildRunKernel(s);
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
    SgExpression *exp = ru::BuildFieldRef(stencil, sb::buildVarRefExp(d));
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

static SgExpression *BuildRedBlackInitOffset(
    SgExpression *idx,
    vector<SgVariableDeclaration*> &indices,
    SgInitializedName *rb_param,
    int nd) {
  // idx + idx & 1 ^ (i1 + i2 + ... + c) % 2  
  SgExpression *rb_offset = sb::buildVarRefExp(rb_param);
  for (int i = 1; i < nd; ++i) {
    rb_offset = sb::buildAddOp(rb_offset, sb::buildVarRefExp(indices[i]));
  }
  rb_offset = sb::buildModOp(rb_offset, sb::buildIntVal(2));
  rb_offset = sb::buildBitXorOp(sb::buildBitAndOp(si::copyExpression(idx),
                                                  sb::buildIntVal(1)),
                                rb_offset);
  return rb_offset;
}

SgBasicBlock* ReferenceTranslator::BuildRunKernelBody(
    StencilMap *s, SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices) {
  LOG_DEBUG() << "Generating run kernel body\n";
  SgInitializedName *stencil_param = param->get_args()[0];
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));

  // Generate code like this
  // for (int k = dom.local_min[2]; k <= dom.local_max[2]-1; k++) {
  //   for (int j = dom.local_min[1]; j <= dom.local_max[1]-1; j++) {
  //     for (int i = dom.local_min[0]; i <= dom.local_max[0]-1; i++) {
  //       kernel(i, j, k, g);
  //     }
  //   }
  // }

  LOG_DEBUG() << "Generating nested loop\n";
  SgScopeStatement *parent_block = block;
  indices.resize(s->getNumDim(), NULL);
  for (int i = s->getNumDim()-1; i >= 0; --i) {
    SgVariableDeclaration *index_decl =         
        ru::BuildVariableDeclaration(getLoopIndexName(i),
                                     sb::buildIntType());
    indices[i] = index_decl;
    if (ru::IsCLikeLanguage()) {
      si::appendStatement(index_decl, parent_block);
    }
    rose_util::AddASTAttribute<RunKernelIndexVarAttribute>(
        index_decl,  new RunKernelIndexVarAttribute(i+1));
    SgExpression *loop_begin =
        rt_builder_->BuildStencilDomMinRef(
            sb::buildVarRefExp(stencil_param), i+1);
    if (i == 0 && s->IsRedBlackVariant()) {
      loop_begin = sb::buildAddOp(
          loop_begin,
          BuildRedBlackInitOffset(loop_begin, indices,
                                  param->get_args()[1],
                                  s->getNumDim()));
    }
    SgInitializedName *loop_var = index_decl->get_variables()[0];
    // <= dom.local_max -1
    SgExpression *loop_end =
        sb::buildSubtractOp(
            rt_builder_->BuildStencilDomMaxRef(
                sb::buildVarRefExp(stencil_param), i+1),
            sb::buildIntVal(1));
    SgExpression *incr =
        sb::buildIntVal((i == 0 && s->IsRedBlackVariant()) ? 2 : 1);
    SgBasicBlock *inner_block = sb::buildBasicBlock();
    //SgForStatement *loop_statement = sb::buildForStatement(init, test, incr, inner_block);
    SgScopeStatement *loop_statement =
        ru::BuildForLoop(loop_var, loop_begin, loop_end, incr, inner_block);    
    si::appendStatement(loop_statement, parent_block);
    rose_util::AddASTAttribute(
        loop_statement,
        new RunKernelLoopAttribute(i+1));
    parent_block = inner_block;
  }

  SgExpressionPtrList index_args;
  for (int i = 0; i < s->getNumDim(); ++i) {
    index_args.push_back(sb::buildVarRefExp(indices[i]));
  }
  SgFunctionCallExp *kernelCall =
      BuildKernelCall(s, index_args, stencil_param);
  si::appendStatement(sb::buildExprStatement(kernelCall),
                      parent_block);
  return block;
}

// TODO: Move this to the RT builder
SgFunctionDeclaration *ReferenceTranslator::BuildRunKernel(StencilMap *s) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();

  // Stencil type
  SgType *stencil_type = NULL;
  if (ru::IsCLikeLanguage()) {
    stencil_type = sb::buildConstType(sb::buildPointerType(
          sb::buildConstType(s->stencil_type())));
  } else if (ru::IsFortranLikeLanguage()) {
    stencil_type = s->stencil_type();
  }    

  SgInitializedName *stencil_param =
      sb::buildInitializedName(getStencilArgName(), stencil_type);
  si::appendArg(parlist, stencil_param);
  
  if (s->IsRedBlackVariant()) {
    SgInitializedName *rb_param =
        sb::buildInitializedName(PS_STENCIL_MAP_RB_PARAM_NAME,
                                 sb::buildIntType());
    si::appendArg(parlist, rb_param);
  }
  
  SgFunctionDeclaration *runFunc = ru::BuildFunctionDeclaration(
      s->GetRunName(), sb::buildVoidType(), parlist, global_scope_);
  rose_util::SetFunctionStatic(runFunc);
  si::attachComment(runFunc, "Generated by " + string(__FUNCTION__));
  vector<SgVariableDeclaration*> indices;
  si::replaceStatement(runFunc->get_definition()->get_body(),
                       BuildRunKernelBody(s, parlist, indices));
  // Parameters and variable declarations need to be put forward in Fortran
  if (ru::IsFortranLikeLanguage()) {
    SgScopeStatement *body = runFunc->get_definition()->get_body();
    BOOST_FOREACH (SgVariableDeclaration *vd,
                   make_pair(indices.rbegin(), indices.rend())) {
      si::prependStatement(vd, body);
    }
    SgVariableDeclaration *vd = ru::BuildVariableDeclaration(
        stencil_param->get_name(), stencil_param->get_type(), NULL, body);
    LOG_DEBUG() << "sp type: " << stencil_param->get_type()->unparseToString()
                << "\n";
    si::prependStatement(vd, body);
  }
  rose_util::AddASTAttribute(runFunc,
                             new RunKernelAttribute(s, stencil_param));

  return runFunc;
}

void ReferenceTranslator::BuildRunBody(
    SgBasicBlock *block, Run *run, SgFunctionDeclaration *run_func) {
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
    if (s->IsRedBlackVariant()) {
      si::appendExpression(
          args,
          sb::buildIntVal(s->IsBlack() ? 1 : 0));
    }
    SgFunctionCallExp *c = sb::buildFunctionCallExp(fs, args);
    si::appendStatement(sb::buildExprStatement(c), loopBody);
    // Call both Red and Black versions for MapRedBlack
    if (s->IsRedBlack()) {
      args =
          sb::buildExprListExp(
              sb::buildAddressOfOp(si::copyExpression(stencil)),
              sb::buildIntVal(1));
      c = sb::buildFunctionCallExp(fs, args);
      si::appendStatement(sb::buildExprStatement(c), loopBody);
    }
    //appendGridSwap(s, stencilName, false, loopBody);
  }

  TraceStencilRun(run, loop, block);
  return;
}

void ReferenceTranslator::TraceStencilRun(Run *run,
                                          SgScopeStatement *loop,
                                          SgScopeStatement *cur_scope) {
  SgExpression *st_ptr = NULL;  
  if (config_.LookupFlag(Configuration::TRACE_KERNEL)) {
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
    st_ptr = sb::buildAddressOfOp(sb::buildVarRefExp(st_decl));  
    // Start the stopwatch
    rose_util::AppendExprStatement(cur_scope, BuildStopwatchStart(st_ptr));
  }

  // Enter the loop
  si::appendStatement(loop, cur_scope);

  if (config_.LookupFlag(Configuration::TRACE_KERNEL)) {
    // Stop the stopwatch and call the post trace function
    si::appendStatement(
        sb::buildVariableDeclaration(
            "f", sb::buildFloatType(),
            sb::buildAssignInitializer(
                BuildStopwatchStop(st_ptr), sb::buildFloatType()),
            cur_scope),
        cur_scope);
    rose_util::AppendExprStatement(
        cur_scope, BuildTraceStencilPost(sb::buildVarRefExp("f")));
    si::appendStatement(
        sb::buildReturnStmt(sb::buildVarRefExp("f")), cur_scope); /* return f; */
  } else {
    si::appendStatement(
        sb::buildReturnStmt(sb::buildFloatVal(0.0f)), cur_scope); /* return f; */
  }

  return;
}

SgFunctionDeclaration *ReferenceTranslator::BuildRun(Run *run) {
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
  //si::replaceStatement(fdef->get_body(), BuildRunBody(run,
  //runFunc));
  BuildRunBody(fdef->get_body(), run, runFunc);
  si::attachComment(runFunc, "Generated by " + string(__FUNCTION__));  
  return runFunc;
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

void ReferenceTranslator::TranslateRun(SgFunctionCallExp *node,
                                       Run *run) {
  // No translation necessary for Fortran binding
  if (ru::IsFortranLikeLanguage()) {
    return;
  }
  
  SgFunctionDeclaration *runFunc = BuildRun(run);
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

