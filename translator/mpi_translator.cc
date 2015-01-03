// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/mpi_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/reference_runtime_builder.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

MPITranslator::MPITranslator(const Configuration &config):
    ReferenceTranslator(config),
    flag_mpi_overlap_(false) {
  grid_type_name_ = "__PSGridMPI";
  grid_create_name_ = "__PSGridNewMPI";
  target_specific_macro_ = "PHYSIS_MPI";
  get_addr_name_ = "__PSGridGetAddr";
  get_addr_no_halo_name_ = "__PSGridGetAddrNoHalo";
  emit_addr_name_ = "__PSGridEmitAddr";
  
  const pu::LuaValue *lv
      = config.Lookup(Configuration::MPI_OVERLAP);
  if (lv) {
    PSAssert(lv->get(flag_mpi_overlap_));
  }
  if (flag_mpi_overlap_) {
    LOG_INFO() << "Overlapping enabled\n";
  }
  
  validate_ast_ = true;
}
#if 0
void MPITranslator::CheckSizes() {
  // Check the grid sizes and dimensions
  global_num_dims_ = 0;
  SizeArray grid_max_size;
  FOREACH (it, tx_->grid_new_map().begin(), tx_->grid_new_map().end()) {
    Grid *grid = it->second;
    // all grids must have static size
    if (!grid->has_static_size()) {
      LOG_ERROR() << "Undefined grid size is not allowed in MPI translation\n";
      LOG_ERROR() << grid->toString() << "\n";
      PSAbort(1);
    }
    global_num_dims_ = std::max(global_num_dims_, grid->getNumDim());
    const SizeVector& gsize = grid->static_size();
    for (unsigned i = 0; i < gsize.size(); ++i) {
      grid_max_size[i] = std::max(grid_max_size[i], gsize[i]);
    }
  }
  LOG_DEBUG() << "Global dimension: " << global_num_dims_ << "\n";
  LOG_DEBUG() << "Grid max size: " << grid_max_size << "\n";  

  // Check the domain sizes
  SizeVector domain_max_point;
  for (int i = 0; i < PS_MAX_DIM; ++i) {
    domain_max_point.push_back(0);
  }
  FOREACH (it, tx_->domain_map().begin(), tx_->domain_map().end()) {
    //const SgExpression *exp = it->first;
    const DomainSet &ds = it->second;
    FOREACH (dsit, ds.begin(), ds.end()) {
      Domain *d = *dsit;
      if (!d->has_static_constant_size()) {
        LOG_ERROR() << "Undefined domain size is not allowed in MPI translation\n";
        LOG_ERROR() << d->toString() << "\n";
        PSAbort(1);
      }
      const SizeVector &minp = d->regular_domain().min_point();
      const SizeVector &maxp = d->regular_domain().max_point();
      for (int i = 0; i < d->num_dims(); ++i) {
        PSAssert(minp[i] >= 0);
        domain_max_point[i] = std::max(domain_max_point[i], maxp[i]);
      }
    }
  }
  LOG_DEBUG() << "Domain max size: " << domain_max_point << "\n";

  // May be a good idea to check this, but grids with a smaller number
  // of dimensions are always determined smaller than domains with a
  // larger number of dimensions.
  // PSAssert(domain_max_point <=  grid_max_size);
  
  //global_size_ = grid_max_size;

  //LOG_DEBUG() << "Global size: " << global_size_ << "\n";
}
#endif

void MPITranslator::Translate() {
  LOG_DEBUG() << "Translating to MPI\n";

  assert(stencil_run_func_ =
         si::lookupFunctionSymbolInParentScopes("__PSStencilRun",
                                                global_scope_));
  //CheckSizes();

  // Insert prototypes of stencil run functions
  FOREACH (it, tx_->run_map().begin(), tx_->run_map().end()) {
    Run *r = it->second;
    SgFunctionParameterTypeList *client_func_params
        = sb::buildFunctionParameterTypeList
        (sb::buildIntType(),
         sb::buildPointerType(sb::buildPointerType(sb::buildVoidType())));
    SgFunctionDeclaration *prototype =
        sb::buildNondefiningFunctionDeclaration(
            r->GetName(),
            sb::buildFloatType(),
            sb::buildFunctionParameterList(client_func_params), global_scope_);
    rose_util::SetFunctionStatic(prototype);
    si::insertStatementBefore(
        si::findFirstDefiningFunctionDecl(global_scope_),
        prototype);
  }
  
  ReferenceTranslator::Translate();
}

void MPITranslator::TranslateInit(SgFunctionCallExp *node) {
  LOG_DEBUG() << "Translating Init call\n";

  // Append the number of run calls
  int num_runs = tx_->run_map().size();
  si::appendExpression(node->get_args(),
                       sb::buildIntVal(num_runs));

  // let the runtime know about the stencil client handlers
  SgFunctionParameterTypeList *client_func_params
      = sb::buildFunctionParameterTypeList
      (sb::buildIntType(),
       sb::buildPointerType(sb::buildPointerType(sb::buildVoidType())));
  SgFunctionType *client_func_type = sb::buildFunctionType(sb::buildFloatType(),
                                                           client_func_params);
  vector<SgExpression*> client_func_exprs;
  client_func_exprs.resize(num_runs, NULL);
  FOREACH (it, tx_->run_map().begin(), tx_->run_map().end()) {
    const Run* run = it->second;
    SgFunctionRefExp *fref = sb::buildFunctionRefExp(run->GetName(),
                                                     client_func_type);
    client_func_exprs[run->id()] = fref;
  }
  SgInitializer *ai = NULL;
  if (client_func_exprs.size()) {
    ai = sb::buildAggregateInitializer(
        sb::buildExprListExp(client_func_exprs));
  } else {
    //ai = sb::buildAssignInitializer(rose_util::buildNULL());
    ai = sb::buildAggregateInitializer(
        sb::buildExprListExp());
  }
  SgBasicBlock *tmp_block = sb::buildBasicBlock();
  SgVariableDeclaration *clients
      = sb::buildVariableDeclaration("stencil_clients",
                                     sb::buildArrayType(sb::buildPointerType(client_func_type)),
                                     ai, tmp_block);
  PSAssert(clients);
  si::appendStatement(clients, tmp_block);
  si::appendExpression(node->get_args(),
                       sb::buildVarRefExp(clients));

  si::appendStatement(
      si::copyStatement(getContainingStatement(node)),
      tmp_block);
  si::replaceStatement(getContainingStatement(node), tmp_block);
  return;
}

void MPITranslator::TranslateRun(SgFunctionCallExp *node,
                                 Run *run) {
  SgFunctionDeclaration *runFunc = builder()->BuildRunFunc(run);
  si::insertStatementBefore(getContainingFunction(node), runFunc);
  
  // redirect the call to __PSStencilRun
  SgFunctionRefExp *ref = sb::buildFunctionRefExp(stencil_run_func_);
  //node->set_function(ref);

  // build argument list
  SgBasicBlock *tmp_block = sb::buildBasicBlock();  
  SgExprListExp *args = sb::buildExprListExp();
  SgExpressionPtrList &original_args =
      node->get_args()->get_expressions();
  // runner id
  si::appendExpression(args, sb::buildIntVal(run->id()));
  // iteration count
  SgExpression *count_exp = run->BuildCount();
  if (!count_exp) {
    count_exp = sb::buildIntVal(1);
  }
  si::appendExpression(args, count_exp);
  // number of stencils
  si::appendExpression(args,
                       sb::buildIntVal(run->stencils().size()));
  
  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    //SgExpression *stencil_arg = it->first;
    SgExpression *stencil_arg =
        si::copyExpression(original_args.at(i));
    StencilMap *stencil = it->second;    
    SgType *stencil_type = stencil->stencil_type();    
    SgVariableDeclaration *sdecl
        = rose_util::buildVarDecl("s" + toString(i), stencil_type,
                                  stencil_arg, tmp_block);
    si::appendExpression(args, sb::buildSizeOfOp(stencil_type));
    si::appendExpression(args, sb::buildAddressOfOp(
        sb::buildVarRefExp(sdecl)));
  }

  si::appendStatement(
      sb::buildExprStatement(sb::buildFunctionCallExp(ref, args)),
      tmp_block);

  si::replaceStatement(getContainingStatement(node), tmp_block);
}

SgExprListExp *MPITranslator::generateNewArg(
    GridType *gt, Grid *g, SgVariableDeclaration *dim_decl) {
  // Prepend the type specifier.
  SgExprListExp *ref_args =
      ReferenceTranslator::generateNewArg(gt, g, dim_decl);
  SgExprListExp *new_args =
      sb::buildExprListExp(gt->BuildElementTypeExpr());
  FOREACH (it, ref_args->get_expressions().begin(),
           ref_args->get_expressions().end()) {
    si::appendExpression(new_args, si::copyExpression(*it));
  }
  si::deleteAST(ref_args);
  return new_args;
}

void MPITranslator::appendNewArgExtra(SgExprListExp *args,
                                      Grid *g,
                                      SgVariableDeclaration *dim_decl) {
  LOG_DEBUG() << "Append New extra arg for "
              << *g << "\n";
  // attribute
  si::appendExpression(args, sb::buildIntVal(0));
  // global offset
  si::appendExpression(args, rose_util::buildNULL(global_scope_));

  const StencilRange &sr = g->stencil_range();

  // ROSE-EDG3 fails to unparse this AST in C++ mode. According to a
  // message to the ROSE ML, ROSE-EDG3 seems to have some problems in
  // unparsing C++ code.
  // https://mailman.nersc.gov/pipermail/rose-public/2011-July/001063.html
  // This may be solved in the latest EDG4 version.
  SgExprListExp *stencil_min_val =
      builder()->BuildStencilOffsetMin(sr);
  if (stencil_min_val == NULL) {
    LOG_ERROR() << "Analyzing stencil for finding left-most offset failed\n";
    PSAbort(1);
  }
  SgVariableDeclaration *stencil_min_var
      = sb::buildVariableDeclaration(
          "stencil_offset_min", ivec_type_,
          sb::buildAggregateInitializer(stencil_min_val, ivec_type_));
  si::insertStatementAfter(dim_decl, stencil_min_var);
  si::appendExpression(args, sb::buildVarRefExp(stencil_min_var));

  SgExprListExp *stencil_max_val =
      builder()->BuildStencilOffsetMax(sr);
  if (stencil_max_val == NULL) {
    LOG_ERROR() << "Analyzing stencil for finding right-most offset failed\n";
    PSAbort(1);
  }
  SgVariableDeclaration *stencil_max_var
      = sb::buildVariableDeclaration(
          "stencil_offset_max", ivec_type_,
          sb::buildAggregateInitializer(stencil_max_val, ivec_type_));
  si::insertStatementAfter(dim_decl, stencil_max_var);
  si::appendExpression(args, sb::buildVarRefExp(stencil_max_var));
  
  return;
}
#if 0
// REFACTORING: This should be common among all translators.
bool MPITranslator::TranslateGetHost(SgFunctionCallExp *node,
                                     SgInitializedName *gv) {
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->rank();
  SgScopeStatement *scope = getContainingScopeStatement(node);    
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  SgExpressionPtrList &args = node->get_args()->get_expressions();
  SgExpressionPtrList indices;
  FOREACH (it, args.begin(), args.begin() + nd) {
    indices.push_back(*it);
  }

  SgExpression *get = rt_builder_->BuildGridGet(
      g, rose_util::GetASTAttribute<GridVarAttribute>(gv),
      gt, &indices, NULL, false, false);
  si::replaceExpression(node, get, true);
  rose_util::CopyASTAttribute<GridGetAttribute>(
      get, node, false);  
  return true;
}

bool MPITranslator::TranslateGetKernel(SgFunctionCallExp *node,
                                       SgInitializedName *gv,
                                       bool is_periodic) {
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->rank();
  SgScopeStatement *scope = si::getEnclosingFunctionDefinition(node);
  SgExpressionPtrList args;
  rose_util::CopyExpressionPtrList(
      node->get_args()->get_expressions(), args);
  SgExpression *offset = rt_builder_->BuildGridOffset(
      sb::buildVarRefExp(gv->get_name(), si::getScope(node)),
      nd, &args,
      true, is_periodic,
      rose_util::GetASTAttribute<GridGetAttribute>(
          node)->GetStencilIndexList());

  SgFunctionCallExp *base_addr = sb::buildFunctionCallExp(
      si::lookupFunctionSymbolInParentScopes("__PSGridGetBaseAddr"),
      sb::buildExprListExp(sb::buildVarRefExp(gv->get_name(),
                                              scope)));
  SgExpression *x = sb::buildPntrArrRefExp(
      sb::buildCastExp(base_addr,
                       sb::buildPointerType(gt->point_type())),
      offset);
  rose_util::CopyASTAttribute<GridGetAttribute>(x, node);
  si::replaceExpression(node, x);
  return true;
}
#endif
#if 0
void MPITranslator::TranslateEmit(SgFunctionCallExp *node,
                                  GridEmitAttribute *attr) {
  // *((gt->getElmType())__PSGridGetAddressND(g, x, y, z)) = v
  
  GridType *gt = attr->gt();
  int nd = gt->rank();
  SgScopeStatement *scope = si::getEnclosingFunctionDefinition(node);
  
  SgInitializedNamePtrList &params =
      getContainingFunction(node)->get_args();
  SgExpressionPtrList args;
  for (int i = 0; i < nd; ++i) {
    SgInitializedName *p = params[i];
    args.push_back(sb::buildVarRefExp(p));
  }

  StencilIndexList sil;
  StencilIndexListInitSelf(sil, nd);
  SgExpression *offset = rt_builder_->BuildGridOffset(
      sb::buildVarRefExp(attr->gv()->get_name(), si::getScope(node)),
      nd, &args, &sil, true, false);
  SgFunctionCallExp *base_addr = sb::buildFunctionCallExp(
      si::lookupFunctionSymbolInParentScopes("__PSGridGetBaseAddr"),
      sb::buildExprListExp(sb::buildVarRefExp(attr->gv()->get_name(),
                                              scope)));
  SgExpression *lhs = sb::buildPntrArrRefExp(
      sb::buildCastExp(base_addr,
                       sb::buildPointerType(gt->point_type())),
      offset);

  if (attr->is_member_access()) {
    lhs = sb::buildDotExp(lhs, sb::buildVarRefExp(attr->member_name()));
    const vector<string> &array_offsets = attr->array_offsets();
    FOREACH (it, array_offsets.begin(), array_offsets.end()) {
      LOG_DEBUG() << "Parsing '" << *it << "'\n";
      SgExpression *e = rose_util::ParseString(
          *it, si::getScope(node));
      lhs = sb::buildPntrArrRefExp(lhs, e);
    }
  }

  SgExpression *emit_val =
      si::copyExpression(node->get_args()->get_expressions().back());

  SgExpression *emit = sb::buildAssignOp(lhs, emit_val);
  
  si::replaceExpression(node, emit);

  if (attr->is_member_access()) {
    RemoveEmitDummyExp(emit);
  }
}
#endif
void MPITranslator::FixAST() {
  if (validate_ast_) {
    si::fixVariableReferences(project_);
  }
}

} // namespace translator
} // namespace physis
