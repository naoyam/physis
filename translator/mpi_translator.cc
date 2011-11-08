// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

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
    ReferenceTranslator(config), mpi_rt_builder_(NULL),
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
}

void MPITranslator::CheckSizes() {
  // Check the grid sizes and dimensions
  global_num_dims_ = 0;
  IntArray grid_max_size;
  FOREACH (it, tx_->grid_new_map().begin(), tx_->grid_new_map().end()) {
    Grid *grid = it->second;
    // all grids must have static size
    if (!grid->has_static_size()) {
      LOG_ERROR() << "Undefined grid size is not allowed in MPI translation\n";
      LOG_ERROR() << grid->toString() << "\n";
      PSAbort(1);
    }
    global_num_dims_ = std::max(global_num_dims_, grid->getNumDim());
    const IntVector& gsize = grid->static_size();
    for (unsigned i = 0; i < gsize.size(); ++i) {
      grid_max_size[i] = std::max(grid_max_size[i], gsize[i]);
    }
  }
  LOG_DEBUG() << "Global dimension: " << global_num_dims_ << "\n";
  LOG_DEBUG() << "Grid max size: " << grid_max_size << "\n";  

  // Check the domain sizes
  IntVector domain_max_point;
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
      const IntVector &minp = d->regular_domain().min_point();
      const IntVector &maxp = d->regular_domain().max_point();
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

void MPITranslator::Translate() {
  LOG_DEBUG() << "Translating to MPI\n";

  assert(stencil_run_func_ =
         si::lookupFunctionSymbolInParentScopes("__PSStencilRun",
                                                global_scope_));
  //CheckSizes();

  mpi_rt_builder_ = new MPIRuntimeBuilder(global_scope_);

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
          sb::buildVoidType(),
          sb::buildFunctionParameterList(client_func_params), global_scope_);
    rose_util::SetFunctionStatic(prototype);
    si::insertStatementBefore(
        si::findFirstDefiningFunctionDecl(global_scope_),
        prototype);
  }
  
  ReferenceTranslator::Translate();
  delete mpi_rt_builder_;
  mpi_rt_builder_ = NULL;
}

void MPITranslator::translateInit(SgFunctionCallExp *node) {
  LOG_DEBUG() << "Translating Init call\n";

  // Append the number of run calls
  int num_runs = tx_->run_map().size();
  node->append_arg(sb::buildIntVal(num_runs));

  // let the runtime know about the stencil client handlers
  SgFunctionParameterTypeList *client_func_params
      = sb::buildFunctionParameterTypeList
      (sb::buildIntType(),
       sb::buildPointerType(sb::buildPointerType(sb::buildVoidType())));
  SgFunctionType *client_func_type = sb::buildFunctionType(sb::buildVoidType(),
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
  tmp_block->append_statement(clients);
  node->append_arg(sb::buildVarRefExp(clients));

  tmp_block->append_statement(si::copyStatement(getContainingStatement(node)));
  si::replaceStatement(getContainingStatement(node), tmp_block);
  return;
}

void MPITranslator::translateRun(SgFunctionCallExp *node,
                                 Run *run) {
  SgFunctionDeclaration *runFunc = GenerateRun(run);
  si::insertStatementBefore(getContainingFunction(node), runFunc);
  
  // redirect the call to __PSStencilRun
  SgFunctionRefExp *ref = sb::buildFunctionRefExp(stencil_run_func_);
  node->set_function(ref);

  // build argument list
  SgBasicBlock *tmp_block = sb::buildBasicBlock();  
  SgExprListExp *args = sb::buildExprListExp();
  // runner id
  args->append_expression(sb::buildIntVal(run->id()));
  // iteration count
  if (run->count()) {
    args->append_expression(run->count());
  } else {
    args->append_expression(sb::buildIntVal(1));
  }
  // number of stencils
  args->append_expression(sb::buildIntVal(run->stencils().size()));

  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    SgExpression *stencil_arg = it->first;
    StencilMap *stencil = it->second;    
    SgType *stencil_type = stencil->stencil_type();    
    SgVariableDeclaration *sdecl
        = rose_util::buildVarDecl("s" + toString(i), stencil_type,
                                  stencil_arg, tmp_block);
    args->append_expression(sb::buildSizeOfOp(stencil_type));
    args->append_expression(sb::buildAddressOfOp(
        sb::buildVarRefExp(sdecl)));
  }

  // OPTIONAL: Is freeing the old argument list object necessary?
  node->set_args(args);

  tmp_block->append_statement(si::copyStatement
                              (getContainingStatement(node)));  
  si::replaceStatement(getContainingStatement(node), tmp_block);
}

void MPITranslator::GenerateLoadRemoteGridRegion(
    StencilMap *smap,
    SgExpression *stencil_ref,
    Run *run,
    SgScopeStatement *scope,
    SgInitializedNamePtrList &remote_grids,
    SgStatementPtrList &statements,
    bool &overlap_eligible,
    int &overlap_width) {
  string loop_var_name = "i";
  SgVarRefExp *loop_var_ref = sb::buildVarRefExp(loop_var_name);
  overlap_eligible = true;
  overlap_width = 0;
  SgIntVal *overlap_arg = sb::buildIntVal(0);
  // Ensure remote grid points available locally
  FOREACH (ait, smap->grid_params().begin(), smap->grid_params().end()) {
    SgInitializedName *grid_param = *ait;
    LOG_DEBUG() <<  "grid param: " << grid_param->unparseToString() << "\n";
    SgExpression *gvref = BuildStencilFieldRef(stencil_ref,
                                               grid_param->get_name());
    Kernel *kernel = tx_->findKernel(smap->getKernel());
    LOG_DEBUG() << "kernel: " << kernel->GetName() << "\n";
    if (!kernel->isGridParamRead(grid_param)) {
      LOG_DEBUG() << "Not read in this kernel\n";
      continue;
    }
    bool read_only = !kernel->isGridParamWritten(grid_param);
    // Read-only grids are loaded just once at the beginning of the
    // loop
    SgExpression *reuse;
    if (read_only) {
      reuse = sb::buildGreaterThanOp(loop_var_ref, sb::buildIntVal(0));
    } else {
      reuse = sb::buildIntVal(0);
    }
    
    StencilRange &sr = smap->GetStencilRange(grid_param);
    LOG_DEBUG() << "Argument stencil range: " << sr << "\n";
    
    if (sr.IsNeighborAccess() && sr.num_dims() == smap->getNumDim()) {
      // If both zero, skip
      if (sr.IsZero()) {
        LOG_DEBUG() << "Stencil just accesses own buffer; no exchange needed\n";
        continue;
      }
      // Create an inner scope for declaring variables
      SgBasicBlock *bb = sb::buildBasicBlock();
      //scope->append_statement(bb);
      statements.push_back(bb);
      SgFunctionCallExp *load_neighbor_call
          = BuildLoadNeighbor(gvref, sr, bb, reuse, overlap_arg);
      rose_util::AppendExprStatement(bb, load_neighbor_call);
      overlap_width = std::max(sr.GetMaxWidth(), overlap_width);
    } else {
      overlap_eligible = false;
      // generate LoadSubgrid call
      if (sr.IsUniqueDim()) {
        // rose_util::AppendExprStatement(
        //     scope, BuildCallLoadSubgridUniqueDim(gvref, sr, reuse));
        statements.push_back(
            sb::buildExprStatement(
                BuildCallLoadSubgridUniqueDim(gvref, sr, reuse)));
      } else {
        SgBasicBlock *tmp_block = sb::buildBasicBlock();
        //scope->append_statement(tmp_block);
        statements.push_back(tmp_block);
        SgVariableDeclaration *srv = sr.BuildPSGridRange("gr",
                                                         tmp_block);
        rose_util::AppendExprStatement(tmp_block,
                                       BuildCallLoadSubgrid(gvref, srv, reuse));
      }
      remote_grids.push_back(grid_param);
    }
  }

  if (overlap_eligible) {
    overlap_arg->set_value(1);
  }
}


void MPITranslator::DeactivateRemoteGrids(
    StencilMap *smap,
    SgExpression *stencil_ref,
    SgScopeStatement *scope,
    const SgInitializedNamePtrList &remote_grids) {
  //  FOREACH (gai, smap->grid_args().begin(),
  //  smap->grid_args().end()) {
  FOREACH (gai, remote_grids.begin(), remote_grids.end()) {
    SgInitializedName *gv = *gai;
    SgExpression *gvref = BuildStencilFieldRef(stencil_ref,
                                               gv->get_name());
    rose_util::AppendExprStatement(scope, BuildActivateRemoteGrid(gvref, false));
  }
}

void MPITranslator::FixGridAddresses(StencilMap *smap,
                                     SgExpression *stencil_ref,
                                     SgScopeStatement *scope) {
  SgClassDefinition *stencilDef = smap->GetStencilTypeDefinition();
  SgDeclarationStatementPtrList &members = stencilDef->get_members();
  // skip the first member, which is always the domain var of this
  // stencil
  SgVariableDeclaration *d = isSgVariableDeclaration(*members.begin());
  SgExpression *dom_var =
      BuildStencilFieldRef(stencil_ref, sb::buildVarRefExp(d));
  rose_util::AppendExprStatement(
      scope, mpi_rt_builder_->BuildDomainSetLocalSize(dom_var));
  
  FOREACH(it, members.begin(), members.end()) {
    SgVariableDeclaration *d = isSgVariableDeclaration(*it);
    assert(d);
    LOG_DEBUG() << "member: " << d->unparseToString() << "\n";
    if (!GridType::isGridType(d->get_definition()->get_type())) {
      continue;
    }
    SgExpression *grid_var =
        sb::buildArrowExp(stencil_ref, sb::buildVarRefExp(d));
    ++it;
    SgVariableDeclaration *grid_id = isSgVariableDeclaration(*it);
    LOG_DEBUG() << "grid var created\n";
    SgFunctionCallExp *grid_real_addr
        = mpi_rt_builder_->BuildGetGridByID(
            BuildStencilFieldRef(stencil_ref,
                                 sb::buildVarRefExp(grid_id)));
    scope->append_statement(sb::buildAssignStatement(grid_var,
                                                     grid_real_addr));
  }
}

void MPITranslator::ProcessStencilMap(StencilMap *smap,
                                      SgVarRefExp *stencils,
                                      int stencil_map_index,
                                      Run *run,
                                      SgScopeStatement *function_body,
                                      SgScopeStatement *loop_body) {
  string stencil_name = "s" + toString(stencil_map_index);
  SgExpression *idx = sb::buildIntVal(stencil_map_index);
  SgType *stencil_ptr_type = sb::buildPointerType(smap->stencil_type());
  SgAssignInitializer *init =
      sb::buildAssignInitializer(sb::buildPntrArrRefExp(stencils, idx),
                                 stencil_ptr_type);
  SgVariableDeclaration *sdecl
      = sb::buildVariableDeclaration(stencil_name, stencil_ptr_type,
                                     init, function_body);
  SgVarRefExp *stencil_var = sb::buildVarRefExp(sdecl);
  function_body->append_statement(sdecl);

  // run kernel function
  SgFunctionSymbol *fs = rose_util::getFunctionSymbol(smap->run());
  PSAssert(fs);
  SgInitializedNamePtrList remote_grids;
  SgStatementPtrList load_statements;
  bool overlap_eligible;
  int overlap_width;
  GenerateLoadRemoteGridRegion(smap, stencil_var, run, loop_body,
                               remote_grids, load_statements,
                               overlap_eligible, overlap_width);
  FOREACH (sit, load_statements.begin(), load_statements.end()) {
    loop_body->append_statement(*sit);
  }
    
  // Call the stencil kernel
  SgExprListExp *args = sb::buildExprListExp(stencil_var);
  SgFunctionCallExp *c = sb::buildFunctionCallExp(fs, args);
  loop_body->append_statement(sb::buildExprStatement(c));
  appendGridSwap(smap, stencil_var, loop_body);
  DeactivateRemoteGrids(smap, stencil_var, loop_body,
                        remote_grids);

  FixGridAddresses(smap, stencil_var, function_body);
}

SgBasicBlock *MPITranslator::BuildRunBody(Run *run) {
  SgBasicBlock *block = sb::buildBasicBlock();
  si::attachComment(block, "Generated by BuildRunBody");

  string stencil_param_name = "stencils";
  SgVarRefExp *stencils = sb::buildVarRefExp(stencil_param_name,
                                             block);
  
  // build main loop
  SgBasicBlock *loopBody = sb::buildBasicBlock();
  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    ProcessStencilMap(it->second, stencils, i, run,
                      block, loopBody);
  }
  SgVariableDeclaration *lv
      = sb::buildVariableDeclaration("i", sb::buildIntType(), NULL, block);
  block->append_statement(lv);
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

  //block->append_statement(loop);  
  TraceStencilRun(run, loop, block);

  return block;
}



SgFunctionDeclaration *MPITranslator::GenerateRun(Run *run) {
  // setup the parameter list
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  parlist->append_arg(sb::buildInitializedName("iter",
                                               sb::buildIntType()));
  SgType *stype = sb::buildPointerType(
      sb::buildPointerType(sb::buildVoidType()));
  parlist->append_arg(sb::buildInitializedName("stencils", stype));

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

SgExprListExp *MPITranslator::generateNewArg(
    GridType *gt, Grid *g, SgVariableDeclaration *dim_decl) {
  // Prepend the type specifier.
  SgExprListExp *new_args = ReferenceTranslator::generateNewArg(gt, g, dim_decl);
  new_args->prepend_expression(gt->BuildElementTypeExpr());
  return new_args;
}

void MPITranslator::appendNewArgExtra(SgExprListExp *args,
                                      Grid *g) {
  args->append_expression(rose_util::buildNULL());
  SgExpression *attr = g->BuildAttributeExpr();
  if (!attr) attr = sb::buildIntVal(0);
  args->append_expression(attr);
  return;
}

// REFACTORING: This should be common among all translators.
bool MPITranslator::translateGetHost(SgFunctionCallExp *node,
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

  SgFunctionCallExp *get_call = ref_rt_builder_->BuildGridGet(
      g, indices, gt->getElmType());
  si::replaceExpression(node, get_call, true);
  return true;
}

bool MPITranslator::translateGetKernel(SgFunctionCallExp *node,
                                       SgInitializedName *gv) {
  // 
  // *((gt->getElmType())__PSGridGetAddressND(g, x, y, z))

  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = getContainingScopeStatement(node);  
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  
  string get_address_name = get_addr_name_ +  GetTypeDimName(gt);
  string get_address_no_halo_name = get_addr_no_halo_name_ +  GetTypeDimName(gt);
  SgFunctionRefExp *get_address;
  const StencilIndexList *sil = tx_->findStencilIndex(node);
  PSAssert(sil);
  LOG_DEBUG() << "Stencil index: " << *sil;
  if (StencilIndexSelf(*sil, nd)) {
    get_address = sb::buildFunctionRefExp(get_address_no_halo_name,

                                          global_scope_);
  } else {
    get_address = sb::buildFunctionRefExp(get_address_name,
                                          global_scope_);
  }
  SgExprListExp *args = sb::buildExprListExp(g);
  FOREACH (it, node->get_args()->get_expressions().begin(),
           node->get_args()->get_expressions().end()) {
    args->append_expression(si::copyExpression(*it));
  }

  SgFunctionCallExp *get_address_exp
      = sb::buildFunctionCallExp(get_address, args);
  //get_address_exp->addNewAttribute("StencilIndexList",
  //new StencilIndexAttribute(*sil));
  get_address_exp->setAttribute("StencilIndexList",
                                new StencilIndexAttribute(*sil));
  AstAttributeMechanism *am = get_address_exp->get_attributeMechanism();
  FOREACH (it, am->begin(), am->end()) {
    LOG_DEBUG() << it->first << "->" << it->second << "\n";
  }
  SgExpression *x = sb::buildPointerDerefExp(get_address_exp);
  si::replaceExpression(node, x);
  return true;
}

void MPITranslator::translateEmit(SgFunctionCallExp *node,
                                  SgInitializedName *gv) {
  // *((gt->getElmType())__PSGridGetAddressND(g, x, y, z)) = v

  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = getContainingScopeStatement(node);  
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);

  string get_address_name = emit_addr_name_ +  GetTypeDimName(gt);  
  SgFunctionRefExp *get_address = sb::buildFunctionRefExp(get_address_name,
                                                          global_scope_);
  SgExprListExp *args = sb::buildExprListExp(g);
  SgInitializedNamePtrList &params = getContainingFunction(node)->get_args();
  FOREACH(it, params.begin(), params.begin() + nd) {
    SgInitializedName *p = *it;
    args->append_expression(sb::buildVarRefExp(p, getContainingScopeStatement(node)));
  }

  SgFunctionCallExp *get_address_exp
      = sb::buildFunctionCallExp(get_address, args);
  
  SgExpression *lhs =
      sb::buildPointerDerefExp(get_address_exp);
  SgExpression *rhs =
      si::copyExpression(node->get_args()->get_expressions()[0]);

  SgExpression *emit = sb::buildAssignOp(lhs, rhs);
  si::replaceExpression(node, emit);
}

} // namespace translator
} // namespace physis
