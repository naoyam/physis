// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/mpi_runtime_builder.h"
#include "translator/kernel.h"
#include "translator/rose_util.h"

namespace sb = SageBuilder;
namespace si = SageInterface;
namespace ru = physis::translator::rose_util;

namespace physis {
namespace translator {

SgFunctionCallExp *BuildCallLoadSubgrid(SgExpression *grid_ref,
                                        SgVariableDeclaration *grid_range,
                                        SgExpression *reuse) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSLoadSubgrid");
  SgExprListExp *args = sb::buildExprListExp(
      grid_ref, sb::buildAddressOfOp(sb::buildVarRefExp(grid_range)),
      reuse);
  SgFunctionCallExp *call = sb::buildFunctionCallExp(fs, args);
  return call;
}


SgFunctionCallExp *BuildCallLoadSubgridUniqueDim(SgExpression *gref,
                                                 StencilRange &sr,
                                                 SgExpression *reuse) {
  SgExprListExp *args = sb::buildExprListExp(gref);
  for (int i = 0; i < sr.num_dims(); ++i) {
    PSAssert(sr.min_indices()[i].size() == 1);
    si::appendExpression(
        args,
        rose_util::BuildIntLikeVal(sr.min_indices()[i].front().dim));
    si::appendExpression(
        args,
        rose_util::BuildIntLikeVal(sr.min_indices()[i].front().offset));
  }
  for (int i = 0; i < sr.num_dims(); ++i) {
    PSAssert(sr.max_indices()[i].size() == 1);
    si::appendExpression(
        args,
        rose_util::BuildIntLikeVal(sr.max_indices()[i].front().dim));
    si::appendExpression(
        args,
        rose_util::BuildIntLikeVal(sr.max_indices()[i].front().offset));
  }
  si::appendExpression(args, reuse);
  
  SgFunctionCallExp *call;
  SgFunctionSymbol *fs = NULL;
  if (sr.num_dims() == 3) {
    fs = si::lookupFunctionSymbolInParentScopes("__PSLoadSubgrid3D");
  } else if (sr.num_dims() == 2) {
    fs = si::lookupFunctionSymbolInParentScopes("__PSLoadSubgrid2D");    
  } else {
    PSAbort(1);
  }
  PSAssert(fs);
  call = sb::buildFunctionCallExp(fs, args);  
  return call;
}

SgFunctionCallExp *BuildLoadNeighbor(SgExpression *grid_var,
                                     StencilRange &sr,
                                     SgScopeStatement *scope,
                                     SgExpression *reuse,
                                     SgExpression *overlap,
                                     bool is_periodic) {
  SgFunctionSymbol *load_neighbor_func
      = si::lookupFunctionSymbolInParentScopes("__PSLoadNeighbor");
  IntVector offset_min, offset_max;
  PSAssert(sr.GetNeighborAccess(offset_min, offset_max));
  SgVariableDeclaration *offset_min_decl =
      rose_util::DeclarePSVectorInt("offset_min", offset_min, scope);
  SgVariableDeclaration *offset_max_decl =
      rose_util::DeclarePSVectorInt("offset_max", offset_max, scope);
  bool diag_needed = sr.IsNeighborAccessDiagonalAccessed();
  SgExprListExp *load_neighbor_args =
      sb::buildExprListExp(grid_var,
                           sb::buildVarRefExp(offset_min_decl),                           
                           sb::buildVarRefExp(offset_max_decl),
                           sb::buildIntVal(diag_needed),
                           reuse, overlap,
                           sb::buildIntVal(is_periodic));
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(load_neighbor_func,
                                                   load_neighbor_args);
  return fc;
}

SgFunctionCallExp *BuildActivateRemoteGrid(SgExpression *grid_var,
                                           bool active) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSActivateRemoteGrid");
  SgExprListExp *args =
      sb::buildExprListExp(grid_var, sb::buildIntVal(active));
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(fs, args);
  return fc;
}

SgFunctionCallExp *MPIRuntimeBuilder::BuildIsRoot() {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSIsRoot");
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(fs);
  return fc;
}

SgFunctionCallExp *MPIRuntimeBuilder::BuildGetGridByID(SgExpression *id_exp) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGetGridByID");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(id_exp));
  return fc;
}

SgFunctionCallExp *MPIRuntimeBuilder::BuildDomainSetLocalSize(
    SgExpression *dom) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSDomainSetLocalSize");
  if (!si::isPointerType(dom->get_type())) {
    dom = sb::buildAddressOfOp(dom);
  }
  SgExprListExp *args = sb::buildExprListExp(dom);
  SgFunctionCallExp *fc = sb::buildFunctionCallExp(fs, args);
  return fc;
}

SgExpression *MPIRuntimeBuilder::BuildGridBaseAddr(
    SgExpression *gvref, SgType *point_type) {
  SgExpression *base_addr = sb::buildFunctionCallExp(
      si::lookupFunctionSymbolInParentScopes(PS_GRID_GET_BASE_ADDR),
      sb::buildExprListExp(si::copyExpression(gvref)));
  base_addr = sb::buildCastExp(base_addr,
                               sb::buildPointerType(point_type));
  return base_addr;
}

void MPIRuntimeBuilder::BuildRunFuncBody(
    Run *run, SgFunctionDeclaration *run_func) {
  SgBasicBlock *block = run_func->get_definition()->get_body();  
  si::attachComment(block, "Generated by " + string(__FUNCTION__));

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
  si::appendStatement(lv, block);
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

  //si::appendStatement(loop, block);  
  TraceStencilRun(run, loop, block);
  
  return;
}

void MPIRuntimeBuilder::ProcessStencilMap(StencilMap *smap,
                                          SgVarRefExp *stencils,
                                          int stencil_map_index,
                                          Run *run,
                                          SgScopeStatement *function_body,
                                          SgScopeStatement *loop_body) {
  string stencil_name = string(PS_STENCIL_MAP_STENCIL_PARAM_NAME)
      + toString(stencil_map_index);
  SgExpression *idx = sb::buildIntVal(stencil_map_index);
  SgType *stencil_ptr_type = sb::buildPointerType(smap->stencil_type());
  SgAssignInitializer *init =
      sb::buildAssignInitializer(sb::buildPntrArrRefExp(stencils, idx),
                                 stencil_ptr_type);
  SgVariableDeclaration *sdecl
      = sb::buildVariableDeclaration(stencil_name, stencil_ptr_type,
                                     init, function_body);
  si::appendStatement(sdecl, function_body);

  // run kernel function
  SgFunctionSymbol *fs = ru::getFunctionSymbol(smap->run());
  PSAssert(fs);
  SgInitializedNamePtrList remote_grids;
  SgStatementPtrList load_statements;
  bool overlap_eligible;
  int overlap_width;
  BuildLoadRemoteGridRegion(smap, sdecl, run,
                            remote_grids, load_statements,
                            overlap_eligible, overlap_width);
  FOREACH (sit, load_statements.begin(), load_statements.end()) {
    si::appendStatement(*sit, loop_body);
  }
    
  // Call the stencil kernel
  SgExprListExp *args = sb::buildExprListExp(
      sb::buildVarRefExp(sdecl));
  SgFunctionCallExp *c = sb::buildFunctionCallExp(fs, args);
  si::appendStatement(sb::buildExprStatement(c), loop_body);
  SgStatementPtrList stmt_lists;
  BuildDeactivateRemoteGrids(smap, sdecl, remote_grids, stmt_lists);
  BOOST_FOREACH(SgStatement *stmt, stmt_lists) {
    si::appendStatement(stmt, loop_body);
  }

  BuildFixGridAddresses(smap, sdecl, function_body);
}

void MPIRuntimeBuilder::BuildDeactivateRemoteGrids(
    StencilMap *smap,
    SgVariableDeclaration *stencil_decl,
    const SgInitializedNamePtrList &remote_grids,
    SgStatementPtrList &statements) {
  FOREACH (gai, remote_grids.begin(), remote_grids.end()) {
    SgInitializedName *gv = *gai;
    SgExpression *gvref = BuildStencilFieldRef(
        sb::buildVarRefExp(stencil_decl),
        gv->get_name());
    statements.push_back(
        sb::buildExprStatement(BuildActivateRemoteGrid(gvref, false)));
  }
}

void MPIRuntimeBuilder::BuildFixGridAddresses(StencilMap *smap,
                                              SgVariableDeclaration *stencil_decl,
                                              SgScopeStatement *scope) {
  SgClassDefinition *stencilDef = smap->GetStencilTypeDefinition();
  SgDeclarationStatementPtrList &members = stencilDef->get_members();
  // skip the first member, which is always the domain var of this
  // stencil
  SgVariableDeclaration *d = isSgVariableDeclaration(*members.begin());
  SgExpression *dom_var =
      BuildStencilFieldRef(sb::buildVarRefExp(stencil_decl),
                                      sb::buildVarRefExp(d));
  rose_util::AppendExprStatement(
      scope, BuildDomainSetLocalSize(dom_var));
  
  FOREACH(it, members.begin(), members.end()) {
    SgVariableDeclaration *d = isSgVariableDeclaration(*it);
    assert(d);
    LOG_DEBUG() << "member: " << d->unparseToString() << "\n";
    if (!GridType::isGridType(d->get_definition()->get_type())) {
      continue;
    }
    SgExpression *grid_var =
        sb::buildArrowExp(sb::buildVarRefExp(stencil_decl),
                          sb::buildVarRefExp(d));
    ++it;
    SgVariableDeclaration *grid_id = isSgVariableDeclaration(*it);
    LOG_DEBUG() << "grid var created\n";
    SgFunctionCallExp *grid_real_addr
        = BuildGetGridByID(
            BuildStencilFieldRef(sb::buildVarRefExp(stencil_decl),
                                 sb::buildVarRefExp(grid_id)));
    si::appendStatement(
        sb::buildAssignStatement(grid_var, grid_real_addr),
        scope);
  }
}

void MPIRuntimeBuilder::BuildLoadRemoteGridRegion(
    StencilMap *smap,
    SgVariableDeclaration *stencil_decl,
    Run *run,
    SgInitializedNamePtrList &remote_grids,
    SgStatementPtrList &statements,
    bool &overlap_eligible,
    int &overlap_width) {
  string loop_var_name = "i";
  overlap_eligible = true;
  overlap_width = 0;
  vector<SgIntVal*> overlap_flags;
  // Ensure remote grid points available locally
  FOREACH (ait, smap->grid_params().begin(), smap->grid_params().end()) {

    SgInitializedName *grid_param = *ait;
    LOG_DEBUG() <<  "grid param: " << grid_param->unparseToString() << "\n";
    Kernel *kernel = ru::GetASTAttribute<Kernel>(smap->getKernel());
    LOG_DEBUG() << "kernel: " << kernel->GetName() << "\n";
    if (!kernel->isGridParamRead(grid_param)) {
      LOG_DEBUG() << "Not read in this kernel\n";
      continue;
    }
    bool read_only = !kernel->isGridParamWritten(grid_param);
    // Read-only grids are loaded just once at the beginning of the
    // loop
    SgExpression *reuse = NULL;
    if (read_only) {
      reuse = sb::buildGreaterThanOp(sb::buildVarRefExp(loop_var_name),
                                     sb::buildIntVal(0));
    } else {
      reuse = sb::buildIntVal(0);
    }

#if 0    
    StencilRange &sr = smap->GetStencilRange(grid_param);
#else
    GridVarAttribute *gva =
        rose_util::GetASTAttribute<GridVarAttribute>(grid_param);
    StencilRange &sr = gva->sr();
#endif
    LOG_DEBUG() << "Argument stencil range: " << sr << "\n";

    bool is_periodic = smap->IsGridPeriodic(grid_param);
    LOG_DEBUG() << "Periodic boundary?: " << is_periodic << "\n";

    SgExpression *gvref = BuildStencilFieldRef(
        sb::buildVarRefExp(stencil_decl),
        grid_param->get_name());
    
    if (sr.IsNeighborAccess() && sr.num_dims() == smap->getNumDim()) {
      // If both zero, skip
      if (sr.IsZero()) {
        LOG_DEBUG() << "Stencil just accesses own buffer; no exchange needed\n";
        continue;
      }
      // Create an inner scope for declaring variables
      SgBasicBlock *bb = sb::buildBasicBlock();
      statements.push_back(bb);
      SgIntVal *overlap_arg = sb::buildIntVal(0);
      overlap_flags.push_back(overlap_arg);
      SgFunctionCallExp *load_neighbor_call
          = BuildLoadNeighbor(gvref, sr, bb, reuse, overlap_arg,
                              is_periodic);
      rose_util::AppendExprStatement(bb, load_neighbor_call);
      overlap_width = std::max(sr.GetMaxWidth(), overlap_width);
    } else {
      // EXPERIMENTAL
      LOG_WARNING()
          << "Support of this type of stencils is experimental,\n"
          << "and may not work as expected. For example,\n"
          << "periodic access is not supported.\n";
      PSAssert(!is_periodic);
      overlap_eligible = false;
      // generate LoadSubgrid call
      if (sr.IsUniqueDim()) {
        statements.push_back(
            sb::buildExprStatement(
                BuildCallLoadSubgridUniqueDim(gvref, sr, reuse)));
      } else {
        SgBasicBlock *tmp_block = sb::buildBasicBlock();
        statements.push_back(tmp_block);
        SgVariableDeclaration *srv = sr.BuildPSGridRange("gr",
                                                         tmp_block);
        rose_util::AppendExprStatement(tmp_block,
                                       BuildCallLoadSubgrid(gvref, srv, reuse));
      }
      remote_grids.push_back(grid_param);
    }
  }

  FOREACH (it, overlap_flags.begin(), overlap_flags.end()) {
    (*it)->set_value(overlap_eligible ? 1 : 0);
  }
}

SgFunctionParameterList *MPIRuntimeBuilder::BuildRunFuncParameterList(Run *run) {
  SgFunctionParameterList *parlist = sb::buildFunctionParameterList();
  si::appendArg(parlist,
                sb::buildInitializedName("iter",
                                         sb::buildIntType()));
  SgType *stype = sb::buildPointerType(
      sb::buildPointerType(sb::buildVoidType()));
  si::appendArg(parlist,
                sb::buildInitializedName("stencils", stype));
  return parlist;
}


} // namespace translator
} // namespace physis
