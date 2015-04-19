// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/mpi_cuda_runtime_builder.h"

#include <string>
#include <vector>

#include "translator/cuda_util.h"

namespace sb = SageBuilder;
namespace si = SageInterface;
namespace ru = physis::translator::rose_util;
namespace cu = physis::translator::cuda_util;

using std::string;
using std::vector;

namespace physis {
namespace translator {

SgExpression *MPICUDARuntimeBuilder::BuildGridBaseAddr(
    SgExpression *gvref, SgType *point_type) {
  return ReferenceRuntimeBuilder::BuildGridBaseAddr(gvref, point_type);
}

SgExpression *MPICUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic) {
  return cuda_rt_builder_->BuildGridGet(gvref, gva, gt, offset_exprs,
                                        sil, is_kernel, is_periodic);
}

SgExpression *MPICUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,                    
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic,
    const string &member_name) {
  return cuda_rt_builder_->BuildGridGet(gvref, gva, gt, offset_exprs,
                                        sil, is_kernel, is_periodic, member_name);
}

SgExpression *MPICUDARuntimeBuilder::BuildGridGet(
    SgExpression *gvref,
    GridVarAttribute *gva,      
    GridType *gt,
    const SgExpressionPtrList *offset_exprs,
    const StencilIndexList *sil,
    bool is_kernel,
    bool is_periodic,
    const string &member_name,
    const SgExpressionVector &array_indices) {
  return cuda_rt_builder_->BuildGridGet(gvref, gva, gt, offset_exprs,
                                        sil, is_kernel, is_periodic,
                                        member_name, array_indices);
}

SgExpression *MPICUDARuntimeBuilder::BuildGridGetDev(SgExpression *grid_var,
                                                     GridType *gt) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(PS_GRID_GET_DEV_NAME);
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(grid_var));
  return fc;
}

SgFunctionCallExp *MPICUDARuntimeBuilder::BuildGetLocalSize(SgExpression *dim) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(PS_GET_LOCAL_SIZE_NAME);
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(dim));
  return fc;
}

SgFunctionCallExp *MPICUDARuntimeBuilder::BuildGetLocalOffset(
    SgExpression *dim) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(PS_GET_LOCAL_OFFSET_NAME);
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(dim));
  return fc;
}

SgFunctionCallExp *MPICUDARuntimeBuilder::BuildDomainShrink(
    SgExpression *dom, SgExpression *width) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(PS_DOMAIN_SHRINK_NAME);
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(
          fs, sb::buildExprListExp(dom, width));
  return fc;
}

SgExpression *MPICUDARuntimeBuilder::BuildStreamBoundaryKernel(int idx) {
  SgVarRefExp *inner_stream = Var("stream_boundary_kernel");
  return sb::buildPntrArrRefExp(inner_stream, sb::buildIntVal(idx));
}

SgExprListExp *MPICUDARuntimeBuilder::BuildKernelCallArgList(
    StencilMap *stencil,
    SgExpressionPtrList &index_args,
    SgFunctionParameterList *run_kernel_params) {
  SgExprListExp *args = cuda_rt_builder_->BuildKernelCallArgList(
      stencil, index_args, run_kernel_params);
  // remove the last offset args
  int dim = stencil->getNumDim();
  int num_offset_args = std::min(2, dim);
  if (num_offset_args > 0) {
    SgExprListExp *new_args = sb::buildExprListExp();
    int num_current_args = args->get_expressions().size();
    for (int i = 0; i < num_current_args - num_offset_args; ++i) {
      si::appendExpression(
          new_args, si::copyExpression(args->get_expressions()[i]));
    }
    si::deleteAST(args);
    args = new_args;
  }
  return args;
}

SgIfStmt *MPICUDARuntimeBuilder::BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgInitializedName *dom_arg, SgStatement *true_stmt) {
  return cuda_rt_builder_->BuildDomainInclusionCheck(
      indices, dom_arg, true_stmt);
}

SgFunctionParameterList *MPICUDARuntimeBuilder::BuildRunKernelFuncParameterList(
    StencilMap *stencil) {
  SgFunctionParameterList *params =
      cuda_rt_builder_->BuildRunKernelFuncParameterList(stencil);
  // add offset for process
  int num_offset_args = std::min(2, stencil->getNumDim());
  for (int i = 1; i <= num_offset_args; ++i) {
    si::appendArg(
        params,
        sb::buildInitializedName(
            PS_RUN_KERNEL_PARAM_OFFSET_NAME + toString(i), sb::buildIntType()));
  }
  return params;
}

SgFunctionDeclaration *MPICUDARuntimeBuilder::BuildRunKernelFunc(
    StencilMap *stencil) {
  return cuda_rt_builder_->BuildRunKernelFunc(stencil);
}

void MPICUDARuntimeBuilder::BuildRunKernelFuncBody(
    StencilMap *stencil, SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices,
    SgBasicBlock *body) {
  cuda_rt_builder_->BuildRunKernelFuncBody(
      stencil, param, indices, body);
}

SgScopeStatement *MPICUDARuntimeBuilder::BuildKernelCallPreamble1D(
    StencilMap *stencil,
    SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  return cuda_rt_builder_->BuildKernelCallPreamble1D(
      stencil, param, indices, call_site);
}

SgScopeStatement *MPICUDARuntimeBuilder::BuildKernelCallPreamble2D(
    StencilMap *stencil,
    SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  return cuda_rt_builder_->BuildKernelCallPreamble2D(
      stencil, param, indices, call_site);
}

SgScopeStatement *MPICUDARuntimeBuilder::BuildKernelCallPreamble3D(
    StencilMap *stencil,
    SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  return cuda_rt_builder_->BuildKernelCallPreamble2D(
      stencil, param, indices, call_site);
}

SgInitializedName *MPICUDARuntimeBuilder::GetDomArgParamInRunKernelFunc(
    SgFunctionParameterList *pl, int dim) {
  // return pl->get_args()[dim-1];
  return pl->get_args()[0];
}

SgScopeStatement *MPICUDARuntimeBuilder::BuildKernelCallPreamble(
    StencilMap *stencil,
    SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices,
    SgScopeStatement *call_site) {
  return cuda_rt_builder_->BuildKernelCallPreamble(
      stencil, param, indices, call_site);
}

void MPICUDARuntimeBuilder::BuildKernelIndices(
    StencilMap *stencil,
    SgBasicBlock *call_site,
    vector<SgVariableDeclaration*> &indices) {
  cuda_rt_builder_->BuildKernelIndices(stencil, call_site, indices);
  int dim = stencil->getNumDim();
  int num_offset_par = (dim >= 2) ? 2 : 1;
  for (int i = 1; i <= num_offset_par; ++i) {
    SgVarRefExp *offset_var =
        Var(PS_RUN_KERNEL_PARAM_OFFSET_NAME + toString(i));
    SgAssignInitializer *asn =
        isSgAssignInitializer(
            indices[i-1]->get_variables()[0]->get_initializer());
    PSAssert(asn);
    SgExpression *new_init_exp =
        Add(si::copyExpression(asn->get_operand()), offset_var);
    si::replaceExpression(asn->get_operand(), new_init_exp);
  }
}

SgType *MPICUDARuntimeBuilder::BuildOnDeviceGridType(GridType *gt) {
  return cuda_rt_builder_->BuildOnDeviceGridType(gt);
}

SgVariableDeclaration *MPICUDARuntimeBuilder::BuildGridDimDeclaration(
    const SgName &name, int dim,
    SgExpression *dom_dim_x, SgExpression *dom_dim_y,
    SgExpression *block_dim_x, SgExpression *block_dim_y,
    SgScopeStatement *scope) {
  return cuda_rt_builder_->BuildGridDimDeclaration(
      name, dim, dom_dim_x, dom_dim_y, block_dim_x, block_dim_y, scope);
}

SgExprListExp *MPICUDARuntimeBuilder::BuildCUDAKernelArgList(
    StencilMap *sm, SgVariableSymbol *sv,
    bool overlap_enabled, int overlap_width) {
  return BuildCUDAKernelArgList(sm, sv, overlap_enabled, overlap_width, false);
}

SgExprListExp *MPICUDARuntimeBuilder::BuildCUDABoundaryKernelArgList(
    StencilMap *sm, SgVariableSymbol *sv,
    bool overlap_enabled, int overlap_width) {
  return BuildCUDAKernelArgList(sm, sv, overlap_enabled, overlap_width, true);
}

SgExprListExp *MPICUDARuntimeBuilder::BuildCUDAKernelArgList(
    StencilMap *sm, SgVariableSymbol *sv,
    bool overlap_enabled, int overlap_width,
    bool is_boundary) {
  SgExprListExp *args = sb::buildExprListExp();
  SgClassDefinition *stencil_def = sm->GetStencilTypeDefinition();
  PSAssert(stencil_def);
  if (is_boundary && !flag_multistream_boundary_)
    si::appendExpression(args, Int(overlap_width));
  // Enumerate members of parameter struct
  const SgDeclarationStatementPtrList &members = stencil_def->get_members();
  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    SgExpression *arg =
        Arrow(Var(sv), Var(member_decl));
    SgType *member_type = si::getFirstVarType(member_decl);
    GridType *gt = ru::GetASTAttribute<GridType>(member_type);
    if (gt) {
      arg = sb::buildPointerDerefExp(
          sb::buildCastExp(
              BuildGridGetDev(arg, gt),
              sb::buildPointerType(BuildOnDeviceGridType(gt))));
      // skip the grid index
      ++member;
    }
    if (!is_boundary) {
      if (overlap_enabled && Domain::isDomainType(member_type)) {
        si::appendExpression(
            args, BuildDomainShrink(sb::buildAddressOfOp(arg),
                                    Int(overlap_width)));
      } else {
        si::appendExpression(args, arg);
      }
    } else {
      si::appendExpression(args, arg);
    }
  }

  // Append the local offset
  for (int i = 1; i <= std::min(2, sm->getNumDim()); ++i) {
    if (!is_boundary || (is_boundary && !flag_multistream_boundary_)) {
      si::appendExpression(args, BuildGetLocalOffset(Int(i)));
    }
  }
  return args;
}

SgExpression *MPICUDARuntimeBuilder::BuildBlockDimX(int nd) {
  return cuda_rt_builder_->BuildBlockDimX(nd);
}

SgExpression *MPICUDARuntimeBuilder::BuildBlockDimY(int nd) {
  return cuda_rt_builder_->BuildBlockDimY(nd);
}

SgExpression *MPICUDARuntimeBuilder::BuildBlockDimZ(int nd) {
  return cuda_rt_builder_->BuildBlockDimZ(nd);
}

void MPICUDARuntimeBuilder::BuildRunFuncBody(
    Run *run, SgFunctionDeclaration *run_func) {
  SgBasicBlock *block = run_func->get_definition()->get_body();
  si::attachComment(block, "Generated by " + string(__FUNCTION__));
  // build main loop
  SgBasicBlock *loopBody = BuildRunFuncLoopBody(run, run_func);
  // Cache configuration is done at the runtime library
  // cache_config_done_.clear();

  SgVariableDeclaration *lv
      = sb::buildVariableDeclaration("i", sb::buildIntType(), NULL, block);
  si::appendStatement(lv, block);
  SgStatement *loopTest =
      sb::buildExprStatement(
          sb::buildLessThanOp(Var(lv),
                              Var("iter", block)));
  SgForStatement *loop =
      sb::buildForStatement(sb::buildAssignStatement(Var(lv),
                                                     Int(0)),
                            loopTest,
                            sb::buildPlusPlusOp(Var(lv)),
                            loopBody);

  TraceStencilRun(run, loop, block);
  // cudaThreadSynchronize after each loop
  si::insertStatementAfter(
      loop,
      sb::buildExprStatement(cu::BuildCUDADeviceSynchronize()));
  return;
}

SgBasicBlock *MPICUDARuntimeBuilder::BuildRunFuncLoopBody(
    Run *run, SgFunctionDeclaration *run_func) {
  SgBasicBlock *loop_body = sb::buildBasicBlock();
  ENUMERATE(i, it, run->stencils().begin(), run->stencils().end()) {
    ProcessStencilMap(it->second, i, run, run_func, loop_body);
  }
  return loop_body;
}

namespace {
string GetBoundarySuffix(int dim, bool fw) {
  return string(PS_STENCIL_MAP_BOUNDARY_SUFFIX_NAME) + "_" +
      toString(dim+1) + "_" +
      (fw ? PS_STENCIL_MAP_BOUNDARY_FW_SUFFIX_NAME :
       PS_STENCIL_MAP_BOUNDARY_BW_SUFFIX_NAME);
}

string GetBoundarySuffix() {
  return string(PS_STENCIL_MAP_BOUNDARY_SUFFIX_NAME);
}

}  // namespace


// REFACTORING
void MPICUDARuntimeBuilder::ProcessStencilMapWithOverlapping(
    StencilMap *smap,
    SgScopeStatement *loop_body,
    SgVariableDeclaration *grid_dim,
    SgVariableDeclaration *block_dim,
    SgExprListExp *args, SgExprListExp *args_boundary,
    const SgStatementPtrList &load_statements,
    SgCudaKernelExecConfig *cuda_config,
    int overlap_width) {

  SgVarRefExp *inner_stream = Var("stream_inner");
  PSAssert(inner_stream);
  SgCudaKernelExecConfig *cuda_config_inner =
      cu::BuildCudaKernelExecConfig(Var(grid_dim),
                                    Var(block_dim),
                                    NULL, inner_stream);

  SgFunctionSymbol *fs_inner =
      ru::getFunctionSymbol(smap->run_inner());
  si::appendStatement(
      sb::buildExprStatement(
          cu::BuildCudaKernelCallExp(sb::buildFunctionRefExp(fs_inner),
                                     args, cuda_config_inner)),
      loop_body);
  // perform boundary exchange concurrently
  FOREACH (sit, load_statements.begin(), load_statements.end()) {
    si::appendStatement(*sit, loop_body);
  }
  LOG_INFO() << "generating call to boundary kernel\n";
  if (overlap_width && !flag_multistream_boundary_) {
    LOG_INFO() << "single-stream version\n";
    SgFunctionSymbol *fs_boundary =
        si::lookupFunctionSymbolInParentScopes(
            smap->GetRunName() + GetBoundarySuffix());
    si::appendStatement(
        sb::buildExprStatement(
            cu::BuildCudaKernelCallExp(
                sb::buildFunctionRefExp(fs_boundary),
                args_boundary, cuda_config)),
        loop_body);
  } else if (overlap_width) {
    LOG_INFO() << "multi-stream version\n";
    // ru::AppendExprStatement(
    //     loop_body,
    //     BuildCUDAStreamSynchronize(Var("stream_boundary_copy")));
    // 6 streams for
    int stream_index = 0;
    int num_x_streams = 5;
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < num_x_streams; ++i) {
        SgExprListExp *args_boundary_strm =
            isSgExprListExp(si::copyExpression(args_boundary));
        SgExpressionPtrList &expressions =
            args_boundary_strm->get_expressions();
        SgExpression *dom =
            si::copyExpression(expressions.front());
        si::deleteAST(expressions.front());
        expressions.erase(expressions.begin());
        SgExpression *bd =
            BuildDomainGetBoundary(sb::buildAddressOfOp(dom),
                                   0, j, Int(overlap_width), num_x_streams, i);
        ru::PrependExpression(args_boundary_strm, bd);
        int dimz = 512 / (overlap_width * 128);
        SgCudaKernelExecConfig *boundary_config =
            cu::BuildCudaKernelExecConfig(
                Int(1), cu::BuildCUDADim3(overlap_width, 128, dimz),
                NULL, BuildStreamBoundaryKernel(stream_index));
        ++stream_index;
        SgFunctionSymbol *fs_boundary
            = si::lookupFunctionSymbolInParentScopes(
                smap->GetRunName() + GetBoundarySuffix(0, j));
        ru::AppendExprStatement(
            loop_body,
            cu::BuildCudaKernelCallExp(
                sb::buildFunctionRefExp(fs_boundary),
                args_boundary_strm, boundary_config));
      }
    }
    for (int j = 1; j < 3; ++j) {
      for (int i = 0; i < 2; ++i) {
        SgExprListExp *args_boundary_strm =
            isSgExprListExp(si::copyExpression(args_boundary));
        SgExpressionPtrList &expressions =
            args_boundary_strm->get_expressions();
        SgExpression *dom =
            si::copyExpression(expressions.front());
        si::deleteAST(expressions.front());
        expressions.erase(expressions.begin());
        SgExpression *bd =
            BuildDomainGetBoundary(
                sb::buildAddressOfOp(dom), j, i, Int(overlap_width),
                1, 0);
        ru::PrependExpression(args_boundary_strm, bd);
        SgCudaKernelExecConfig *boundary_config;
        if (j == 1) {
          int dimz = 512 / (overlap_width * 128);
          boundary_config = cu::BuildCudaKernelExecConfig(
              Int(1), cu::BuildCUDADim3(128, overlap_width, dimz),
              NULL, BuildStreamBoundaryKernel(stream_index));
        } else {
          boundary_config = cu::BuildCudaKernelExecConfig(
              Int(1), cu::BuildCUDADim3(128, 4),
              NULL, BuildStreamBoundaryKernel(stream_index));
        }
        ++stream_index;
        SgFunctionSymbol *fs_boundary
            = si::lookupFunctionSymbolInParentScopes(
                smap->GetRunName() + GetBoundarySuffix(j, i));
        ru::AppendExprStatement(
            loop_body,
            cu::BuildCudaKernelCallExp(
                sb::buildFunctionRefExp(fs_boundary),
                args_boundary_strm, boundary_config));
      }
    }
    si::appendStatement(
        sb::buildExprStatement(cu::BuildCUDADeviceSynchronize()),
        loop_body);
  }
}

SgVariableDeclaration *MPICUDARuntimeBuilder::BuildStencilDecl(
    StencilMap *smap, int stencil_map_index,
    SgFunctionDeclaration *run_func) {
  SgBasicBlock *run_func_body = run_func->get_definition()->get_body();
  string stencil_name = PS_STENCIL_MAP_STENCIL_PARAM_NAME + toString(
      stencil_map_index);
  SgType *stencil_ptr_type = sb::buildPointerType(smap->stencil_type());
  SgAssignInitializer *init =
      sb::buildAssignInitializer(
          sb::buildCastExp(
              ArrayRef(Var("stencils", run_func_body), Int(stencil_map_index)),
              stencil_ptr_type), stencil_ptr_type);
  SgVariableDeclaration *sdecl
      = sb::buildVariableDeclaration(stencil_name, stencil_ptr_type,
                                     init, run_func_body);
  si::appendStatement(sdecl, run_func_body);
  return sdecl;
}

void MPICUDARuntimeBuilder::ProcessStencilMap(
    StencilMap *smap,  int stencil_map_index, Run *run,
    SgFunctionDeclaration *run_func, SgScopeStatement *loop_body) {
  int nd = smap->getNumDim();
  // FunctionDef does not work for the below variable
  // declarations. Its body needs to be used instead.
  // SgFunctionDef *run_func_def = run_func->get_definition();
  SgBasicBlock *run_func_body = run_func->get_definition()->get_body();
  SgVariableDeclaration *sdecl
      = BuildStencilDecl(smap, stencil_map_index, run_func);

  SgInitializedNamePtrList remote_grids;
  SgStatementPtrList load_statements;
  bool overlap_eligible;
  int overlap_width;
  MPIRuntimeBuilder::BuildLoadRemoteGridRegion(smap, sdecl, run,
                                               remote_grids, load_statements,
                                               overlap_eligible, overlap_width);
  bool overlap_enabled = IsOverlappingEnabled() && overlap_eligible;

  LOG_INFO() << (overlap_enabled ? "Generating overlapping code\n" :
                 "Generating non-overlapping code\n");

  // Build a CUDA block variable declaration
  SgVariableDeclaration *block_dim =
      cu::BuildDim3Declaration(
          "block_dim" + toString(stencil_map_index),
          BuildBlockDimX(nd), BuildBlockDimY(nd),
          BuildBlockDimZ(nd), run_func_body);
  si::appendStatement(block_dim, run_func_body);

  // Build a CUDA grid variable declaration
  SgVariableDeclaration *grid_dim = BuildGridDimDeclaration(
      "grid_dim" + toString(stencil_map_index), nd,
      BuildGetLocalSize(Int(1)), BuildGetLocalSize(Int(2)),
      BuildBlockDimX(nd), BuildBlockDimY(nd), run_func_body);
  si::appendStatement(grid_dim, run_func_body);

  // Build kernel call argument lists for normal kernel and interior kernel
  SgExprListExp *args = BuildCUDAKernelArgList(
      smap, si::getFirstVarSym(sdecl), overlap_enabled, overlap_width);
  SgExprListExp *args_boundary = BuildCUDABoundaryKernelArgList(
      smap, si::getFirstVarSym(sdecl), overlap_enabled, overlap_width);

  // Generate Kernel invocation code
  SgCudaKernelExecConfig *cuda_config =
      cu::BuildCudaKernelExecConfig(Var(grid_dim), Var(block_dim), NULL, NULL);

  // Append calls to kernels
  if (overlap_enabled) {
    ProcessStencilMapWithOverlapping(
        smap, loop_body, grid_dim, block_dim, args, args_boundary,
        load_statements, cuda_config, overlap_width);
  } else {
    // perform boundary exchange before kernel invocation synchronously
    FOREACH (sit, load_statements.begin(), load_statements.end()) {
      si::appendStatement(*sit, loop_body);
    }
    SgFunctionSymbol *fs = ru::getFunctionSymbol(smap->run());
    PSAssert(fs);
    SgCudaKernelCallExp *c =
        cu::BuildCudaKernelCallExp(sb::buildFunctionRefExp(fs),
                                   args, cuda_config);
    si::appendStatement(sb::buildExprStatement(c), loop_body);
  }

  // Appends calls to deactivate remote grids
  SgStatementPtrList stmt_list;
  BuildDeactivateRemoteGrids(smap, sdecl, remote_grids, stmt_list);
  BOOST_FOREACH(SgStatement *stmt, stmt_list) {
    si::appendStatement(stmt, loop_body);
  }

  BuildFixGridAddresses(smap, sdecl, run_func_body);
}

SgClassDeclaration *MPICUDARuntimeBuilder::BuildGridDevTypeForUserType(
    SgClassDeclaration *grid_decl,
    const GridType *gt) {
  
  // First, create the declaration as the CUDA translator, then update
  // it for MPI-CUDA. Update incldues insertion of local dimension
  // informatin. 

  // Step 1. Create the declaration for CUDA
  SgClassDeclaration *decl = cuda_rt_builder_->BuildGridDevTypeForUserType(
      grid_decl, gt);
  SgClassDefinition *type_def = decl->get_definition();

  // Step 2. Update the declaration for MPI-CUDA
  
  // Step 2.1 Insert local_size after dim field
  SgVariableDeclaration *dim_decl = isSgVariableDeclaration(
      si::getFirstStatement(type_def));
  PSAssert(dim_decl);
  SgType *dim_type = si::getFirstVarSym(dim_decl)->get_type();
  SgVariableDeclaration *local_size_decl =
      sb::buildVariableDeclaration(PS_GRID_TYPE_LOCAL_SIZE_NAME,
                                   dim_type, NULL, type_def);
  si::insertStatementAfter(dim_decl, local_size_decl);

  // Step 2.2 Insert local_offset after local_size field
  SgVariableDeclaration *local_offset_decl =
      sb::buildVariableDeclaration(PS_GRID_TYPE_LOCAL_OFFSET_NAME,
                                   dim_type, NULL, type_def);
  si::insertStatementAfter(local_size_decl, local_offset_decl);
  
  return decl;
}

void BuildPackingBuffer(SgBasicBlock *body,
                        SgClassDefinition *user_type_def,
                        SgVariableDeclaration *buf_decl,
                        SgInitializedName *pack_buf,
                        SgInitializedName *num_elms,
                        bool is_copyout) {
  const SgDeclarationStatementPtrList &members =
      user_type_def->get_members();

  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  ENUMERATE (i, member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl =
        isSgVariableDeclaration(*member);
    SgType *member_type = ru::GetType(member_decl);
    SgExpression *rhs = NULL;
    if (i == 0) {
      // For copyin, packing buffer is creatd here. For copyout, there
      // is already packing buffer filled with packed data. We just
      // need to transpose the data
      if (is_copyout) {
        rhs = Var(pack_buf);
      } else {
        SgExpression *size_exp =
            Mul(sb::buildSizeOfOp(user_type_def->get_declaration()->get_type()),
                Var(num_elms));
        rhs = sb::buildFunctionCallExp(
            "malloc", ru::VoidPointerType(),
            sb::buildExprListExp(size_exp));
      }
    } else {
      SgExpression *size_exp =
          Mul(sb::buildSizeOfOp(si::getArrayElementType(member_type)),
              Var(num_elms));
      if (isSgArrayType(member_type)) {
        size_exp = Mul(size_exp, Int(
            si::getArrayElementCount(isSgArrayType(member_type))));
      }
      rhs = ru::BuildOffsetVoidPointer(ArrayRef(Var(buf_decl), Int(i-1)),
                                       size_exp);
    }
    si::appendStatement(
        sb::buildAssignStatement(ArrayRef(Var(buf_decl), Int(i)), rhs),
        body);
  }
}


SgFunctionDeclaration *MPICUDARuntimeBuilder::BuildGridCopyFuncForUserType(
    const GridType *gt, bool is_copyout) {

  SgClassType *dev_type = static_cast<SgClassType*>(gt->aux_type());
  string func_name = dev_type->get_name();
  if (is_copyout) func_name += "Copyout"; else func_name += "Copyin";
  int num_point_elms = gt->point_def()->get_members().size();
  string host_name = is_copyout ? "dst" : "src";
  
  SgInitializedNamePtrList params;
  SgFunctionParameterList *pl =
      BuildGridCopyFuncSigForUserType(is_copyout, params);
  SgInitializedName *user_ptr = params[0];
  SgInitializedName *pack_buf = is_copyout ? params[1] : NULL;
  SgInitializedName *num_elms = params.back();

  SgFunctionDeclaration *fdecl = sb::buildDefiningFunctionDeclaration(
      func_name,
      is_copyout? isSgType(sb::buildVoidType()) : isSgType(ru::VoidPointerType()),
      pl);
  si::setStatic(fdecl);    
  
  // Function body
  SgBasicBlock *body = fdecl->get_definition()->get_body();
  
  //Type *dstp = (Type *)dst;
  SgType *hostp_type =
      sb::buildPointerType(
          (is_copyout) ? gt->point_type() :
          sb::buildConstType(gt->point_type()));
  SgVariableDeclaration *hostp_decl =
      sb::buildVariableDeclaration(
          host_name + "p", hostp_type,
          sb::buildAssignInitializer(
              sb::buildCastExp(Var(user_ptr), hostp_type)));
  si::appendStatement(hostp_decl, body);
  // void *tbuf[3];
  SgVariableDeclaration *tbuf_decl =
      sb::buildVariableDeclaration(
          "tbuf",
          sb::buildArrayType(
              is_copyout ? ru::ConstVoidPointerType() :
              ru::VoidPointerType(),
              Int(num_point_elms)));
  si::appendStatement(tbuf_decl, body);

  // cudaMallocHost((void**)&tbuf[0], sizeof(type) * num_elms);
  BuildPackingBuffer(body, gt->point_def(), tbuf_decl,
                     pack_buf, num_elms, is_copyout);
  
  // for (size_t i = 0; i < num_elms; ++i) {
  SgVariableDeclaration *init =
      sb::buildVariableDeclaration(
          "i", 
          si::lookupNamedTypeInParentScopes("size_t"),
          sb::buildAssignInitializer(Int(0)));
  SgExpression *cond =
      sb::buildLessThanOp(
          Var(init), Var(num_elms));
  SgExpression *incr = sb::buildPlusPlusOp(Var(init));
  SgBasicBlock *loop_body = sb::buildBasicBlock();
  SgForStatement *trans_loop =
      sb::buildForStatement(init, sb::buildExprStatement(cond),
                            incr, loop_body);
  si::appendStatement(trans_loop, body);

  cuda_rt_builder_->BuildUserTypeTranspose(
      loop_body, gt->point_def(), tbuf_decl, hostp_decl,
      is_copyout, init->get_variables()[0], num_elms);

  // Return packing buffer if copyin
  if (!is_copyout) {
    si::appendStatement(
        sb::buildReturnStmt(ArrayRef(Var(tbuf_decl), Int(0))),
        body);
  }
  
  return fdecl;
}

SgFunctionParameterList *MPICUDARuntimeBuilder::BuildGridCopyFuncSigForUserType(
    bool is_copyout, SgInitializedNamePtrList &params) {
  // Build a parameter list
  SgFunctionParameterList *pl = sb::buildFunctionParameterList();
  // const void *src or void *dst
  SgType *host_param_type = 
      sb::buildPointerType(
          is_copyout ? (SgType*)sb::buildVoidType() :
          (SgType*)sb::buildConstType(sb::buildVoidType()));
  string host_name = is_copyout ? "dst" : "src";  
  SgInitializedName *host_param =
      sb::buildInitializedName(host_name, host_param_type);
  si::appendArg(pl, host_param);
  params.push_back(host_param);
  if (is_copyout) {
    SgType *param_type = sb::buildPointerType(
        sb::buildConstType(sb::buildVoidType()));
    string param_name = "pack";
    SgInitializedName *param =
        sb::buildInitializedName(param_name, param_type);
    si::appendArg(pl, param);
    params.push_back(param);
  }
  // size_t num_elms
  SgInitializedName *num_elms =
      sb::buildInitializedName("num_elms", ru::SizeType());
  si::appendArg(pl, num_elms);
  params.push_back(num_elms);
  return pl;
}

SgExpression *MPICUDARuntimeBuilder::BuildGridEmit(
    SgExpression *grid_exp,
    GridEmitAttribute *attr,
    const SgExpressionPtrList *offset_exprs,
    SgExpression *emit_val,
    SgScopeStatement *scope) {
  return cuda_rt_builder_->BuildGridEmit(grid_exp, attr, offset_exprs,
                                         emit_val, scope);
}

void MPICUDARuntimeBuilder::BuildLoadRemoteGridRegion(
    SgInitializedName &grid_param,
    StencilMap &smap,
    SgVariableDeclaration &stencil_decl,
    SgInitializedNamePtrList &remote_grids,
    SgStatementPtrList &statements,
    bool &overlap_eligible,
    int &overlap_width,
    vector<SgIntVal*> &overlap_flags) {
  GridVarAttribute *gva =
      ru::GetASTAttribute<GridVarAttribute>(&grid_param);
  // If no member-specific accesses are detected, handle this case as
  // in the MPI case
  if (gva->gt()->IsPrimitivePointType() ||
      gva->member_sr().size() == 0) {
    MPIRuntimeBuilder::BuildLoadRemoteGridRegion(
        grid_param, smap, stencil_decl, remote_grids,
        statements, overlap_eligible, overlap_width, overlap_flags);
    return;
  }
  LOG_DEBUG() <<  "grid param: " << grid_param.unparseToString() << "\n";
  MemberStencilRangeMap &msr = gva->member_sr();
  BOOST_FOREACH (MemberStencilRangeMap::value_type kv, msr) {
    StencilRange &sr = kv.second;
    LOG_DEBUG() << "Argument stencil range: " << sr << "\n";
    const string &member = kv.first.first;
    int member_index = gva->gt()->GetMemberIndex(member);
    MPIRuntimeBuilder::BuildLoadRemoteGridRegion(
        grid_param, member_index, sr, smap, stencil_decl,
        remote_grids, statements, overlap_eligible,
        overlap_width, overlap_flags);
  }
  
}

}  // namespace translator
}  // namespace physis
