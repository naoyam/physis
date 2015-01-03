// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/mpi_cuda_runtime_builder.h"

namespace sb = SageBuilder;
namespace si = SageInterface;
namespace ru = physis::translator::rose_util;

namespace physis {
namespace translator {

SgExpression *MPICUDARuntimeBuilder::BuildGridBaseAddr(
    SgExpression *gvref, SgType *point_type) {
  return ReferenceRuntimeBuilder::BuildGridBaseAddr(gvref, point_type);
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
SgFunctionCallExp *MPICUDARuntimeBuilder::BuildGetLocalOffset(SgExpression *dim) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(PS_GET_LOCAL_OFFSET_NAME);
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(dim));
  return fc;
}

SgFunctionCallExp *MPICUDARuntimeBuilder::BuildDomainShrink(SgExpression *dom,
                                                            SgExpression *width) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes(PS_DOMAIN_SHRINK_NAME);
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(
          fs, sb::buildExprListExp(dom, width));
  return fc;
}

SgExpression *MPICUDARuntimeBuilder::BuildStreamBoundaryKernel(int idx) {
  SgVarRefExp *inner_stream = sb::buildVarRefExp("stream_boundary_kernel");
  return sb::buildPntrArrRefExp(inner_stream, sb::buildIntVal(idx));
}

SgExprListExp *MPICUDARuntimeBuilder::BuildKernelCallArgList(
    StencilMap *stencil,
    SgExpressionPtrList &index_args,
    SgFunctionParameterList *run_kernel_params) {
  SgExprListExp *args = cuda_rt_builder_->BuildKernelCallArgList(
      stencil, index_args, run_kernel_params);
  // remove the last offset args
  int dim = 3;
  int num_offset_args = dim - 1;
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
  for (int i = 1; i <= stencil->getNumDim()-1; ++i) {
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

SgBasicBlock *MPICUDARuntimeBuilder::BuildRunKernelFuncBody(
    StencilMap *stencil, SgFunctionParameterList *param,
    vector<SgVariableDeclaration*> &indices) {
  return cuda_rt_builder_->BuildRunKernelFuncBody(
      stencil, param, indices);
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
  //return pl->get_args()[dim-1];
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
  for (int i = 1; i < dim; ++i) {
    SgVarRefExp *offset_var =
        sb::buildVarRefExp(PS_RUN_KERNEL_PARAM_OFFSET_NAME + toString(i));
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
  for (int i = 1; i < sm->getNumDim(); ++i) {
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

} // namespace translator
} // namespace physis
