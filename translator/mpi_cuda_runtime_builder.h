// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/cuda_builder_interface.h"
#include "translator/cuda_runtime_builder.h"

namespace physis {
namespace translator {

class MPICUDARuntimeBuilder: virtual public MPIRuntimeBuilder,
                             virtual public CUDABuilderInterface {
 public:
  MPICUDARuntimeBuilder(SgScopeStatement *global_scope,
                        const Configuration &config,
                        BuilderInterface *delegator=NULL):
      ReferenceRuntimeBuilder(global_scope, config, delegator),
      MPIRuntimeBuilder(global_scope, config),
      cuda_rt_builder_(new CUDARuntimeBuilder(global_scope, config, this)),
      flag_multistream_boundary_(false) {
    const pu::LuaValue *lv =
        config.Lookup(Configuration::MULTISTREAM_BOUNDARY);
    if (lv) {
      PSAssert(lv->get(flag_multistream_boundary_));
    }
  }
  
  virtual ~MPICUDARuntimeBuilder() {
    delete cuda_rt_builder_;
  }

  virtual SgExpression *BuildGridBaseAddr(
      SgExpression *gvref, SgType *point_type);

  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic) {
    return cuda_rt_builder_->BuildGridOffset(
        gvref, num_dim, offset_exprs, sil, is_kernel,
        is_periodic);
  }

  virtual SgExprListExp *BuildKernelCallArgList(
      StencilMap *stencil,
      SgExpressionPtrList &index_args,
      SgFunctionParameterList *params);
  

  virtual void BuildKernelIndices(
      StencilMap *stencil,
      SgBasicBlock *call_site,
      vector<SgVariableDeclaration*> &indices);

  virtual SgIfStmt *BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgInitializedName *dom_arg, SgStatement *true_stmt);
  
  virtual SgFunctionDeclaration *BuildRunKernelFunc(StencilMap *s);
  virtual SgBasicBlock *BuildRunKernelFuncBody(
      StencilMap *stencil, SgFunctionParameterList *param,
      vector<SgVariableDeclaration*> &indices);
  virtual SgFunctionParameterList *BuildRunKernelFuncParameterList(
      StencilMap *stencil);

  virtual void BuildRunFuncBody(
      Run *run, SgFunctionDeclaration *run_func);
  virtual SgBasicBlock *BuildRunFuncLoopBody(
      Run *run, SgFunctionDeclaration *run_func);

  // CUDABuilderInterface functions

  // TODO (user-defined type)
  virtual SgClassDeclaration *BuildGridDevTypeForUserType(
      SgClassDeclaration *grid_decl,
      const GridType *gt) {
    return NULL;
  }
  virtual SgFunctionDeclaration *BuildGridNewFuncForUserType(
      const GridType *gt) {
    return NULL;
  }
  virtual SgFunctionDeclaration *BuildGridFreeFuncForUserType(
      const GridType *gt) {
    return NULL;
  }
  virtual SgFunctionDeclaration *BuildGridCopyinFuncForUserType(
      const GridType *gt) {
    return NULL;
  }
  virtual SgFunctionDeclaration *BuildGridCopyoutFuncForUserType(
      const GridType *gt) {
    return NULL;
  }
  virtual SgFunctionDeclaration *BuildGridGetFuncForUserType(
      const GridType *gt) {
    return NULL;
  }
  virtual SgFunctionDeclaration *BuildGridEmitFuncForUserType(
      const GridType *gt) {
    return NULL;
  }
  
  // RunKernel(int offset1, int offset2, Domain *dom, [original params]) -> dom
  virtual SgInitializedName *GetDomArgParamInRunKernelFunc(
      SgFunctionParameterList *pl, int dim);

  virtual SgScopeStatement *BuildKernelCallPreamble(
      StencilMap *stencil,      
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site);

  //! Helper function for BuildKernelCallPreamble for 1D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble1D(
      StencilMap *stencil,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site);

  //! Helper function for BuildKernelCallPreamble for 2D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble2D(
      StencilMap *stencil,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site);

  //! Helper function for BuildKernelCallPreamble for 3D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble3D(
      StencilMap *stencil,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site);

  virtual SgType *BuildOnDeviceGridType(GridType *gt);
  
  virtual SgExpression *BuildGridGetDev(SgExpression *grid_var,
                                        GridType *gt);

  virtual SgVariableDeclaration *BuildGridDimDeclaration(
      const SgName &name,
      int dim,
      SgExpression *dom_dim_x, SgExpression *dom_dim_y,      
      SgExpression *block_dim_x, SgExpression *block_dim_y,
      SgScopeStatement *scope = NULL);

  /*!
    REFACTORING: Signature does not match with the CUDA runtime
    builder. overlap_enabled and overlap_width can be determined at
    the analysis time, so they can be attributes of StencilMap.
   */
  virtual SgExprListExp *BuildCUDAKernelArgList(
      StencilMap *sm, SgVariableSymbol *sv,
      bool overlap_enabled, int overlap_width);

  virtual SgExpression *BuildBlockDimX(int nd);
  virtual SgExpression *BuildBlockDimY(int nd);
  virtual SgExpression *BuildBlockDimZ(int nd);
  

  // Not derived functions

  //! Build a call to the GetLocalSize function in the MPI-CUDA runtime
  /*
    Not a derived function.
   */
  virtual SgFunctionCallExp *BuildGetLocalSize(SgExpression *dim);

  //! Build a call to the GetLocalOffset function in the MPI-CUDA runtime
  /*
    Not a derived function.
   */
  virtual SgFunctionCallExp *BuildGetLocalOffset(SgExpression *dim);

  //! Build a call to the DomainShrink function in the MPI-CUDA runtime
  /*
    Not a derived function.
   */
  virtual SgFunctionCallExp *BuildDomainShrink(SgExpression *dom,
                                               SgExpression *width);
  
  //! Build an object referencing to a boundary kernel
  /*

    Not a derived function.
    
    \param idx Boundary kernel index
   */
  virtual SgExpression *BuildStreamBoundaryKernel(int idx);

  virtual SgExprListExp *BuildCUDABoundaryKernelArgList(
      StencilMap *sm, SgVariableSymbol *sv,
      bool overlap_enabled, int overlap_width);
  
 protected:
  CUDARuntimeBuilder *cuda_rt_builder_;
  //! Optimization flag to enable the multi-stream boundary
  //! processing.
  // REFACTORING: duplication with MPICUDATranslator. Move the
  //! functionality to this class entirely.
  bool flag_multistream_boundary_;
  string boundary_suffix_;
  /*!
    Not a derived function.
   */
  virtual SgExprListExp *BuildCUDAKernelArgList(
      StencilMap *sm, SgVariableSymbol *sv,
      bool overlap_enabled, int overlap_width,
      bool is_boundary);

  //! Helper function for BuildRunFuncLoopBody
  /*! 
    Not a derived function.
   */
  virtual void ProcessStencilMap(StencilMap *smap, 
                                 int stencil_index, Run *run,
                                 SgFunctionDeclaration *run_func,
                                 SgScopeStatement *loop_body);
  
  //! Helper function for ProcessStencilMap
  /*! 
    Not a derived function.
   */
  virtual void ProcessStencilMapWithOverlapping(
      StencilMap *smap,
      SgScopeStatement *loop_body,
      SgVariableDeclaration *grid_dim,
      SgVariableDeclaration *block_dim,
      SgExprListExp *args, SgExprListExp *args_boundary,
      const SgStatementPtrList &load_statements,
      SgCudaKernelExecConfig *cuda_config,    
      int overlap_width);
  
  //! Helper function for ProcessStencilMap
  virtual SgVariableDeclaration *BuildStencilDecl(
      StencilMap *smap, int stencil_map_index,
      SgFunctionDeclaration *run_func);
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_ */

