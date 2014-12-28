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
                        BuilderInterface *delegator=NULL):
      ReferenceRuntimeBuilder(global_scope, delegator),
      MPIRuntimeBuilder(global_scope),
      cuda_rt_builder_(new CUDARuntimeBuilder(global_scope, this))
  {}
  
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

  // CUDABuilderInterface functions

  // TODO
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
  
 protected:
  CUDARuntimeBuilder *cuda_rt_builder_;

};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_ */

