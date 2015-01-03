// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_CUDA_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_CUDA_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/reference_runtime_builder.h"
#include "translator/cuda_builder_interface.h"
#include "translator/map.h"

namespace physis {
namespace translator {

class CUDARuntimeBuilder: virtual public ReferenceRuntimeBuilder,
                          virtual public CUDABuilderInterface {
 public:
  CUDARuntimeBuilder(SgScopeStatement *global_scope,
                     const Configuration &config,
                     BuilderInterface *delegator=NULL);
  virtual ~CUDARuntimeBuilder() {}
  virtual SgExpression *BuildGridRefInRunKernel(
      SgInitializedName *gv,
      SgFunctionDeclaration *run_kernel);
  virtual SgExpression *BuildGridOffset(
      SgExpression *gv,
      int num_dim,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic);

  virtual SgExpression *BuildGridArrayMemberOffset(
    SgExpression *gvref,
    const GridType *gt,
    const string &member_name,
    const SgExpressionVector &array_indices);

  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,                  
      GridType *gt,      
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,      
      bool is_kernel,
      bool is_periodic);
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,            
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name);
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,            
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name,
      const SgExpressionVector &array_indices);

  virtual SgExpression *BuildGridEmit(
      SgExpression *grid_exp,      
      GridEmitAttribute *attr,
      const SgExpressionPtrList *offset_exprs,
      SgExpression *emit_val,
      SgScopeStatement *scope=NULL);
  
  virtual SgClassDeclaration *BuildGridDevTypeForUserType(
      SgClassDeclaration *grid_decl,
      const GridType *gt);
  virtual SgFunctionDeclaration *BuildGridNewFuncForUserType(
      const GridType *gt);
  virtual SgFunctionDeclaration *BuildGridFreeFuncForUserType(
      const GridType *gt);
  virtual SgFunctionDeclaration *BuildGridCopyinFuncForUserType(
      const GridType *gt);
  virtual SgFunctionDeclaration *BuildGridCopyoutFuncForUserType(
      const GridType *gt);
  virtual SgFunctionDeclaration *BuildGridGetFuncForUserType(
      const GridType *gt);
  virtual SgFunctionDeclaration *BuildGridEmitFuncForUserType(
      const GridType *gt);

  virtual SgExprListExp *BuildKernelCallArgList(
      StencilMap *stencil,
      SgExpressionPtrList &index_args,
      SgFunctionParameterList *params);

  virtual void BuildKernelIndices(
      StencilMap *stencil,
      SgBasicBlock *call_site,
      vector<SgVariableDeclaration*> &indices);


  virtual SgFunctionDeclaration *BuildRunKernelFunc(StencilMap *s);
  virtual SgFunctionParameterList *BuildRunKernelFuncParameterList(
      StencilMap *s);
  virtual SgBasicBlock *BuildRunKernelFuncBody(
      StencilMap *stencil, SgFunctionParameterList *param,
      vector<SgVariableDeclaration*> &indices);

  //! Generates an IF block to exclude indices outside a domain.
  /*!
    
    \param indices The indices to check.
    \param dom_arg Name of the domain parameter.
    \param true_stmt Statement to execute if outside the domain
    \return The IF block.
   */
  virtual SgIfStmt *BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgInitializedName *dom_arg, SgStatement *true_stmt);

  virtual void BuildRunFuncBody(
      Run *run, SgFunctionDeclaration *run_func);

  virtual SgBasicBlock *BuildRunFuncLoopBody(
      Run *run, SgFunctionDeclaration *run_func);
  
  virtual SgType *BuildOnDeviceGridType(GridType *gt);

  virtual SgExpression *BuildGridGetDev(SgExpression *grid_var,
                                        GridType *gt);

  // RunKernel(Domain *dom, [original params]) -> dom
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

  virtual SgVariableDeclaration *BuildGridDimDeclaration(
      const SgName &name,
      int dim,
      SgExpression *dom_dim_x, SgExpression *dom_dim_y,      
      SgExpression *block_dim_x, SgExpression *block_dim_y,
      SgScopeStatement *scope = NULL);

  //! Generates an argument list for a CUDA kernel call.
  /*!
    TODO (Interface)
    
    \param sm The stencil map object.
    \param sv Stencil parameter symbol
    \return The argument list for the call to the stencil map.
   */
  virtual SgExprListExp *BuildCUDAKernelArgList(
      StencilMap *sm, SgVariableSymbol *sv);

  //! Generates an expression of the x dimension of thread blocks.
  virtual SgExpression *BuildBlockDimX(int nd);
  //! Generates an expression of the y dimension of thread blocks.
  virtual SgExpression *BuildBlockDimY(int nd);
  //! Generates an expression of the z dimension of thread blocks.
  virtual SgExpression *BuildBlockDimZ(int nd);

  /*!
    TODO (Interface decl)
  */
  const std::vector<SgExpression *> cuda_block_size_vals() const {
    return cuda_block_size_vals_;
  }
  SgType *&cuda_block_size_type() {
    return cuda_block_size_type_;
  }

  virtual void AddDynamicParameter(SgFunctionParameterList *parlist);
  virtual void AddDynamicArgument(SgExprListExp *args, SgExpression *a_exp);
  virtual void AddSyncAfterDlclose(SgScopeStatement *scope);
  
  
 protected:
  int block_dim_x_;
  int block_dim_y_;
  int block_dim_z_;
  /** hold all CUDA_BLOCK_SIZE values */
  std::vector<SgExpression *> cuda_block_size_vals_;
  /** hold __cuda_block_size_struct type */
  SgType *cuda_block_size_type_;

  virtual SgFunctionDeclaration *BuildGridCopyFuncForUserType(
      const GridType *gt, bool is_copyout);
  
  //CUDABuilderInterface *delegator_;
  CUDABuilderInterface *CUDABuilder() {
    CUDABuilderInterface *x =
        dynamic_cast<CUDABuilderInterface*>(delegator_);
    return x ? x : this;
  }
  CUDABuilderInterface *Builder() {
    CUDABuilderInterface *x =
        dynamic_cast<CUDABuilderInterface*>(delegator_);
    return x ? x : this;
  }
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
