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
                     BuilderInterface *delegator=NULL):
      ReferenceRuntimeBuilder(global_scope, delegator) {}
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

  
  virtual SgType *BuildOnDeviceGridType(GridType *gt);

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
  
 protected:
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
