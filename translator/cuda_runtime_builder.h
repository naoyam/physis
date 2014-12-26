// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_CUDA_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_CUDA_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/reference_runtime_builder.h"
#include "translator/map.h"

namespace physis {
namespace translator {

class CUDARuntimeBuilder: public ReferenceRuntimeBuilder {
 public:
  CUDARuntimeBuilder(SgScopeStatement *global_scope):
      ReferenceRuntimeBuilder(global_scope) {}
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

  //! Build code for grid emit.
  /*!
    \param grid_exp Grid expression    
    \param attr GridEmit attribute
    \param offset_exprs offset expressions
    \param emit_val Value to emit
    \return Expression implementing the emit.
   */
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
  
 protected:
  virtual SgFunctionDeclaration *BuildGridCopyFuncForUserType(
      const GridType *gt, bool is_copyout);
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
