// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/map.h"
#include "translator/builder_interface.h"

namespace physis {
namespace translator {

class ReferenceRuntimeBuilder: virtual public BuilderInterface {
 public:
  ReferenceRuntimeBuilder(SgScopeStatement *global_scope);
  virtual ~ReferenceRuntimeBuilder() {}
  virtual SgFunctionCallExp *BuildGridGetID(SgExpression *grid_var);
  virtual SgBasicBlock *BuildGridSet(
      SgExpression *grid_var, int num_dims,
      const SgExpressionPtrList &indices, SgExpression *val);
  virtual SgFunctionCallExp *BuildGridDim(SgExpression *grid_ref,
                                          int dim);
  virtual SgExpression *BuildGridRefInRunKernel(
      SgInitializedName *gv,
      SgFunctionDeclaration *run_kernel);


  virtual SgExpression *BuildGridBaseAddr(
      SgExpression *gvref, SgType *point_type);

  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
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
    \param scope Scope where this expression is built
    \return Expression implementing the emit.
   */
  virtual SgExpression *BuildGridEmit(
      SgExpression *grid_exp,
      GridEmitAttribute *attr,
      const SgExpressionPtrList *offset_exprs,
      SgExpression *emit_val,
      SgScopeStatement *scope=NULL);
  
  

  /*
  virtual SgExpression *BuildGet(  
    SgInitializedName *gv,
    SgExprListExp *offset_exprs,
    SgScopeStatement *scope,
    TranslationContext *tx, bool is_kernel,
    bool is_periodic);
  */

  virtual SgExpression *BuildDomMinRef(
      SgExpression *domain, int dim);
  virtual SgExpression *BuildDomMaxRef(
      SgExpression *domain, int dim);
  //! Build a domain min expression
  /*!
    \param domain Domain expression
    \return Domain min expression
   */
  virtual SgExpression *BuildDomMinRef(
      SgExpression *domain);
  //! Build a domain max expression
  /*!
    \param domain Domain expression
    \return Domain max expression
   */
  virtual SgExpression *BuildDomMaxRef(
      SgExpression *domain);
  
  virtual string GetStencilDomName();
  
  virtual SgExpression *BuildStencilFieldRef(
      SgExpression *stencil_ref, std::string name);
  virtual SgExpression *BuildStencilFieldRef(
      SgExpression *stencil_ref, SgExpression *field);
  //! Build a domain min expression for a dimension from a stencil      
  virtual SgExpression *BuildStencilDomMinRef(
      SgExpression *stencil);
  virtual SgExpression *BuildStencilDomMinRef(
      SgExpression *stencil, int dim);
  //! Build a domain max expression for a dimension from a stencil    
  virtual SgExpression *BuildStencilDomMaxRef(
      SgExpression *stencil);
  virtual SgExpression *BuildStencilDomMaxRef(
      SgExpression *stencil, int dim);

  virtual SgExprListExp *BuildStencilOffsetMax(
      const StencilRange &sr);
  virtual SgExprListExp *BuildStencilOffsetMin(
      const StencilRange &sr);

  virtual SgExprListExp *BuildSizeExprList(const Grid *g);  
  
  virtual SgClassDeclaration *BuildStencilMapType(StencilMap *s);
  virtual SgFunctionDeclaration *BuildMap(StencilMap *stencil);

  virtual SgFunctionCallExp* BuildKernelCall(
      StencilMap *stencil, SgExpressionPtrList &index_args,
      SgFunctionParameterList *run_kernel_params);
  
  virtual SgExprListExp *BuildKernelCallArgList(
      StencilMap *stencil,
      SgExpressionPtrList &index_args,
      SgFunctionParameterList *params);

  virtual SgFunctionDeclaration *BuildRunKernelFunc(StencilMap *s);
  
  virtual SgBasicBlock *BuildRunKernelFuncBody(
      StencilMap *stencil, SgFunctionParameterList *param,
      vector<SgVariableDeclaration*> &indices);

  virtual SgVariableDeclaration *BuildLoopIndexVarDecl(
      int dim,
      SgExpression *init,
      SgScopeStatement *block);
  
 protected:
  static const std::string  grid_type_name_;
  SgScopeStatement *gs_;
  SgTypedefType *dom_type_;
  SgClassDeclaration *GetGridDecl();
  virtual SgExpression *BuildDomFieldRef(SgExpression *domain,
                                         string fname);
  
  virtual SgExprListExp *BuildStencilOffset(
      const StencilRange &sr, bool is_max);
  
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
