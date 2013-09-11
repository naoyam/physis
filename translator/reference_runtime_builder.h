// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/map.h"
#include "translator/runtime_builder.h"

namespace physis {
namespace translator {

class ReferenceRuntimeBuilder: public RuntimeBuilder {
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
  
  
  //!
  /*!
   */
  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
      const SgExpressionPtrList *offset_exprs, bool is_kernel,
      bool is_periodic, const StencilIndexList *sil);

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
  
  
 protected:
  static const std::string  grid_type_name_;
  SgTypedefType *dom_type_;
  SgClassDeclaration *GetGridDecl();
  virtual SgExpression *BuildDomFieldRef(SgExpression *domain,
                                         string fname);
  
  
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_REFERENCE_RUNTIME_BUILDER_H_ */
