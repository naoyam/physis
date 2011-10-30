// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_REFERENCE_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_REFERENCE_TRANSLATOR_H_

#include "translator/translator.h"
#include "translator/translator_common.h"
#include "translator/reference_runtime_builder.h"

#define PHYSIS_REFERENCE_HEADER "physis_ref.h"

namespace physis {
namespace translator {

// NOTE: Some functions might be better suited to be declared in
// the root class
class ReferenceTranslator : public Translator {
 private:
  // If this flag is on, the Translator replace grid_dim[xyz] to the constant
  // value of grid size when it's feasible.
  bool flag_constant_grid_size_optimization_;

 public:
  ReferenceTranslator(const Configuration &config);
  virtual ~ReferenceTranslator();
  virtual void run();
  virtual void optimize();
  virtual void SetUp(SgProject *project, TranslationContext *context);
  virtual void Finish();  

  bool flag_constant_grid_size_optimization() const {
    return flag_constant_grid_size_optimization_;
  }
  void set_flag_constant_grid_size_optimization(bool flag) {
    flag_constant_grid_size_optimization_ = flag;
  }

 protected:

  virtual void translateKernelDeclaration(SgFunctionDeclaration *node);
  virtual void translateNew(SgFunctionCallExp *node, GridType *gt);
  virtual SgExprListExp *generateNewArg(GridType *gt, Grid *g,
                                        SgVariableDeclaration *dim_decl);
  virtual void appendNewArgExtra(SgExprListExp *args, Grid *g);
  virtual void translateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool isKernel);
  virtual void translateEmit(SgFunctionCallExp *node, SgInitializedName *gv);
  virtual void translateSet(SgFunctionCallExp *node, SgInitializedName *gv); 
  virtual SgExpression *buildOffset(SgInitializedName *gv,
                                    SgScopeStatement *scope,
                                    int numDim,
                                    SgExpressionPtrList &args);
  virtual void translateMap(SgFunctionCallExp *node, StencilMap *s);
  virtual SgFunctionDeclaration *GenerateMap(StencilMap *s);
  virtual SgFunctionDeclaration *BuildRunKernel(StencilMap *s);
  virtual SgFunctionDeclaration *BuildRunInteriorKernel(StencilMap *s) {
    return NULL;
  }
  virtual SgFunctionDeclarationPtrVector BuildRunBoundaryKernel(
      StencilMap *s) {
    std::vector<SgFunctionDeclaration*> v;
    return v;
  }
  //! A helper function for BuildRunKernel.
  /*!
    \param s The stencil map object.
    \param stencil_param 
    \return The body of the run function.
   */
  virtual SgBasicBlock *BuildRunKernelBody(
      StencilMap *s, SgInitializedName *stencil_param);
  virtual void appendGridSwap(StencilMap *mc, SgExpression *stencil,
                              SgScopeStatement *scope);
  virtual SgFunctionCallExp* BuildKernelCall(
      StencilMap *s, SgExpressionPtrList &indexArgs,
      SgScopeStatement *containingScope);
  virtual void defineMapSpecificTypesAndFunctions();
  virtual SgBasicBlock *BuildRunBody(Run *run);
  virtual SgFunctionDeclaration *GenerateRun(Run *run);
  virtual void translateRun(SgFunctionCallExp *node, Run *run);

  virtual void optimizeConstantSizedGrids();
  string grid_create_name_;
  ReferenceRuntimeBuilder *ref_rt_builder_;
  virtual std::string GetStencilDomName() const;
  virtual SgExpression *BuildDomMaxRef(SgExpression *domain) const;
  virtual SgExpression *BuildDomMinRef(SgExpression *domain) const;
  virtual SgExpression *BuildDomMaxRef(SgExpression *domain, int dim) const;
  virtual SgExpression *BuildDomMinRef(SgExpression *domain, int dim) const;
  virtual SgExpression *BuildStencilDomRef(SgExpression *stencil) const;
  virtual SgExpression *BuildStencilDomMaxRef(SgExpression *stencil) const;
  virtual SgExpression *BuildStencilDomMaxRef(SgExpression *stencil,
                                              int dim) const;
  virtual SgExpression *BuildStencilDomMinRef(SgExpression *stencil) const;
  virtual SgExpression *BuildStencilDomMinRef(SgExpression *stencil,
                                              int dim) const;
  /*
  virtual SgVarRefExp *BuildDomainMinRef(SgClassDeclaration* dom_decl) const;
  virtual SgVarRefExp *BuildDomainMaxRef(SgClassDeclaration* dom_decl)
  const;
  */
  virtual SgExpression *BuildStencilFieldRef(SgExpression *stencil_ref,
                                             std::string name) const;
  virtual SgExpression *BuildStencilFieldRef(SgExpression *stencil_ref,
                                             SgExpression *field) const;
  virtual void TraceStencilRun(Run *run, SgScopeStatement *loop,
                               SgScopeStatement *cur_scope);
  
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_REFERENCE_TRANSLATOR_H_ */
