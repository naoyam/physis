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
  virtual void Translate();
  virtual void Optimize();
  virtual void SetUp(SgProject *project, TranslationContext *context,
                     RuntimeBuilder *rt_builder);  
  virtual void Finish();  

  bool flag_constant_grid_size_optimization() const {
    return flag_constant_grid_size_optimization_;
  }
  void set_flag_constant_grid_size_optimization(bool flag) {
    flag_constant_grid_size_optimization_ = flag;
  }

 protected:
  bool validate_ast_;
  //! Fixes inconsistency in AST.
  virtual void FixAST();
  //! Validates AST consistency.
  /*!
    Aborts when validation fails. Checking can be disabled with toggle
    variable validate_ast_.
   */
  virtual void ValidateASTConsistency();
  virtual void TranslateKernelDeclaration(SgFunctionDeclaration *node);
  virtual void TranslateNew(SgFunctionCallExp *node, GridType *gt);
  virtual SgExprListExp *generateNewArg(GridType *gt, Grid *g,
                                        SgVariableDeclaration *dim_decl);
  virtual void appendNewArgExtra(SgExprListExp *args, Grid *g,
                                 SgVariableDeclaration *dim_decl);
  virtual void TranslateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool is_kernel,
                            bool is_periodic);
  virtual void TranslateEmit(SgFunctionCallExp *node,
                             GridEmitAttribute *attr);
  virtual void RemoveEmitDummyExp(SgExpression *emit);
  virtual void TranslateSet(SgFunctionCallExp *node, SgInitializedName *gv);
  virtual void TranslateMap(SgFunctionCallExp *node, StencilMap *s);
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
    \param param Parameter list of the run function
    \return The body of the run function.
   */
  virtual SgBasicBlock *BuildRunKernelBody(
      StencilMap *s, SgFunctionParameterList *param,
      vector<SgVariableDeclaration*> &indices);
#ifdef DEPRECATED  
  virtual void appendGridSwap(StencilMap *mc, const string &stencil,
                              bool is_stencil_ptr,
                              SgScopeStatement *scope);
#endif
  virtual SgFunctionCallExp* BuildKernelCall(
      StencilMap *s, SgExpressionPtrList &indexArgs,
      SgInitializedName *stencil_param);

  virtual void DefineMapSpecificTypesAndFunctions();
  virtual void InsertStencilSpecificType(StencilMap *s,
                                         SgClassDeclaration *type_decl);
  virtual void InsertStencilSpecificFunc(StencilMap *s,
                                         SgFunctionDeclaration *func);
  
  virtual void BuildRunBody(
      SgBasicBlock *block, Run *run, SgFunctionDeclaration *run_func);
  virtual SgFunctionDeclaration *BuildRun(Run *run);
  virtual void TranslateRun(SgFunctionCallExp *node, Run *run);

  /** generate dlopen and dlsym code
   * @param[in] run
   * @param[in] ref ... function reference  '__PSStencilRun_0(1, ...);'
   * @param[in] index ... VAR's expression
   * @param[in] scope
   * @return    statement of dlopen and dlsym
   */
  virtual SgStatement *GenerateDlopenDlsym(
      Run *run, SgFunctionRefExp *ref,
      SgExpression *index, SgScopeStatement *scope);
  /** generate trial code
   * @param[in] run
   * @param[in] ref ... function reference  '__PSStencilRun_0(1, ...);'
   * @return    function declaration
   */
  virtual SgFunctionDeclaration *GenerateTrial(
      Run *run, SgFunctionRefExp *ref);
  /** add dynamic parameter
   * @param[in/out] parlist ... parameter list
   */
  virtual void AddDynamicParameter(SgFunctionParameterList *parlist);
  /** add dynamic argument
   * @param[in/out] args ... arguments
   * @param[in] a_exp ... index expression
   */
  virtual void AddDynamicArgument(SgExprListExp *args, SgExpression *a_exp);
  /** add some code after dlclose()
   * @param[in] scope
   */
  virtual void AddSyncAfterDlclose(SgScopeStatement *scope);

  virtual void TranslateReduceGrid(Reduce *rd);
  virtual void TranslateReduceKernel(Reduce *rd);
  //! Build a real function for reducing a grid.
  /*!
    \param rd A reduction of grid.
    \return A function declaration for reducing the grid.
   */
  virtual SgFunctionDeclaration *BuildReduceGrid(Reduce *rd);

  virtual void optimizeConstantSizedGrids();
  string grid_create_name_;
  virtual std::string GetStencilDomName() const;
  virtual void TraceStencilRun(Run *run, SgScopeStatement *loop,
                               SgScopeStatement *cur_scope);
  virtual void FixGridType();
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_REFERENCE_TRANSLATOR_H_ */
