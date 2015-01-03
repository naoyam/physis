// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_BUILDER_INTERFACE_H_
#define PHYSIS_BUILDER_INTERFACE_H_

#include "translator/translator_common.h"
#include "translator/translation_context.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

// REFACTORING: Move these to other files
SgFunctionCallExp *BuildTraceStencilPre(SgExpression *msg);
SgFunctionCallExp *BuildTraceStencilPost(SgExpression *time);

SgVariableDeclaration *BuildStopwatch(const std::string &name,
                                      SgScopeStatement *scope,
                                      SgScopeStatement *global_scope);
SgFunctionCallExp *BuildStopwatchStart(SgExpression *sw);
SgFunctionCallExp *BuildStopwatchStop(SgExpression *sw);

SgFunctionCallExp *BuildDomainGetBoundary(SgExpression *dom,
                                          int dim, int right,
                                          SgExpression *width,
                                          int factor, int offset);
//! Pure interface class defining builder methods
class BuilderInterface {
 public:
  virtual ~BuilderInterface() {}
  //!
  /*!
    \param 
    \return
   */
  virtual SgFunctionCallExp *BuildGridDim(
      SgExpression *grid_ref,
      int dim) = 0;
  //!
  /*!
    \param
    \return
   */
  virtual SgExpression *BuildGridRefInRunKernel(
      SgInitializedName *gv,
      SgFunctionDeclaration *run_kernel) = 0;

  //! Build a pointer expression to the raw array
  /*!
    Example: (float*)(a->p)
    
    \param gvref unused grid variable expression
    \param point_type point type
    \return a pointer to the array casted to the point type
  */
  virtual SgExpression *BuildGridBaseAddr(
      SgExpression *gvref, SgType *point_type) = 0;
  
  //! Build an offset expression.
  /*!
    Parameter offset_exprs will be used in the returned offset 
    expression without cloning.

    @param num_dim Number of dimensions.
    @param offset_exprs Index argument list.
    @param is_kernel True if the expression is used in a stencil
    kernel. 
    @param is_periodic True if it is a periodic access.
    @param sil The stencil index list of this access.
   */
  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic) = 0;
  //!
#if 0  
  /*!
    \param offset_exprs Free AST node of offset expressions
    
    Parameter offset_exprs will be used in the returned offset
    expression without cloning.
   */
  virtual SgExpression *BuildGet(  
    SgInitializedName *gv,
    SgExprListExp *offset_exprs,
    SgScopeStatement *scope,
    TranslationContext *tx, bool is_kernel,
    bool is_periodic) = 0;
#endif

  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,                  
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic) = 0;
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,                  
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name) = 0;
  virtual SgExpression *BuildGridGet(
      SgExpression *gvref,
      GridVarAttribute *gva,                  
      GridType *gt,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic,
      const string &member_name,
      const SgExpressionVector &array_indices) = 0;

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
      SgScopeStatement *scope=NULL) = 0;
  

  virtual SgBasicBlock *BuildGridSet(
      SgExpression *grid_var, int num_dims,
      const SgExpressionPtrList &indices, SgExpression *val) = 0;

  virtual SgFunctionCallExp *BuildGridGetID(SgExpression *grid_var) = 0;

  virtual SgExprListExp *BuildStencilOffsetMax(
      const StencilRange &sr) = 0;
  virtual SgExprListExp *BuildStencilOffsetMin(
      const StencilRange &sr) = 0;

  //! Build an ivec array containing the size of a given grid.
  /*!
    \param g Grid object
    \return ivec expression of the grid size
   */
  virtual SgExprListExp *BuildSizeExprList(const Grid *g) = 0;


  //! Build a domain min expression for a dimension.
  /*!
    \param domain Domain expression
    \param dim Dimension
    \return Domain min expression
   */
  virtual SgExpression *BuildDomMinRef(
      SgExpression *domain, int dim) = 0;
  //! Build a domain max expression for a dimension.
  /*!
    \param domain Domain expression
    \param dim Dimension
    \return Domain max expression
   */
  virtual SgExpression *BuildDomMaxRef(
      SgExpression *domain, int dim) = 0;

  //! Build a stencil field reference expression
  virtual SgExpression *BuildStencilFieldRef(
      SgExpression *stencil_ref, std::string name) = 0;
  //! Build a stencil field reference expression  
  virtual SgExpression *BuildStencilFieldRef(
      SgExpression *stencil_ref, SgExpression *field) = 0;
  //! Build a domain min expression for a dimension from a stencil
  virtual SgExpression *BuildStencilDomMinRef(
      SgExpression *stencil, int dim) = 0;
  //! Build a domain max expression for a dimension from a stencil  
  virtual SgExpression *BuildStencilDomMaxRef(
      SgExpression *stencil, int dim) = 0;

  //! Build the type to hold map arguments for a given stencil
  virtual SgClassDeclaration *BuildStencilMapType(StencilMap *s) = 0;
  //! Build the real map function for a given stencil
  virtual SgFunctionDeclaration *BuildMap(StencilMap *stencil) = 0;

  //! Generates a call to a stencil function.
  /*!
    Used by BuildRunKernel. 
    
    \param stencil The stencil kernel to call.
    \param index_args The index parameters declared in the stencil
    function.
    \param params Parameter list of the surrounding function    
    \return The call object to the stencil kernel.
   */
  virtual SgFunctionCallExp* BuildKernelCall(
      StencilMap *stencil, SgExpressionPtrList &index_args,
      SgFunctionParameterList *run_kernel_params) = 0;
  
  //! Generates an argument list for a call to a stencil function.
  /*!
    This is a helper function for BuildKernelCall.
    
    \param stencil The stencil kernel to call.
    \param index_args The index parameters declared in the stencil
    function.
    \param run_kernel_params Parameter list of the surrounding function
    \return The list of kernel call arguments.
   */
  virtual SgExprListExp *BuildKernelCallArgList(
      StencilMap *stencil,
      SgExpressionPtrList &index_args,
      SgFunctionParameterList *run_kernel_params) = 0;

  //! Build a function declaration that runs a stencil map. 
  /*!
    \param s The stencil map object.
    \param params Parameter of the run-kernel function
    \param body The function body
    \param indices Kernel index variables
    \return The function declaration.
   */
#if 0  
  virtual SgFunctionDeclaration *BuildRunKernelFunc(
      StencilMap *s, SgFunctionParameterList *params,
      SgBasicBlock *body,
      const vector<SgVariableDeclaration*> &indices) = 0;
#endif  
  //! Build a function declaration that runs a stencil map. 
  /*!
    \param s The stencil map object.
    \return The function declaration.
   */
  virtual SgFunctionDeclaration *BuildRunKernelFunc(
      StencilMap *s) = 0;
  //! A helper function for BuildRunKernel.
  /*!
    \param stencil Stencil map object
    \return Parameter list for the run-kernel function
  */
  virtual SgFunctionParameterList *BuildRunKernelFuncParameterList(
      StencilMap *s) = 0;
  //! A helper function for BuildRunKernel.
  /*!
    \param stencil The stencil map object.
    \param param Parameters for the run function.
    \param indices Output parameter to return generated indices
    \return The body of the run function.
   */
  virtual SgBasicBlock *BuildRunKernelFuncBody(
      StencilMap *stencil, SgFunctionParameterList *param,
      vector<SgVariableDeclaration*> &indices) = 0;

  //! Build a variable declaration represnting loop index
  /*
   * \param dim Dimension number
   * \param init Initializer expression
   * \param block Containing block
   * \return a variable declaration for the loop index
   */
  virtual SgVariableDeclaration *BuildLoopIndexVarDecl(
      int dim,
      SgExpression *init,
      SgScopeStatement *block) = 0;

  //! Build a function body for StencilRun function.
  /*!
    \param run Stencil run object
    \param run_func StencilRun function
   */
  virtual void BuildRunFuncBody(
      Run *run, SgFunctionDeclaration *run_func) = 0;
  /*!
    This is a helper function for BuildRunFuncBody. The run parameter
    contains stencil kernel calls and the number of iteration. This
    function generates a sequence of code to call the stencil kernels,
    which is then included in the for loop that iterates the given
    number of times. 
    
    \param run The stencil run object.
    \param run_func The run function.    
   */
  virtual SgBasicBlock *BuildRunFuncLoopBody(
      Run *run, SgFunctionDeclaration *run_func) = 0;

  
 protected:
  
  virtual SgExprListExp *BuildStencilOffset(
      const StencilRange &sr, bool is_max) = 0;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_RUNTIME_BUILDER_H_ */
