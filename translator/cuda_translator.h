// Copyright 2011, Tokyo Institute of Technology.
// All rights reserved.
//
// This file is distributed under the license described in
// LICENSE.txt.
//
// Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#ifndef PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_

#include "translator/reference_translator.h"

namespace physis {
namespace translator {

class CUDATranslator : public ReferenceTranslator {
 public:
  CUDATranslator(const Configuration &config);
  virtual ~CUDATranslator() {}
  
 protected:  
  int block_dim_x_;
  int block_dim_y_;
  int block_dim_z_;
  /** hold all CUDA_BLOCK_SIZE values */
  std::vector<SgExpression *> cuda_block_size_vals_;
  /** hold __cuda_block_size_struct type */
  SgType *cuda_block_size_type_;

  //! An optimization flag on grid address calculations.
  /*!
    If this flag is true, the Translator generates for grid_get in the
    begining of the kernel. This optimization would be effective for
    some cases such as consecutive code block of 'if' and
    'grid_get'. In which case, size of if block could be reduced.
    For example,
    \code
    kernel(x, y, z, g, ...) {
      ...
      v1 = grid_get(g, x, y, z);
      v2 = grid_get(g, x+1, y, z);      
      ...
    }
    \endcode
    It generates code as follows:
    \code
    kernel(x, y, z, g, ...) {
      float *p = (float *)(g->p0) + z*nx*ny + y*nx + x;
      ...
      v1 = *p;
      v2 = *(p+1);
      ...
    }
    \endcode
  */  
  bool flag_pre_calc_grid_address_;
  
  virtual void FixAST();
  virtual void FixGridType();
  
 public:

  virtual void SetUp(SgProject *project, TranslationContext *context,
                     RuntimeBuilder *rt_builder);

  virtual void appendNewArgExtra(SgExprListExp *args,
                                 Grid *g,
                                 SgVariableDeclaration *dim_decl);
  

  //! Generates a CUDA grid declaration for a stencil.
  /*!
    The x and y dimensions are decomposed by the thread block, whereas
    the z dimension is processed entirely by each thread block.

    Each dimension parameter must be a free AST node and not be used
    other tree locations.
    
    \param name The name of the grid variable.
    \param dom_dim_x
    \param dom_dim_y
    \param block_dim_x
    \param block_dim_y    
    \param scope The scope where the grid is declared.
    \return The grid declaration.
   */
  virtual SgVariableDeclaration *BuildGridDimDeclaration(
      const SgName &name,
      SgExpression *dom_dim_x, SgExpression *dom_dim_y,      
      SgExpression *block_dim_x, SgExpression *block_dim_y,
      SgScopeStatement *scope = NULL) const;

  virtual void ProcessUserDefinedPointType(SgClassDeclaration *grid_decl,
                                           GridType *gt);

  virtual void TranslateKernelDeclaration(SgFunctionDeclaration *node);
  virtual void TranslateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool is_kernel, bool is_periodic);
  virtual void TranslateEmit(SgFunctionCallExp *node,
                             SgInitializedName *gv);
  virtual void TranslateFree(SgFunctionCallExp *node,
                             GridType *gt);
  virtual void TranslateCopyin(SgFunctionCallExp *node,
                               GridType *gt);
  
  //virtual void TranslateSet(SgFunctionCallExp *node,
  //SgInitializedName *gv);
  //! Generates a basic block of the stencil run function.
  /*!
    \param run The top-level function basic block.
    \param run The stencil run object.
    \param run_func The run function.
   */
  virtual void GenerateRunBody(
      SgBasicBlock *block, Run *run, SgFunctionDeclaration *run_func);
  //! Generates a basic block of the run loop body.
  /*!
    This is a helper function for BuildRunBody. The run parameter
    contains stencil kernel calls and the number of iteration. This
    function generates a sequence of code to call the stencil kernels,
    which is then included in the for loop that iterates the given
    number of times. 
    
    \param run The stencil run object.
    \return outer_block The outer block where the for loop is included.
   */
  virtual SgBasicBlock *BuildRunLoopBody(Run *run,
                                         SgScopeStatement *outer_block);
  //! Generates an argument list for a CUDA kernel call.
  /*!
    \param stencil_idx The index of the stencil in PSStencilRun.
    \param sm The stencil map object.
    \param sv_name Name of the stencil parameter.
    \return The argument list for the call to the stencil map.
   */
  virtual SgExprListExp *BuildCUDAKernelArgList(
      int stencil_idx, StencilMap *sm, const string &sv_name) const;

  //! Generates a CUDA function declaration that runs a stencil map. 
  /*!
    \param s The stencil map object.
    \return The function declaration.
   */
  virtual SgFunctionDeclaration *BuildRunKernel(StencilMap *s);
  //! A helper function for BuildRunKernel.
  /*!
    \param stencil The stencil map object.
    \param dom_arg The stencil domain.
    \return The body of the run function.
   */
  virtual SgBasicBlock *BuildRunKernelBody(
      StencilMap *stencil, SgInitializedName *dom_arg);
  //! Generates an argument list for a call to a stencil function.
  /*!
    This is a helper function for BuildKernelCall.
    
    \param stencil The stencil kernel to call.
    \param index_args The index parameters declared in the stencil
    function.
    \return The list of kernel call arguments.
   */
  virtual SgExprListExp* BuildKernelCallArgList(
      StencilMap *stencil, SgExpressionPtrList &index_args);
  //! Generates a call to a stencil function.
  /*!
    Used by BuildRunKernel. 
    
    \param stencil The stencil kernel to call.
    \param index_args The index parameters declared in the stencil
    function.
    \return The call object to the stencil kernel.
   */
  virtual SgFunctionCallExp* BuildKernelCall(
      StencilMap *stencil, SgExpressionPtrList &index_args);
  //! Generates an expression of the x dimension of thread blocks.
  virtual SgExpression *BuildBlockDimX();
  //! Generates an expression of the y dimension of thread blocks.  
  virtual SgExpression *BuildBlockDimY();
  //! Generates an expression of the z dimension of thread blocks.
  virtual SgExpression *BuildBlockDimZ();
  //! Generates an IF block to exclude indices outside a domain.
  /*!
    \param indices The indices to check.
    \param dom_arg Name of the domain parameter.
    \return The IF block.
   */
  virtual SgIfStmt *BuildDomainInclusionCheck(
      const vector<SgVariableDeclaration*> &indices,
      SgInitializedName *dom_arg) const;
  //! Generates a device type corresponding to a given grid type.
  /*!
    \param gt The grid type.
    \return A type object corresponding to the given grid type.
   */
  virtual SgType *BuildOnDeviceGridType(GridType *gt) const;
  bool flag_pre_calc_grid_address() const {
    return flag_pre_calc_grid_address_;
  }

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

};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_ */
