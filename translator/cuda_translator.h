// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_CUDA_TRANSLATOR_H_

#include "translator/reference_translator.h"
#include "translator/cuda_runtime_builder.h"

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
  
  virtual void FixAST();
  virtual void FixGridType();

  virtual CUDABuilderInterface *builder() {
    return dynamic_cast<CUDABuilderInterface*>(rt_builder_);
  }

  
 public:

  virtual void SetUp(SgProject *project, TranslationContext *context,
                     BuilderInterface *rt_builder);

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
    \param dim Domain dimension
    \param dom_dim_x
    \param dom_dim_y
    \param block_dim_x
    \param block_dim_y    
    \param scope The scope where the grid is declared.
    \return The grid declaration.
   */
  virtual SgVariableDeclaration *BuildGridDimDeclaration(
      const SgName &name,
      int dim,
      SgExpression *dom_dim_x, SgExpression *dom_dim_y,      
      SgExpression *block_dim_x, SgExpression *block_dim_y,
      SgScopeStatement *scope = NULL) const;

  virtual void ProcessUserDefinedPointType(SgClassDeclaration *grid_decl,
                                           GridType *gt);

  virtual void TranslateKernelDeclaration(SgFunctionDeclaration *node);
  virtual void TranslateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool is_kernel, bool is_periodic);
  virtual void TranslateGetForUserDefinedType(
      SgDotExp *node, SgPntrArrRefExp *array_top);
  virtual void TranslateEmit(SgFunctionCallExp *node,
                             GridEmitAttribute *attr);
  virtual void TranslateFree(SgFunctionCallExp *node,
                             GridType *gt);
  virtual void TranslateCopyin(SgFunctionCallExp *node,
                               GridType *gt);
  virtual void TranslateCopyout(SgFunctionCallExp *node,
                                GridType *gt);

  virtual void Visit(SgExpression *node);

  //virtual void TranslateSet(SgFunctionCallExp *node,
  //SgInitializedName *gv);
  //! Generates a basic block of the stencil run function.
  /*!
    \param run The top-level function basic block.
    \param run The stencil run object.
    \param run_func The run function.
   */
  virtual void BuildRunBody(
      SgBasicBlock *block, Run *run, SgFunctionDeclaration *run_func);
  //! Generates a basic block of the run loop body.
  /*!
    This is a helper function for BuildRunBody. The run parameter
    contains stencil kernel calls and the number of iteration. This
    function generates a sequence of code to call the stencil kernels,
    which is then included in the for loop that iterates the given
    number of times. 
    
    \param run The stencil run object.
    \param outer_block The outer block where the for loop is included.
    \param run_func The run function.    
   */
  virtual SgBasicBlock *BuildRunLoopBody(SgBasicBlock *outer_block,
                                         Run *run,
                                         SgFunctionDeclaration *run_func);
  //! Generates an argument list for a CUDA kernel call.
  /*!
    \param stencil_idx The index of the stencil in PSStencilRun.
    \param sm The stencil map object.
    \param sv Stencil parameter symbol
    \return The argument list for the call to the stencil map.
   */
  virtual SgExprListExp *BuildCUDAKernelArgList(
      int stencil_idx, StencilMap *sm, SgVariableSymbol *sv) const;

  //! Generates an expression of the x dimension of thread blocks.
  virtual SgExpression *BuildBlockDimX(int nd);
  //! Generates an expression of the y dimension of thread blocks.  
  virtual SgExpression *BuildBlockDimY(int nd);
  //! Generates an expression of the z dimension of thread blocks.
  virtual SgExpression *BuildBlockDimZ(int nd);


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
