// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_CUDA_BUILDER_INTERFACE_H_
#define PHYSIS_TRANSLATOR_CUDA_BUILDER_INTERFACE_H_

#include "translator/translator_common.h"
#include "translator/builder_interface.h"


namespace physis {
namespace translator {

class CUDABuilderInterface: virtual public BuilderInterface {
 public:

  virtual ~CUDABuilderInterface() {}

  //! Generates a device type corresponding to a given grid type.
  /*!
    
    \param gt The grid type.
    \return A type object corresponding to the given grid type.
   */
  virtual SgType *BuildOnDeviceGridType(GridType *gt) = 0;

  //! Get the pointer to the object to be used on the device
  virtual SgExpression *BuildGridGetDev(SgExpression *grid_var,
                                        GridType *gt) = 0;

  virtual SgClassDeclaration *BuildGridDevTypeForUserType(
      SgClassDeclaration *grid_decl,
      const GridType *gt) = 0;
  virtual SgFunctionDeclaration *BuildGridNewFuncForUserType(
      const GridType *gt) = 0;
  virtual SgFunctionDeclaration *BuildGridFreeFuncForUserType(
      const GridType *gt) = 0;
  virtual SgFunctionDeclaration *BuildGridCopyinFuncForUserType(
      const GridType *gt) = 0;
  virtual SgFunctionDeclaration *BuildGridCopyoutFuncForUserType(
      const GridType *gt) = 0;
  virtual SgFunctionDeclaration *BuildGridGetFuncForUserType(
      const GridType *gt) = 0;
  virtual SgFunctionDeclaration *BuildGridEmitFuncForUserType(
      const GridType *gt) = 0;

  virtual SgInitializedName *GetDomArgParamInRunKernelFunc(
      SgFunctionParameterList *pl, int dim) = 0;

  //! Build a code block that sorrounds the call to 1D kernel.
  /*!
    The return object is the same as the call_site parameter when the
    kernel call should be just appended to the block (e.g., 1D and 2D
    stencil). When loops are introduced to cover the domain (e.g., 3D
    stencil), the returned call site refers to the inner-most block,
    where the call should be placed.
    
    \param stencil Stencil map object
    \param param Function parameter list of the RunKernel function
    \param indices Kernel index vector
    \param call_site Current call site block
    \return Call site for the kernel call
   */
  virtual SgScopeStatement *BuildKernelCallPreamble(
      StencilMap *stencil,      
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site) = 0;
  
  //! Helper function for BuildKernelCallPreamble for 1D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble1D(
      StencilMap *stencil,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site) = 0;

  //! Helper function for BuildKernelCallPreamble for 2D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble2D(
      StencilMap *stencil,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site) = 0;

  //! Helper function for BuildKernelCallPreamble for 3D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble3D(
      StencilMap *stencil,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site) = 0;

  //! Build variables for kernel indices
  /*!
    \param stencil Stencil map object
    \param call_site Call site basic block
    \param indices Output paramter to return variable declarations
   */
  virtual void BuildKernelIndices(
      StencilMap *stencil,
      SgBasicBlock *call_site,
      vector<SgVariableDeclaration*> &indices) = 0;

  //! Build a CUDA grid declaration for a stencil.
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
      SgScopeStatement *scope = NULL) = 0;

  //! Generates an expression of the x dimension of thread blocks.
  virtual SgExpression *BuildBlockDimX(int nd) = 0;
  //! Generates an expression of the y dimension of thread blocks.
  virtual SgExpression *BuildBlockDimY(int nd) = 0;
  //! Generates an expression of the z dimension of thread blocks.
  virtual SgExpression *BuildBlockDimZ(int nd) = 0;
  
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_CUDA_BUILDER_INTERFACE_H_ */

