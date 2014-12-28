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
    This is not derived.
    
    \param gt The grid type.
    \return A type object corresponding to the given grid type.
   */
  virtual SgType *BuildOnDeviceGridType(GridType *gt) = 0;

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
  

  //! Build a code block that sorrounds the call to 1D kernel.
  /*!
    The return object is the same as the call_site parameter when the
    kernel call should be just appended to the block (e.g., 1D and 2D
    stencil). When loops are introduced to cover the domain (e.g., 3D
    stencil), the returned call site refers to the inner-most block,
    where the call should be placed.
    
    \param stencil Stencil map object
    \param dom_arg The domain argument to StencilMap
    \param param Function parameter list of the RunKernel function
    \param indices Kernel index vector
    \param call_site Current call site block
    \return Call site for the kernel call
   */
  virtual SgScopeStatement *BuildKernelCallPreamble(
      StencilMap *stencil,      
      SgInitializedName *dom_arg,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site) = 0;
  
  //! Helper function for BuildKernelCallPreamble for 1D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble1D(
      StencilMap *stencil,
      SgInitializedName *dom_arg,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site) = 0;

  //! Helper function for BuildKernelCallPreamble for 2D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble2D(
      StencilMap *stencil,
      SgInitializedName *dom_arg,
      SgFunctionParameterList *param,      
      vector<SgVariableDeclaration*> &indices,
      SgScopeStatement *call_site) = 0;

  //! Helper function for BuildKernelCallPreamble for 3D stencil
  virtual SgScopeStatement *BuildKernelCallPreamble3D(
      StencilMap *stencil,
      SgInitializedName *dom_arg,
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
};

} // namespace translator
} // namespace physis


#endif /* PHYSIS_TRANSLATOR_CUDA_BUILDER_INTERFACE_H_ */

