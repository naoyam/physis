// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/cuda_builder_interface.h"
#include "translator/cuda_runtime_builder.h"

namespace physis {
namespace translator {

// REFACTORING: move these into the builder class
SgFunctionCallExp *BuildGridGetDev(SgExpression *grid_var);
SgFunctionCallExp *BuildGetLocalSize(SgExpression *dim);
SgFunctionCallExp *BuildGetLocalOffset(SgExpression *dim);
SgFunctionCallExp *BuildDomainShrink(SgExpression *dom,
                                     SgExpression *width);
SgExpression *BuildStreamBoundaryKernel(int idx);

class MPICUDARuntimeBuilder: virtual public MPIRuntimeBuilder {
 public:
  MPICUDARuntimeBuilder(SgScopeStatement *global_scope):
      ReferenceRuntimeBuilder(global_scope),
      MPIRuntimeBuilder(global_scope),
      cuda_rt_builder_(new CUDARuntimeBuilder(global_scope))
  {}
  
  virtual ~MPICUDARuntimeBuilder() {
    delete cuda_rt_builder_;
  }

  virtual SgExpression *BuildGridBaseAddr(
      SgExpression *gvref, SgType *point_type);

  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
      const SgExpressionPtrList *offset_exprs,
      const StencilIndexList *sil,
      bool is_kernel,
      bool is_periodic) {
    return cuda_rt_builder_->BuildGridOffset(
        gvref, num_dim, offset_exprs, sil, is_kernel,
        is_periodic);
  }

  virtual SgExprListExp *BuildKernelCallArgList(
      StencilMap *stencil,
      SgExpressionPtrList &index_args,
      SgFunctionParameterList *params);

  virtual SgIfStmt *BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgInitializedName *dom_arg, SgStatement *true_stmt);
  
  virtual SgFunctionDeclaration *BuildRunKernelFunc(StencilMap *s);
  virtual SgBasicBlock *BuildRunKernelFuncBody(
      StencilMap *stencil, SgFunctionParameterList *param,
      vector<SgVariableDeclaration*> &indices);

  //! Generates a device type corresponding to a given grid type.
  /*!
    This is not derived.
    
    \param gt The grid type.
    \return A type object corresponding to the given grid type.
   */
  virtual SgType *BuildOnDeviceGridType(GridType *gt);
  
  
 protected:
  CUDARuntimeBuilder *cuda_rt_builder_;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_ */

