// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_
#define PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_

#include "translator/translator_common.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/cuda_runtime_builder.h"

namespace physis {
namespace translator {

SgFunctionCallExp *BuildGridGetDev(SgExpression *grid_var);
SgFunctionCallExp *BuildGetLocalSize(SgExpression *dim);
SgFunctionCallExp *BuildGetLocalOffset(SgExpression *dim);
SgFunctionCallExp *BuildDomainShrink(SgExpression *dom,
                                     SgExpression *width);
SgExpression *BuildStreamBoundaryKernel(int idx);

class MPICUDARuntimeBuilder: public MPIRuntimeBuilder {
 public:
  MPICUDARuntimeBuilder(SgScopeStatement *global_scope):
      MPIRuntimeBuilder(global_scope),
      cuda_rt_builder_(new CUDARuntimeBuilder(global_scope))
  {}
  
  virtual ~MPICUDARuntimeBuilder() {
    delete cuda_rt_builder_;
  }

  virtual SgExpression *BuildGridOffset(
      SgExpression *gvref, int num_dim,
      const SgExpressionPtrList *offset_exprs, bool is_kernel,
      bool is_periodic,
      const StencilIndexList *sil) {
    return cuda_rt_builder_->BuildGridOffset(
        gvref, num_dim, offset_exprs, is_kernel,
        is_periodic, sil);
  }

  
 protected:
  CUDARuntimeBuilder *cuda_rt_builder_;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_CUDA_RUNTIME_BUILDER_H_ */

