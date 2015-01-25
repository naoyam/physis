// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_OPENCL_OPTIMIZER_H_
#define PHYSIS_TRANSLATOR_MPI_OPENCL_OPTIMIZER_H_

#include "translator/translator_common.h"
#include "translator/mpi_opencl_translator.h"

namespace physis {
namespace translator {

class MPIOpenCLOptimizer {
 public:
  MPIOpenCLOptimizer(const MPIOpenCLTranslator &trans);
  virtual ~MPIOpenCLOptimizer() {}
  virtual void GridPreCalcAddr(SgFunctionDeclaration *func);
 protected:
  const MPIOpenCLTranslator &trans_;
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_OPENCL_OPTIMIZER_H_ */
