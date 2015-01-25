// Licensed under the BSD license. See LICENSE.txt for more details.

#ifndef PHYSIS_TRANSLATOR_MPI_OPENMP_TRANSLATOR_H_
#define PHYSIS_TRANSLATOR_MPI_OPENMP_TRANSLATOR_H_

#include "translator/mpi_translator.h"
#include "translator/reference_translator.h"

#define MPI_OPENMP_DIVISION_X_DEFAULT (1)
#define MPI_OPENMP_DIVISION_Y_DEFAULT (1)
#define MPI_OPENMP_DIVISION_Z_DEFAULT (2)

#define MPI_OPENMP_CACHESIZE_X_DEFAULT (100)
#define MPI_OPENMP_CACHESIZE_Y_DEFAULT (100)
#define MPI_OPENMP_CACHESIZE_Z_DEFAULT (100)

namespace physis {
namespace translator {

class MPIOpenMPTranslator : public MPITranslator {
 private:

 public:
  MPIOpenMPTranslator(const Configuration &config);
  virtual ~MPIOpenMPTranslator();

  //virtual void Translate();

  //virtual void SetUp(SgProject *project, TranslationContext *context);
  //virtual void Finish();

 protected:
  virtual void translateInit(SgFunctionCallExp *node);

  // Nothing performed for this target for now
  virtual void FixAST() {}

 public:
  virtual SgBasicBlock *BuildRunKernelBody(
      StencilMap *s, SgInitializedName *stencil_param);

 protected:
  int division_[3];
  int cache_size_[3];


};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_OPENMP_TRANSLATOR_H_ */
