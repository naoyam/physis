// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_TRANSLATOR_MPI_TRANSLATOR2_H_
#define PHYSIS_TRANSLATOR_MPI_TRANSLATOR2_H_

#include "translator/mpi_translator.h"

namespace physis {
namespace translator {

class MPITranslator2: public MPITranslator {
 public:
  MPITranslator2(const Configuration &config);
  virtual ~MPITranslator2() {}
  virtual void appendNewArgExtra(
      SgExprListExp *args,
      Grid *g,
      SgVariableDeclaration *dim_decl);
};

} // namespace translator
} // namespace physis

#endif /* PHYSIS_TRANSLATOR_MPI_TRANSLATOR2_H_ */

