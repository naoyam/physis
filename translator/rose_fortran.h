// Copyright 2011-2013, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#ifndef PHYSIS_TRANSLATOR_ROSE_FORTRAN_H_
#define PHYSIS_TRANSLATOR_ROSE_FORTRAN_H_

#include "translator/translator_common.h"
#include "physis/internal_common.h"

namespace physis {
namespace translator {
namespace rose_fortran {

SgDerivedTypeStatement *BuildDerivedTypeStatementAndDefinition(
    std::string name, SgScopeStatement *scope);

SgFortranDo *BuildFortranDo(SgExpression *initialization,
                            SgExpression *bound,
                            SgExpression *increment,
                            SgBasicBlock *body);

SgAllocateStatement *BuildAllocateStatement();


}  // namespace rose_fortran
}  // namespace translator
}  // namespace physis

#endif /* PHYSIS_TRANSLATOR_ROSE_FORTRAN_H__ */
