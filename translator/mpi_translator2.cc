// Copyright 2011-2012, RIKEN AICS.
// All rights reserved.
//
// This file is distributed under the BSD license. See LICENSE.txt for
// details.

#include "translator/mpi_translator2.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

MPITranslator2::MPITranslator2(const Configuration &config):
    MPITranslator(config) {
  grid_create_name_ = "__PSGridNewMPI2";
}

void MPITranslator2::appendNewArgExtra(SgExprListExp *args,
                                       Grid *g,
                                       SgVariableDeclaration *dim_decl) {
  LOG_DEBUG() << "Append New extra arg for "
              << *g << "\n";
  MPITranslator::appendNewArgExtra(args, g, dim_decl);
  
  const StencilRange &sr = g->stencil_range();
  
  SgExprListExp *stencil_min_val =
      mpi_rt_builder_->BuildStencilOffsetMin(sr);
  assert(stencil_min_val);
  SgVariableDeclaration *stencil_min_var
      = sb::buildVariableDeclaration(
          "stencil_offset_min", ivec_type_,
          sb::buildAggregateInitializer(stencil_min_val, ivec_type_));
  si::insertStatementAfter(dim_decl, stencil_min_var);
  si::appendExpression(args, sb::buildVarRefExp(stencil_min_var));

  SgExprListExp *stencil_max_val =
      mpi_rt_builder_->BuildStencilOffsetMax(sr);
  assert(stencil_max_val);
  SgVariableDeclaration *stencil_max_var
      = sb::buildVariableDeclaration(
          "stencil_offset_max", ivec_type_,
          sb::buildAggregateInitializer(stencil_max_val, ivec_type_));
  si::insertStatementAfter(dim_decl, stencil_max_var);
  si::appendExpression(args, sb::buildVarRefExp(stencil_max_var));
  
  return;
}


} // namespace translator
} // namespace physis
