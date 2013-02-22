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
  
  SgExprListExp *stencil_fw_val =
      mpi_rt_builder_->BuildStencilWidthFW(sr);
  assert(stencil_fw_val);
  SgVariableDeclaration *stencil_fw_var
      = sb::buildVariableDeclaration(
          "stencil_width_fw", ivec_type_,
          sb::buildAggregateInitializer(stencil_fw_val, ivec_type_));
  si::insertStatementAfter(dim_decl, stencil_fw_var);
  si::appendExpression(args, sb::buildVarRefExp(stencil_fw_var));

  SgExprListExp *stencil_bw_val =
      mpi_rt_builder_->BuildStencilWidthBW(sr);
  assert(stencil_bw_val);
  SgVariableDeclaration *stencil_bw_var
      = sb::buildVariableDeclaration(
          "stencil_width_bw", ivec_type_,
          sb::buildAggregateInitializer(stencil_bw_val, ivec_type_));
  si::insertStatementAfter(dim_decl, stencil_bw_var);
  si::appendExpression(args, sb::buildVarRefExp(stencil_bw_var));
  
  return;
}


} // namespace translator
} // namespace physis
