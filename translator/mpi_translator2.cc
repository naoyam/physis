// Licensed under the BSD license. See LICENSE.txt for more details.

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

bool MPITranslator2::translateGetKernel(SgFunctionCallExp *node,
                                        SgInitializedName *gv,
                                        bool is_periodic) {
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = si::getEnclosingFunctionDefinition(node);
  
  SgExpression *offset = BuildOffset(
      gv, nd, node->get_args(),
      true, is_periodic,
      rose_util::GetASTAttribute<GridGetAttribute>(
          node)->GetStencilIndexList(),
      scope);
  SgFunctionCallExp *base_addr = sb::buildFunctionCallExp(
      si::lookupFunctionSymbolInParentScopes("__PSGridGetBaseAddr"),
      sb::buildExprListExp(sb::buildVarRefExp(gv->get_name(),
                                              scope)));
  SgExpression *x = sb::buildPntrArrRefExp(
      sb::buildCastExp(base_addr,
                       sb::buildPointerType(gt->getElmType())),
      offset);
  rose_util::CopyASTAttribute<GridGetAttribute>(x, node);
  rose_util::GetASTAttribute<GridGetAttribute>(x)->offset() = offset;
  si::replaceExpression(node, x);
  return true;
}

} // namespace translator
} // namespace physis
