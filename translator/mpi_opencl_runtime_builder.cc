#include "translator/translator_common.h"
#include "translator/mpi_opencl_translator.h"

namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

SgFunctionCallExp *MPIOpenCLTranslator::BuildGridGetDev(SgExpression *grid_var) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGridGetDev");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(grid_var));
  return fc;
}

SgFunctionCallExp *MPIOpenCLTranslator::BuildGetLocalSize(SgExpression *dim) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGetLocalSize");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(dim));
  return fc;
}  
SgFunctionCallExp *MPIOpenCLTranslator::BuildGetLocalOffset(SgExpression *dim) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSGetLocalOffset");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(fs, sb::buildExprListExp(dim));
  return fc;
}

SgFunctionCallExp *MPIOpenCLTranslator::BuildDomainShrink(SgExpression *dom,
                                                          SgExpression *width) {
  SgFunctionSymbol *fs
      = si::lookupFunctionSymbolInParentScopes("__PSDomainShrink");
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(
          fs, sb::buildExprListExp(dom, width));
  return fc;
}

SgFunctionCallExp *MPIOpenCLTranslator::BuildCLThreadSynchronize(void)
{
  SgFunctionCallExp *fc =
      sb::buildFunctionCallExp(
          sb::buildFunctionRefExp("__PS_CL_ThreadSynchronize"),
          sb::buildExprListExp(NULL)
                               );
  return fc;
}

SgExpression *MPIOpenCLTranslator::BuildStreamBoundaryKernel(int idx) {
  SgVarRefExp *inner_stream = sb::buildVarRefExp("stream_boundary_kernel");
  return sb::buildPntrArrRefExp(inner_stream, sb::buildIntVal(idx));
}
  
} // namespace translator
} // namespace physis
