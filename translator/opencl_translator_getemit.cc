#include "translator/opencl_translator.h"
#include "translator/rose_util.h"
#include "translator/translation_context.h"

namespace si = SageInterface;
namespace sb = SageBuilder;

namespace physis {
namespace translator {

SgExpression *OpenCLTranslator::buildOffset(
    SgInitializedName *gv,
    SgScopeStatement *scope,
    int numDim,
    SgExpressionPtrList &args)
{

  /*
    Create the offset of PSGridGet or Emit

    if (3d)
    offset = grid_dimx(ga) * grid_dimy(ga) * k + grid_dimx(ga) * j + i
    if (2d)
    offset = grid_dimx(ga) * j + i
    if (1d)
    offset = i

    Where i is the first argument of &args, and j, k is so on.
    For OpenCL, grid_dimx(ga) should be translated as __PS_g_dim_x
    i.e. For 3d, for example
    x + y * __PS_ga_dim_x + z * __PS_ga_dim_x * __PS_ga_dim_y

  */

  SgExpression *offset = NULL;
  SgTreeCopy ch;
  for (int i = 0; i < numDim; i++) {
    // z-4, for example
    SgExpression *dim_offset = isSgExpression(args[i]->copy(ch));

    for (int j = 0; j < i; j++) { // j is the dimension
      // __PS_ga_dim_x
      std::string newargname = name_var_grid_dim(gv->get_name().getString(), j);
       // z * __PS_ga_dim_x * __PS_ga_dim_y, for example
      dim_offset =
          sb::buildMultiplyOp(dim_offset,
                              sb::buildVarRefExp(newargname));
    }
    if (offset) {
      offset = sb::buildAddOp(offset, dim_offset);
    } else {
      offset = dim_offset;
    }
  }

  return offset;

} // buildOffset


void OpenCLTranslator::translateGet(
    SgFunctionCallExp *node,
    SgInitializedName *gv,
    bool isKernel
)
{
  /*
    This function should return __PS_ga_buf[offset]
    No cast should be needed as compared with CUDA translation code.
  */
  SgTreeCopy tc;

  // Dimension
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();

  SgScopeStatement *scope = getContainingScopeStatement(node);
  // offset: x + y * __PS_ga_dim_x + z * __PS_ga_dim_x * __PS_ga_dim_y
  SgExpression *offset = buildOffset(gv, scope, nd,
                                     node->get_args()->get_expressions());

  // p0: __PS_ga_buf
  std::string newargname = name_var_gridptr(gv->get_name().getString());
  SgExpression *p0 = sb::buildVarRefExp(newargname);

  // No cast needed
  // p0 = sb::buildCastExp(p0, sb::buildPointerType(gt->getElmType()));

  // p0: __PS_ga_buf[offset]
  p0 = sb::buildPntrArrRefExp(p0, offset);

  si::replaceExpression(node, p0);

} // translateGet


void OpenCLTranslator::translateEmit(SgFunctionCallExp *node,
                                        SgInitializedName *gv) {
  /*
    This function should return __PS_ga_buf[offset] = value;
    No cast should be needed as compared with CUDA translation code.
  */

  // Dimension
  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();

  // Get the arguments of _the function_ (in device), not the argument of
  // PSGridEmit.
  // The first (nd) arguments should be "int x, int y, int z", for example.
  SgInitializedNamePtrList &params = getContainingFunction(node)->get_args();
  SgExpressionPtrList args;
  FOREACH(it, params.begin(), params.end()) {
    SgInitializedName *p = *it;
    args.push_back(sb::buildVarRefExp(p, getContainingScopeStatement(node)));
  }

  // Now args should be "x, y, z", for example
  // The following offset should return
  // x + y * __PS_ga_dim_x + z * __PS_ga_dim_x * __PS_ga_dim_y, for example
  SgExpression *offset = buildOffset(gv, getContainingScopeStatement(node),
                                     nd, args);

  // __PS_ga_buf
  std::string newargname = name_var_gridptr(gv->get_name().getString());
  SgExpression *p1 = sb::buildVarRefExp(newargname);
  // __PS_ga_buf[offset]
  SgExpression *lhs = sb::buildPntrArrRefExp(p1, offset);
  LOG_DEBUG() << "emit lhs: " << lhs->unparseToString() << "\n";

  SgTreeCopy tc;
  SgExpression *rhs =
      isSgExpression(node->get_args()->get_expressions()[0]->copy(tc));
  LOG_DEBUG() << "emit rhs: " << rhs->unparseToString() << "\n";

  SgExpression *emit = sb::buildAssignOp(lhs, rhs);
  LOG_DEBUG() << "emit: " << emit->unparseToString() << "\n";

  si::replaceExpression(node, emit, false);

} // translateEmit


} // namespace translator
} // namespace physis
