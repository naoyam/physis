#include "translator/mpi_opencl_translator.h"
#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/rose_util.h"
#include "translator/rose_ast_attribute.h"

namespace pu = physis::util;
namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

SgType *MPIOpenCLTranslator::BuildOnDeviceGridType(GridType *gt, int inner_device){

  PSAssert(gt);
  string gt_name;
  int nd = gt->getNumDim();
  string elm_name = gt->getElmType()->unparseToString();
  std::transform(elm_name.begin(), elm_name.begin() +1,
                 elm_name.begin(), toupper);
  string ondev_type_name = "__PSGrid" + toString(nd) + "D"
      + elm_name + "Dev";
  if (inner_device)
    ondev_type_name += "_CLKernel";
  LOG_DEBUG() << "On device grid type name: "
              << ondev_type_name << "\n";
  SgType *t =
      sb::buildOpaqueType(ondev_type_name, global_scope_);
  PSAssert(t);

  return t;
}

SgType *MPIOpenCLTranslator::BuildOnDeviceDomType(void){
  SgType *t_ret = sb::buildOpaqueType("__PSDomain_CLKernel", global_scope_);
  return t_ret;
}


SgIfStmt *MPIOpenCLTranslator::BuildDomainInclusionCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgExpression *dom_ref) const {
  // check x and y domain coordinates, like:
  // if (x < dom.local_min[0] || x >= dom.local_max[0] ||
  //     y < dom.local_min[1] || y >= dom.local_max[1]) {
  //   return;
  // }
  
  SgExpression *test_all = NULL;
  ENUMERATE (dim, index_it, indices.begin(), indices.end()) {
    SgExpression *idx = sb::buildVarRefExp(*index_it);
    SgExpression *dom_min = sb::buildPntrArrRefExp(
        sb::buildDotExp(dom_ref,
                        sb::buildVarRefExp("local_min")),
        sb::buildIntVal(dim));
    SgExpression *dom_max = sb::buildPntrArrRefExp(
        sb::buildDotExp(dom_ref,
                        sb::buildVarRefExp("local_max")),
        sb::buildIntVal(dim));
    SgExpression *test = sb::buildOrOp(
        sb::buildLessThanOp(idx, dom_min),
        sb::buildGreaterOrEqualOp(idx, dom_max));
    if (test_all) {
      test_all = sb::buildOrOp(test_all, test);
    } else {
      test_all = test;
    }
  }
  SgIfStmt *ifstmt =
      sb::buildIfStmt(test_all, sb::buildReturnStmt(), NULL);
  return ifstmt;
}

SgIfStmt *MPIOpenCLTranslator::BuildDomainInclusionInnerCheck(
    const vector<SgVariableDeclaration*> &indices,
    SgExpression *dom_ref, SgExpression *width,
    SgStatement *ifclause) const {
  SgExpression *test_all = NULL;
  ENUMERATE (dim, index_it, indices.begin(), indices.end()) {
    SgExpression *idx = sb::buildVarRefExp(*index_it);
    SgExpression *test = sb::buildOrOp(
        sb::buildLessThanOp(idx,
                            sb::buildAddOp(BuildDomMinRef(dom_ref, dim),
                                           width)),
        sb::buildGreaterOrEqualOp(
            idx,
            sb::buildSubtractOp(BuildDomMaxRef(dom_ref, dim), width)));
    if (test_all) {
      test_all = sb::buildOrOp(test_all, test);
    } else {
      test_all = test;
    }
  }
  SgIfStmt *ifstmt = sb::buildIfStmt(test_all, ifclause, NULL);
  return ifstmt;
}


} // namespace translator
} // namespace physis
