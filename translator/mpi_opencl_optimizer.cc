// Licensed under the BSD license. See LICENSE.txt for more details.

#include "translator/mpi_opencl_optimizer.h"
#include "translator/rose_ast_attribute.h"
#include "translator/rose_util.h"
#include "translator/translation_context.h"

namespace pu = physis::util;
namespace sb = SageBuilder;
namespace si = SageInterface;

namespace physis {
namespace translator {

MPIOpenCLOptimizer::MPIOpenCLOptimizer(const MPIOpenCLTranslator &trans)
    : trans_(trans) {}


// REFACTORING: mostly the same as GetMaximumNumberOfDimensions in
// mpi_opencl_translator.cc 
static int GetIndexParams(SgFunctionDeclaration *func,
                          SgInitializedNamePtrList &indices) {
  SgFunctionParameterList *params = func->get_parameterList();
  SgInitializedNamePtrList &param_args = params->get_args();
  int dim = 0;
  ENUMERATE (i, it, param_args.begin(), param_args.end()) {
    SgInitializedName *p = *it;
    if (!rose_util::IsIntLikeType(p)) {
      dim = i;
      break;
    }
    indices.push_back(p);    
  }
  return dim;
}

static SgExpression *BuildOffset(SgInitializedName *gv,
                                 const StencilIndexList &sil,
                                 int num_dims) {
  PSAssert(StencilIndexRegularOrder(sil, sil.size()));
  SgExpression *offset = NULL;
  for (int i = 0; i < num_dims; ++i) {
    const StencilIndex &si = sil[i];
    SgExpression *e = sb::buildIntVal(si.offset);
    if (i >= 1) {
      // * pitch
      e = sb::buildMultiplyOp(e,
                              sb::buildArrowExp(sb::buildVarRefExp(gv),
                                                sb::buildVarRefExp("pitch")));
    }
    if (i >= 2) {
      // * local_size[1]
      e = sb::buildMultiplyOp(
          e,
          sb::buildArrowExp(sb::buildVarRefExp(gv),
                            sb::buildPntrArrRefExp(
                                sb::buildVarRefExp("local_size"),
                                sb::buildIntVal(1))));
    }
    if (i >= 3) {
      LOG_ERROR() << "Not supported dimension\n";
      PSAbort(1);
    }
    if (offset) {
      offset = sb::buildAddOp(offset, e);
    } else {
      offset = e;
    }
  }
  return offset;
}

static void CopyAttr(SgExpression *e1, SgExpression *e2) {
  AstAttribute *attr;
  attr = e1->getAttribute(StencilIndexAttribute::name);
  e2->setAttribute(StencilIndexAttribute::name, attr);
  attr = e1->getAttribute(GridCallAttribute::name);
  e2->setAttribute(GridCallAttribute::name, attr);
}

void MPIOpenCLOptimizer::GridPreCalcAddr(SgFunctionDeclaration *func) {
  LOG_INFO() << "Optimizing grid get address calculation\n";
  TranslationContext &tx = *trans_.tx_;
  Rose_STL_Container<SgNode*> calls =
      NodeQuery::querySubTree(func, V_SgFunctionCallExp);
  typedef map<SgInitializedName*, vector<SgFunctionCallExp*> > MapType;
  //map<SgInitializedName*, vector<SgFunctionCallExp*> > grid_gets;
  MapType grid_gets;
  FOREACH (it, calls.begin(), calls.end()) {
    SgFunctionCallExp *fc = isSgFunctionCallExp(*it);
    LOG_INFO() << "Call: " << fc->unparseToString() << "\n";

    GridCallAttribute *gca = static_cast<GridCallAttribute*>(
        fc->getAttribute(GridCallAttribute::name));
    // if null, not a grid call
    if (!gca) continue;
    SgInitializedName *grid_var = gca->grid_var();
    if (!gca->IsGet()) continue;    
    LOG_INFO() << "grid get call: " << fc->unparseToString()
               << ", grid var: " << grid_var->unparseToString() << "\n";

    MapType::iterator it = grid_gets.find(grid_var);
    if (it == grid_gets.end()) {
      MapType::mapped_type v;
      grid_gets.insert(make_pair(grid_var, v));
      it = grid_gets.find(grid_var);
    }
    MapType::mapped_type &v = it->second;
    v.push_back(fc);
  }

  // collect the index parameters
  SgInitializedNamePtrList indices;
  int nd = GetIndexParams(func, indices);
  LOG_INFO() << "Assuming " << nd << "-d function\n";
  SgFunctionDefinition *func_def = func->get_definition();

  FOREACH (gv_it, grid_gets.begin(), grid_gets.end()) {
    SgInitializedName *gv = gv_it->first;
    const GridSet *gs = tx.findGrid(gv);
    PSAssert(gs);
    GridType *gt = (*gs->begin())->getType();
    MapType::mapped_type &calls = gv_it->second;
    // if the grid is accessed just once, optimization is not applied
    if (calls.size() == 1) continue;

    // 1. create a grid_get_addr call with zero offset
    SgFunctionSymbol *get_addr_symbol =
        calls.front()->getAssociatedFunctionSymbol();
    SgExprListExp *params = sb::buildExprListExp(sb::buildVarRefExp(gv));
    FOREACH (it, indices.begin(), indices.end()) {
      params->append_expression(sb::buildVarRefExp(*it));
    }
    SgFunctionCallExp *center =
        sb::buildFunctionCallExp(get_addr_symbol, params);
    // 2. create variable declaration to hold the addresss
    SgVariableDeclaration *center_addr = sb::buildVariableDeclaration(
        rose_util::generateUniqueName(func_def->get_body()),
        sb::buildPointerType(gt->getElmType()),
        sb::buildAssignInitializer(center), func_def->get_body());
    // 3. Insert the variable where all the get calls can resolve
    // variable reference. It's safe to put it to the very beginning
    // of the function.
    si::prependStatement(center_addr, func_def->get_body());
    // 4. Replace each call with the variable reference plus offset
    FOREACH (cit, calls.begin(), calls.end()) {
      SgFunctionCallExp *call = *cit;
      StencilIndexAttribute *sia =
          static_cast<StencilIndexAttribute*>(
              call->getAttribute(StencilIndexAttribute::name));
      SgExpression *addr =
          sb::buildAddOp(sb::buildVarRefExp(center_addr),
                         BuildOffset(gv, sia->stencil_index_list(), nd));
      CopyAttr(call, addr);
      si::replaceExpression(call, addr, true);
    }
  }
}

} // namespace translator
} // namespace physis
