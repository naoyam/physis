#include "translator/mpi_opencl_translator.h"

#include "translator/translation_context.h"
#include "translator/translation_util.h"
#include "translator/mpi_runtime_builder.h"
#include "translator/mpi_opencl_runtime_builder.h"
#include "translator/reference_runtime_builder.h"
#include "translator/SageBuilderEx.h"
#include "translator/rose_util.h"
#include "translator/runtime_builder.h"
#include "translator/mpi_opencl_optimizer.h"
#include "translator/rose_ast_attribute.h"

#include <cstring>
#include <string>

namespace sb = SageBuilder;
namespace si = SageInterface;


namespace physis {
namespace translator {

void MPIOpenCLTranslator::translateKernelDeclaration(
    SgFunctionDeclaration *node) {
  LOG_DEBUG() << "Translating to MPI-OpenCL kernel\n";
  //node->get_functionModifier().setCudaDevice();

  SgFunctionDeclaration *node_pos;

  // e.g., PSGrid3DFloat -> __PSGrid3DFloatDev *
  Rose_STL_Container<SgNode*> exps =
      NodeQuery::querySubTree(node, V_SgInitializedName);
  FOREACH (it, exps.begin(), exps.end()) {
    SgInitializedName *exp = isSgInitializedName(*it);
    PSAssert(exp);
    SgType *cur_type = exp->get_type();
    GridType *gt = tx_->findGridType(cur_type);
    // not a grid type
    if (!gt) continue;
    SgType *new_type = sb::buildPointerType(
        BuildOnDeviceGridType(gt, 1));
    exp->set_type(new_type);
  }
  
  Rose_STL_Container<SgNode*> gdim_calls =
      NodeQuery::querySubTree(node, V_SgFunctionCallExp);
  SgFunctionSymbol *gdim_dev =
      si::lookupFunctionSymbolInParentScopes("__PSGridDimDev");
  FOREACH (it, gdim_calls.begin(), gdim_calls.end()) {
    SgFunctionCallExp *fc = isSgFunctionCallExp(*it);
    PSAssert(fc);
    if (rose_util::getFuncName(fc) != PS_GRID_DIM_NAME)     
      continue;
    fc->set_function(sb::buildFunctionRefExp(gdim_dev));
  }

  node_pos = node;

  // Rename the call to kernel function to __PS_opencl_kernel
  if (1) {
    // Search function call
    SgNodePtrList calls =
        NodeQuery::querySubTree(node_pos, V_SgFunctionCallExp);
    FOREACH(it, calls.begin(), calls.end()) {
      SgFunctionCallExp *callexp = isSgFunctionCallExp(*it);
      if (!callexp) continue;

      // If it is not a call to device function, continue
      SgFunctionDeclaration *calldec = callexp->getAssociatedFunctionDeclaration();
      if (! tx_->isKernel(calldec)) continue;

      // Rename caller to __PS_opencl_kernel if the called
      // function name is "kernel" (OpenCL won't allow this)
      std::string oldfuncname = calldec->get_name().getString();
      const char *oldfuncname_c = oldfuncname.c_str();

      if (strcmp(oldfuncname_c, "kernel")) // No need to rename
        continue;

      std::string newfuncname = opencl_trans_->name_new_kernel(oldfuncname);
      LOG_DEBUG() << "Calling to kernel function " << oldfuncname << " found.\n";
      LOG_DEBUG() << "Changing the call to " << newfuncname << ".\n";
      SgName newsgname(newfuncname);
      callexp->set_function(sb::buildFunctionRefExp(newsgname));
    } // FOREACH(it, calls.begin(), calls.end())
  }

  // And rename the function name of this node if the name is
  // "kernel"

  do {
    std::string oldkernelname = node->get_name().getString();
    const char *oldkernelname_c = oldkernelname.c_str();
    if (strcmp(oldkernelname_c, "kernel")) // No need to rename
      break;
    std::string newkernelname = opencl_trans_->name_new_kernel(oldkernelname);

    LOG_DEBUG() << "Changing the name of the function declaration: from " 
                << oldkernelname << " to " << newkernelname << ".\n";
    // Create declaration copy, and rename it
    SgFunctionDeclaration *node_new = 
        rose_util::CloneFunction(node, newkernelname);

    // Replace the statement, and add the symbol to the scope if missing
    si::replaceStatement(node, node_new);
    SgFunctionSymbol *newkernelsym = global_scope_->lookup_function_symbol(newkernelname, node_new->get_type());
    if (!newkernelsym) {
      newkernelsym = new SgFunctionSymbol(node_new);
      global_scope_->insert_symbol(newkernelname, newkernelsym);
    }
    node_pos = node_new;
  } while(0);

  add_macro_mark_function_inner_device(node_pos);

  if (flag_mpi_overlap_){
    SgFunctionDeclaration *node_add = BuildInteriorKernel(node);
    add_macro_mark_function_inner_device(node_add);
    si::insertStatementBefore(node_pos, node_add, false);
  }

  if (flag_multistream_boundary_) {
    std::vector<SgFunctionDeclaration*> boundary_kernels
        = BuildBoundaryKernel(node);
    FOREACH (it, boundary_kernels.begin(), boundary_kernels.end()) {
      SgFunctionDeclaration *node_add = *it;
      add_macro_mark_function_inner_device(node_add);
      si::insertStatementBefore(node_pos, node_add, false);
    }
  }
  return;
}


bool MPIOpenCLTranslator::translateGetKernel(
    SgFunctionCallExp *node,
    SgInitializedName *gv,
    bool is_periodic
                                             ) {
  // 
  // *((gt->getElmType())__PSGridGetAddressND(g, x, y, z))

  GridType *gt = tx_->findGridType(gv->get_type());
  int nd = gt->getNumDim();
  SgScopeStatement *scope = getContainingScopeStatement(node);  
  SgVarRefExp *g = sb::buildVarRefExp(gv->get_name(), scope);
  
  string get_address_name = get_addr_name_ +  GetTypeDimName(gt);
  string get_address_no_halo_name = get_addr_no_halo_name_ +  GetTypeDimName(gt);
  SgFunctionRefExp *get_address = NULL;
  const StencilIndexList *sil = tx_->findStencilIndex(node);
  PSAssert(sil);
  LOG_DEBUG() << "Stencil index: " << *sil << "\n";
  if (StencilIndexSelf(*sil, nd)) {
    get_address = sb::buildFunctionRefExp(get_address_no_halo_name,
                                          global_scope_);
  } else if (StencilIndexRegularOrder(*sil, nd)) {
    for (int i = 0; i < nd; ++i) {
      int offset = (*sil)[i].offset;
      if (!offset) continue;
      string method_name = 
          get_address_name + "_" +
          toString(i) + "_" + ((offset < 0) ? "bw" : "fw");
      LOG_INFO() << "Using " << method_name << "\n";
      get_address = sb::buildFunctionRefExp(method_name, global_scope_);
      break;
    }
  } else {
    get_address = sb::buildFunctionRefExp(get_address_name,
                                          global_scope_);
  }
  SgExprListExp *args = sb::buildExprListExp(g);
  FOREACH (it, node->get_args()->get_expressions().begin(),
           node->get_args()->get_expressions().end()) {
    args->append_expression(si::copyExpression(*it));
  }

  SgFunctionCallExp *get_address_exp
      = sb::buildFunctionCallExp(get_address, args);
  // refactoring: merge the two attributes
  get_address_exp->setAttribute(StencilIndexAttribute::name,
                                new StencilIndexAttribute(*sil));
  get_address_exp->setAttribute(GridCallAttribute::name,
                                node->getAttribute(GridCallAttribute::name));
  SgExpression *x = sb::buildPointerDerefExp(get_address_exp);
  si::replaceExpression(node, x);
  return true;
}


} // namespace translator
} // namespace physis
