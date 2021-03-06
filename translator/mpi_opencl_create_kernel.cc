// Licensed under the BSD license. See LICENSE.txt for more details.

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

void MPIOpenCLTranslator::BuildFunctionParamList(
    SgClassDefinition *param_struct_def,
    SgFunctionParameterList *&params,
    SgInitializedName *&grid_arg,
    SgInitializedName *&dom_arg)

{
  LOG_DEBUG() << "Building function parameter list\n";
  const SgDeclarationStatementPtrList &members =
      param_struct_def->get_members();

  // Ugly hack 1 !!
  SgType *sg_nulltype = sb::buildOpaqueType(" ", global_scope_);

  FOREACH(member, members.begin(), members.end()) {
    SgVariableDeclaration *member_decl = isSgVariableDeclaration(*member);
    const SgInitializedNamePtrList &vars = member_decl->get_variables();
    SgInitializedName *arg = sb::buildInitializedName(
        vars[0]->get_name(), vars[0]->get_type());
    SgType *type = arg->get_type();

    LOG_DEBUG() << "type: " << type->unparseToString() << "\n";
    if (Domain::isDomainType(type)) {
      // Domain type
      if (!dom_arg) {
#if 0        
        dom_arg =  
            sb::buildInitializedName(
                arg->get_name(), arg->get_type());
#else
        dom_arg = arg;
#endif
      }

      // Adding __PS_CL_ARG_EXPAND_ELEMENT_DOM_WITH_TYPE(dom)
      // Defined with macro !!
      std::string name_dom = arg->get_name().getString();
      std::string name_macro_dom = 
          "__PS_CL_ARG_EXPAND_ELEMENT_DOM_WITH_TYPE(" + name_dom + ")";
      // VERY ugly hack 2!!
      SgInitializedName *init_dom = 
          sb::buildInitializedName(name_macro_dom, sg_nulltype);
      si::appendArg(params, init_dom);
      continue; // Done for dom type

    } else if (GridType::isGridType(type)) {
      SgType *gt = BuildOnDeviceGridType(
          tx_->findGridType(type), 1);
      arg->set_type(gt);
      if (!grid_arg) {
        grid_arg = 
            sb::buildInitializedName(
                arg->get_name(), arg->get_type());
      }
      // skip the grid index
      ++member;

      // Adding __PS_CL_ARG_EXPAND_ELEMENT_G_WITH_TYPE(dom)
      // Defined with macro !!
      std::string name_grid = arg->get_name().getString();
      std::string name_macro_grid = 
          "__PS_CL_ARG_EXPAND_ELEMENT_G_WITH_TYPE";
      {
        GridType *gt = tx_->findGridType(type);
        SgType *ty = gt->getElmType();
        if (isSgTypeDouble(ty))
          name_macro_grid += "_DOUBLE";
      }
      name_macro_grid += "(";
      name_macro_grid += name_grid;
      name_macro_grid += ")";

      // VERY ugly hack 3!!
      SgInitializedName *init_grid = 
          sb::buildInitializedName(name_macro_grid, sg_nulltype);
      si::appendArg(params, init_grid);

      continue; // Done for dom type
    }
    // Add the rest
    si::appendArg(
        params,
        sb::buildInitializedName(arg->get_name(), arg->get_type()));
  }
  return;
}

void MPIOpenCLTranslator::add_macro_mark_function_inner_device(
    SgFunctionDeclaration *func
                                                               ){
  {
    LOG_INFO() << "Adding #ifdef << " << kernel_mode_macro() << " \n";
    std::string str_insert = "#endif /* #ifndef ";
    str_insert += kernel_mode_macro();
    str_insert += "*/\n";
    str_insert += "#ifdef ";
    str_insert += kernel_mode_macro();
    si::attachArbitraryText(
        func,
        str_insert,
        PreprocessingInfo::before
                            );
    str_insert = "#endif /* #ifdef ";
    str_insert += kernel_mode_macro();
    str_insert += "*/";
    si::attachArbitraryText(
        func,
        str_insert,
        PreprocessingInfo::after
                            );
    str_insert = "#ifndef ";
    str_insert += kernel_mode_macro();
    si::attachArbitraryText(
        func,
        str_insert,
        PreprocessingInfo::after
                            );
  }
}

SgFunctionDeclaration *MPIOpenCLTranslator::BuildRunKernel(
    StencilMap *stencil
                                                           )
{
  // Initialize
  SgFunctionParameterList *params = sb::buildFunctionParameterList();

  // Stencils
  SgClassDefinition *param_struct_def = stencil->GetStencilTypeDefinition();
  PSAssert(param_struct_def);

  SgInitializedName *grid_arg = NULL;
  SgInitializedName *dom_arg = NULL;

  // add offset for process
  for (int i = 0; i < stencil->getNumDim()-1; ++i) {
    si::appendArg(
        params,
        sb::buildInitializedName("offset" + toString(i), sb::buildLongType()));
  }

  // Add dom, grid type to params
  BuildFunctionParamList(param_struct_def, params, grid_arg, dom_arg);

  PSAssert(dom_arg);
  LOG_INFO() << "Declaring and defining function named "
             << stencil->getRunName() << "\n";
  SgFunctionDeclaration *run_func =
      sb::buildDefiningFunctionDeclaration(stencil->getRunName(),
                                           sb::buildVoidType(),
                                           params, global_scope_);
  
  si::attachComment(run_func, "Generated by " + string(__FUNCTION__));

  // set body
  SgBasicBlock *func_body = BuildRunKernelBody(stencil, dom_arg);
  si::appendStatement(func_body, run_func->get_definition());

  // Add "__kernel" modifier to __PSStencilRun_kernel
  SgFunctionModifier &modifier = run_func->get_functionModifier();
  modifier.setOpenclKernel();

  // Add macro and mark this function as inner device
  add_macro_mark_function_inner_device(run_func);

  return run_func;
}

// This is almost equivalent as CUDATranslator::BuildRunKernel, except
// for having offset.

SgFunctionDeclaration *MPIOpenCLTranslator::BuildRunInteriorKernel(
    StencilMap *stencil) {

  if (!flag_mpi_overlap_) return NULL;
  
  SgFunctionParameterList *params = sb::buildFunctionParameterList();
  SgClassDefinition *param_struct_def = stencil->GetStencilTypeDefinition();
  PSAssert(param_struct_def);

  SgInitializedName *grid_arg = NULL;
  SgInitializedName *dom_arg = NULL;

  // add offset for process
  for (int i = 0; i < stencil->getNumDim()-1; ++i) {
    si::appendArg(
        params,
        sb::buildInitializedName("offset" + toString(i), sb::buildLongType()));
  }

  // Add dom, grid type to params
  BuildFunctionParamList(param_struct_def, params, grid_arg, dom_arg);
  PSAssert(dom_arg);

  string func_name = stencil->getRunName() + inner_prefix_;
  LOG_INFO() << "Declaring and defining function named "
             << func_name << "\n";
  SgFunctionDeclaration *run_func =
      sb::buildDefiningFunctionDeclaration(func_name,
                                           sb::buildVoidType(),
                                           params, global_scope_);
  
  si::attachComment(run_func, "Generated by " + string(__FUNCTION__));
  SgBasicBlock *func_body = BuildRunInteriorKernelBody(stencil, dom_arg);
  si::appendStatement(func_body, run_func->get_definition());

  // Add "__kernel" modifier to __PSStencilRun_kernel
  SgFunctionModifier &modifier = run_func->get_functionModifier();
  modifier.setOpenclKernel();

  // Add macro and mark this function as inner device
  add_macro_mark_function_inner_device(run_func);

  return run_func;
}


SgFunctionDeclarationPtrVector
MPIOpenCLTranslator::BuildRunBoundaryKernel(StencilMap *stencil) {
  std::vector<SgFunctionDeclaration*> run_funcs;
  if (!flag_mpi_overlap_) return run_funcs;;
  if (flag_multistream_boundary_)
    return BuildRunMultiStreamBoundaryKernel(stencil);

  SgFunctionParameterList *params = sb::buildFunctionParameterList();
  SgClassDefinition *param_struct_def = stencil->GetStencilTypeDefinition();
  PSAssert(param_struct_def);

  SgInitializedName *grid_arg = NULL;
  SgInitializedName *dom_arg = NULL;

  // add offset for process
  for (int i = 0; i < stencil->getNumDim()-1; ++i) {
    si::appendArg(
        params,
        sb::buildInitializedName(
            "offset" + toString(i), sb::buildLongType()));
  }

  si::appendArg(
      params,
      sb::buildInitializedName(
          boundary_kernel_width_name_, sb::buildLongType()));
  
  // Add dom, grid type to params
  BuildFunctionParamList(param_struct_def, params, grid_arg, dom_arg);
  PSAssert(grid_arg);
  PSAssert(dom_arg);

  LOG_INFO() << "Declaring and defining function named "
             << stencil->getRunName() << "\n";
  SgFunctionDeclaration *run_func =
      sb::buildDefiningFunctionDeclaration(stencil->getRunName()
                                           + boundary_suffix_,
                                           sb::buildVoidType(),
                                           params, global_scope_);
  
  si::attachComment(run_func, "Generated by " + string(__FUNCTION__));
  SgBasicBlock *func_body = BuildRunBoundaryKernelBody(
      stencil, dom_arg);
  si::appendStatement(func_body, run_func->get_definition());

  // Add "__kernel" modifier to __PSStencilRun_kernel
  SgFunctionModifier &modifier = run_func->get_functionModifier();
  modifier.setOpenclKernel();

  // Add macro and mark this function as inner device
  add_macro_mark_function_inner_device(run_func);

  run_funcs.push_back(run_func);
  return run_funcs;
}

} // namespace translator
} // namespace physis
